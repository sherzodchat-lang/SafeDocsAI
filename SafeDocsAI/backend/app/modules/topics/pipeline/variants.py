"""Шесть способов убрать из векторов язык и жанр — и сравнение их между собой.

Предыдущий прогон установил три вещи. Кластеры ложатся на язык и жанр, а не на
тему (при k=20 ARI против языка +0.415 против темы +0.063). Тематический сигнал
в векторах при этом есть (ближайший сосед той же темы у 68% документов при
случайном уровне 6%). Глобальное центрирование не помогает — анизотропии нет.

Отсюда задача этого модуля: подавить не «всё сразу», а именно те две оси,
которые перекрывают тему, и посмотреть, проявится ли она. Способов ровно два, и
каждый применён к каждой оси:

  * ВЫЧЕСТЬ — центрирование по группе. Корпус остаётся единым, кластер может
    объединять документы разных языков;
  * РАЗДЕЛИТЬ — своя модель на каждую группу. Ось устранена по построению,
    ценой того, что одна тема на трёх языках даёт три разных кластера.

Что здесь сделано, чтобы сравнение не обмануло.

Средние по группам считаются ТОЛЬКО на train. Среднее по всему корпусу
подглядывает в test: положение тестового документа относительно центра его
группы зависело бы от него самого.

Каждый вариант считается против трёх разметок сразу — тема, язык,
происхождение. Иначе «ось подавлена» проверить нечем: ARI по теме может
подрасти и оттого, что нарезка случайно оказалась удачнее.

Каждый вариант считает согласие ближайших соседей по теме — величину, не
зависящую ни от k, ни от K-means вообще. Она отвечает на вопрос, который по
одному ARI не решается: улучшилась ГЕОМЕТРИЯ или мы просто удачнее нарезали.
Если ARI вырос, а согласие соседей стоит на месте, преобразование ни при чём.

Число кластеров сравнивается честно. Разделяющие варианты дают союз шести (или
трёх) разбиений, то есть 60 кластеров против 20 у общих вариантов, а ARI за
дробление наказывает. Поэтому общие варианты считаются ещё и при 60 кластерах —
в отчёте это отдельный режим, и строки таблицы сопоставляются с одинаковым
числом кластеров.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from app.modules.topics.kmeans import KMeans, choose_k, l2_normalize
from app.modules.topics.metrics import inertia, silhouette_score
from app.modules.topics.pipeline.dataset import Corpus
from app.modules.topics.pipeline.experiment import RANDOM_STATE, external_scores
from app.modules.topics.pipeline.model_io import ClusterTopic, TopicModel, dominant_topics
from app.modules.topics.pipeline.transforms import ClusterRouting, GroupCentering, group_keys

SPLIT_NAMES = ("train", "validation", "test")

# Ячейки «язык x жанр» — самая мелкая сетка, по которой считаются и слои, и
# контрольные разрезы. Держится в одном месте: разъехавшиеся определения ячейки
# в двух местах дали бы две таблицы, которые нельзя сравнить.
CELL_FIELDS = ("language", "dataset_origin")

# Сетка перебора k. Та же, что в основном прогоне, — иначе разница между
# вариантами включала бы разницу сеток.
K_GRID = (2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40)


@dataclass(frozen=True)
class VariantSpec:
    """Описание варианта: что вычитаем и по чему делим.

    center_by пустой и split_by пустой — это базовый вариант «как есть». Он
    считается тем же кодом, а не переписывается отдельно: иначе к разнице между
    базой и вариантом добавилась бы разница двух реализаций.
    """

    name: str
    center_by: tuple[str, ...]
    split_by: tuple[str, ...]
    title: str

    @property
    def is_split(self) -> bool:
        return bool(self.split_by)


VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("baseline", (), (), "как есть"),
    VariantSpec("per_language", (), ("language",), "своя модель на каждый язык"),
    VariantSpec("centered_language", ("language",), (), "минус среднее языка"),
    VariantSpec("centered_origin", ("dataset_origin",), (), "минус среднее жанра"),
    VariantSpec("centered_cell", CELL_FIELDS, (), "минус среднее ячейки язык x жанр"),
    VariantSpec("per_cell", (), CELL_FIELDS, "своя модель в каждой ячейке"),
)


# --- подготовка данных -------------------------------------------------------


def fit_transform(
    spec: VariantSpec, splits: dict[str, Corpus], data: dict[str, np.ndarray]
) -> tuple[GroupCentering | None, dict[str, np.ndarray]]:
    """Средние по train, применение ко всем разбиениям.

    Возвращает и само преобразование: без него сохранённую модель нельзя будет
    применить к новому документу, а отчёт не сможет сказать, что именно с
    векторами делали.
    """
    if not spec.center_by:
        return None, {name: data[name] for name in data}
    transform = GroupCentering.fit(
        splits["train"].documents, data["train"], spec.center_by
    )
    return transform, {
        name: transform.apply(splits[name].documents, data[name]) for name in data
    }


def strata_of(corpus: Corpus, fields: Sequence[str]) -> dict[str, list[int]]:
    """Индексы документов по группам, с сохранением исходного порядка."""
    keys = group_keys(corpus.documents, fields)
    result: dict[str, list[int]] = {}
    for index, key in enumerate(keys):
        result.setdefault(key, []).append(index)
    return {key: result[key] for key in sorted(result)}


def n_topics_of(corpus: Corpus, indices: Sequence[int] | None = None) -> int:
    topics = corpus.labels("topic_id")
    if indices is None:
        return len(set(topics))
    return len({topics[index] for index in indices})


def k_grid_for(n_train: int, n_topics: int) -> tuple[int, ...]:
    """Сетка k для одного слоя.

    Два ограничения. Сверху — тройное число настоящих тем: дальше вопрос «а не
    выбрали ли вы слишком грубое k» уже закрыт, а время перебора растёт. Ещё
    выше — четверть обучающей выборки слоя: в ячейке синтетики на один язык
    всего 140 документов, и k=40 означало бы кластеры по три-четыре документа,
    у которых силуэт меряет уже не структуру, а шум.
    """
    limit = min(3 * n_topics, max(2, n_train // 4))
    values = tuple(k for k in K_GRID if 2 <= k <= limit)
    return values or (2,)


# --- обучение ----------------------------------------------------------------


@dataclass
class VariantFit:
    """Обученный вариант: один K-means или несколько, сшитых в одну разметку.

    labels — глобальные номера кластеров, выровненные по документам каждого
    разбиения. У разделяющих вариантов номера сдвинуты так, что кластеры разных
    слоёв не пересекаются: иначе ARI считал бы русский кластер 0 и таджикский
    кластер 0 одной группой, и разделяющий вариант выглядел бы лучше, чем есть.
    """

    spec: VariantSpec
    centroids: np.ndarray
    cluster_groups: tuple[str, ...] | None
    cluster_topics: tuple[ClusterTopic, ...]
    labels: dict[str, np.ndarray]
    per_stratum: dict[str, dict[str, Any]] = field(default_factory=dict)
    total_inertia: float = 0.0
    n_iter: int = 0


def fit_variant(
    spec: VariantSpec,
    splits: dict[str, Corpus],
    tdata: dict[str, np.ndarray],
    k_of: dict[str, int],
    *,
    random_state: int = RANDOM_STATE,
) -> VariantFit:
    """Обучение варианта на train и назначение меток всем разбиениям.

    k_of — число кластеров по слоям: ключ "" для общего варианта, ключ группы
    для разделяющего. Передаётся снаружи, потому что один и тот же вариант
    считается в нескольких режимах k, и выбор k — предмет отчёта, а не детали
    обучения.
    """
    strata = (
        strata_of(splits["train"], spec.split_by) if spec.is_split else {"": list(range(len(splits["train"])))}
    )

    centroid_blocks: list[np.ndarray] = []
    cluster_groups: list[str] = []
    labels = {name: np.full(len(splits[name]), -1, dtype=np.int64) for name in SPLIT_NAMES}
    per_stratum: dict[str, dict[str, Any]] = {}
    total_inertia = 0.0
    max_iter_used = 0
    offset = 0

    for key, train_index in strata.items():
        k = k_of[key]
        X_train = tdata["train"][train_index]
        model = KMeans(n_clusters=k, random_state=random_state, normalize=True).fit(X_train)
        assert model.centroids_ is not None and model.labels_ is not None
        centroid_blocks.append(model.centroids_)
        cluster_groups += [key] * model.centroids_.shape[0]
        total_inertia += float(model.inertia_ or 0.0)
        max_iter_used = max(max_iter_used, int(model.n_iter_ or 0))

        stratum_sizes: dict[str, int] = {}
        for name in SPLIT_NAMES:
            index = (
                strata_of(splits[name], spec.split_by).get(key, [])
                if spec.is_split
                else list(range(len(splits[name])))
            )
            stratum_sizes[name] = len(index)
            if not index:
                continue
            predicted = model.predict(tdata[name][index])
            labels[name][index] = predicted + offset

        per_stratum[key] = {
            "k": k,
            "n_true_topics": n_topics_of(splits["train"], train_index),
            "n_documents": stratum_sizes,
            "cluster_range": [offset, offset + k],
            "inertia": float(model.inertia_ or 0.0),
            "n_iter": int(model.n_iter_ or 0),
        }
        offset += k

    for name in SPLIT_NAMES:
        if (labels[name] < 0).any():
            # Документ, чья группа не встретилась при обучении, остался бы без
            # метки, а метрики посчитались бы по -1 как по отдельному кластеру.
            missing = int((labels[name] < 0).sum())
            raise ValueError(
                f"{spec.name}: в {name} осталось {missing} документов без модели "
                "(их группа не встречалась в train)"
            )

    centroids = np.vstack(centroid_blocks)
    topics = dominant_topics(
        labels["train"],
        splits["train"].labels("topic_id"),
        # Одноязычное основное имя: колонка topic в корпусе переведена вместе с
        # документом, и подпись «как у первого документа кластера» доставалась
        # на случайном языке.
        splits["train"].localized_labels("en"),
        centroids.shape[0],
        # Переводы — часть артефакта победителя: именно этот файл уезжает в
        # продукт, где английских подписей показывать некому (интерфейс
        # переведён на ru и tg).
        splits["train"].localized_labels("ru"),
        splits["train"].localized_labels("tg"),
    )
    return VariantFit(
        spec=spec,
        centroids=centroids,
        cluster_groups=tuple(cluster_groups) if spec.is_split else None,
        cluster_topics=topics,
        labels=labels,
        per_stratum=per_stratum,
        total_inertia=total_inertia,
        n_iter=max_iter_used,
    )


# --- оценка ------------------------------------------------------------------


def _internal(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> dict[str, Any]:
    data = l2_normalize(np.asarray(X, dtype=np.float64))
    distinct = int(len(np.unique(labels)))
    silhouette = (
        float(silhouette_score(data, labels, sample_size=None))
        if 2 <= distinct < data.shape[0]
        else float("nan")
    )
    return {
        "inertia": float(inertia(data, labels, centroids)),
        "silhouette": silhouette,
        "n_clusters_used": distinct,
        "n_samples": int(data.shape[0]),
    }


def evaluate_fit(
    fit: VariantFit,
    splits: dict[str, Corpus],
    tdata: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Все числа одного обученного варианта: внутренние, внешние, по ячейкам."""
    report: dict[str, Any] = {
        "n_clusters_total": int(fit.centroids.shape[0]),
        "inertia_train_sum": fit.total_inertia,
        "per_stratum": fit.per_stratum,
        "splits": {},
        "by_cell": {},
    }
    for name in SPLIT_NAMES:
        labels = fit.labels[name]
        block = _internal(tdata[name], labels, fit.centroids)
        block["external"] = external_scores(labels, splits[name])
        report["splits"][name] = block

    # Тот же вариант, но метрики посчитаны внутри каждой ячейки «язык x жанр».
    # Нужно, чтобы сравнивать общие варианты с разделяющими на равных: у
    # разделяющего по ячейкам это его собственный слой, у общего — подмножество
    # его единой разметки, и вопрос к обоим один: находятся ли темы там, где ни
    # язык, ни жанр объяснить разбиение уже не могут.
    for name in SPLIT_NAMES:
        cells: dict[str, Any] = {}
        for key, index in strata_of(splits[name], CELL_FIELDS).items():
            subset = Corpus(tuple(splits[name].documents[i] for i in index))
            sub_labels = fit.labels[name][index]
            scores = external_scores(sub_labels, subset)
            cells[key] = {
                "n_documents": len(index),
                "n_true_topics": len(set(subset.labels("topic_id"))),
                "n_clusters_used": int(len(np.unique(sub_labels))),
                "ari_topic": scores["topic_id"]["ari"],
                "purity_topic": scores["topic_id"]["purity"],
            }
        report["by_cell"][name] = {
            "cells": cells,
            "mean_ari_topic": float(np.mean([value["ari_topic"] for value in cells.values()])),
            "mean_purity_topic": float(
                np.mean([value["purity_topic"] for value in cells.values()])
            ),
        }
    return report


# --- геометрия ---------------------------------------------------------------


def neighbour_agreement_scoped(
    corpus: Corpus, X: np.ndarray, scope_fields: Sequence[str]
) -> dict[str, Any]:
    """Согласие ближайших соседей, где сосед ищется только внутри своей группы.

    scope_fields пустой — сосед ищется по всему корпусу, это та самая величина
    0.683 из первого отчёта. Непустой — соседи ограничены той же группой, и
    число отвечает на вопрос «различает ли модель темы ВНУТРИ языка», к которому
    общая величина отношения не имеет: там ближайшим соседом почти всегда
    оказывается документ того же языка просто потому, что язык — ось сильнее.

    Случайный уровень считается в той же области, что и само согласие: внутри
    группы состав тем другой, и сравнивать долю по ячейке с общекорпусной
    случайностью было бы подменой.
    """
    data = l2_normalize(np.asarray(X, dtype=np.float64))
    similarity = data @ data.T
    np.fill_diagonal(similarity, -np.inf)
    if scope_fields:
        keys = np.asarray(group_keys(corpus.documents, scope_fields))
        similarity = np.where(keys[:, None] == keys[None, :], similarity, -np.inf)
    nearest = np.argmax(similarity, axis=1)
    reachable = np.isfinite(similarity[np.arange(similarity.shape[0]), nearest])

    result: dict[str, Any] = {}
    chance: dict[str, Any] = {}
    for field_name in ("topic_id", "language", "dataset_origin"):
        labels = np.asarray(corpus.labels(field_name))
        result[field_name] = float(np.mean(labels[nearest][reachable] == labels[reachable]))
        if scope_fields:
            keys = np.asarray(group_keys(corpus.documents, scope_fields))
            weighted = 0.0
            for key in np.unique(keys):
                inside = labels[keys == key]
                _, counts = np.unique(inside, return_counts=True)
                shares = counts / counts.sum()
                weighted += float(np.sum(shares**2)) * inside.size / labels.size
            chance[field_name] = weighted
        else:
            _, counts = np.unique(labels, return_counts=True)
            shares = counts / counts.sum()
            chance[field_name] = float(np.sum(shares**2))
    return {
        "scope": list(scope_fields),
        "n_compared": int(reachable.sum()),
        "same_label_as_nearest_neighbour": result,
        "chance_level": chance,
    }


def geometry_of(corpus: Corpus, X: np.ndarray) -> dict[str, Any]:
    """Согласие соседей в четырёх областях поиска сразу."""
    return {
        "global": neighbour_agreement_scoped(corpus, X, ()),
        "within_language": neighbour_agreement_scoped(corpus, X, ("language",)),
        "within_origin": neighbour_agreement_scoped(corpus, X, ("dataset_origin",)),
        "within_cell": neighbour_agreement_scoped(corpus, X, CELL_FIELDS),
    }


# --- прогон варианта целиком -------------------------------------------------


def search_k(
    spec: VariantSpec,
    splits: dict[str, Corpus],
    tdata: dict[str, np.ndarray],
    *,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Перебор k по каждому слою варианта.

    Силуэт на этом корпусе плоский (в первом прогоне 0.033..0.071 на всём
    диапазоне), и выбор по его максимуму — это выбор по шуму. Таблица всё равно
    считается: без неё утверждение «силуэт здесь не работает» было бы словом
    без числа, а не выводом.
    """
    strata = (
        strata_of(splits["train"], spec.split_by) if spec.is_split else {"": list(range(len(splits["train"])))}
    )
    result: dict[str, Any] = {}
    for key, index in strata.items():
        n_topics = n_topics_of(splits["train"], index)
        grid = k_grid_for(len(index), n_topics)
        search = choose_k(
            tdata["train"][index],
            grid,
            random_state=random_state,
            normalize=True,
            silhouette_sample_size=None,
        )
        result[key] = {
            "k_values": list(grid),
            "n_true_topics": n_topics,
            "best_k_by_silhouette": int(search.best_k),
            "elbow_k": int(search.elbow_k),
            "table": [
                {
                    "k": score.k,
                    "inertia": float(score.inertia),
                    "silhouette": float(score.silhouette),
                    "n_iter": int(score.n_iter),
                }
                for score in search.scores
            ],
        }
    return result


def k_regimes(
    spec: VariantSpec,
    splits: dict[str, Corpus],
    k_search: dict[str, Any],
    *,
    matched_clusters: int,
) -> dict[str, dict[str, int]]:
    """Режимы выбора k, в которых считается каждый вариант.

    true_k — k равно числу настоящих тем слоя. Только он отвечает на вопрос
    «нашлись ли эти темы»: при другом k ARI мешает ответ с наказанием за
    неверную грубость разбиения.

    selected_k — k по максимуму силуэта. Оставлен не как рекомендация, а как
    улика: он выбирает шум, и это видно только рядом с true_k.

    matched — одинаковое ОБЩЕЕ число кластеров у всех вариантов. Без него
    разделяющие варианты сравнивались бы с общими при 60 кластерах против 20, а
    ARI за дробление наказывает, и разница включала бы этот штраф.
    """
    strata = (
        strata_of(splits["train"], spec.split_by) if spec.is_split else {"": list(range(len(splits["train"])))}
    )
    true_k = {key: n_topics_of(splits["train"], index) for key, index in strata.items()}
    selected = {key: int(k_search[key]["best_k_by_silhouette"]) for key in strata}

    regimes = {"true_k": true_k, "selected_k": selected}
    if sum(true_k.values()) != matched_clusters:
        # Разделить matched поровну между слоями нельзя: у ячеек разное число
        # тем. Делим пропорционально числу тем слоя — так сохраняется
        # соотношение, ради которого true_k и выбирался.
        total_topics = sum(true_k.values())
        share = {key: max(2, round(matched_clusters * value / total_topics)) for key, value in true_k.items()}
        regimes["matched"] = share
    return regimes


def run_variant(
    spec: VariantSpec,
    splits: dict[str, Corpus],
    data: dict[str, np.ndarray],
    full: Corpus,
    full_X: np.ndarray,
    *,
    matched_clusters: int,
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, Any], dict[str, VariantFit], GroupCentering | None]:
    """Один вариант целиком: преобразование, перебор k, режимы, геометрия."""
    started = time.monotonic()
    transform, tdata = fit_transform(spec, splits, data)
    full_transformed = (
        transform.apply(full.documents, full_X) if transform is not None else full_X
    )

    k_search = search_k(spec, splits, tdata, random_state=random_state)
    regimes = k_regimes(spec, splits, k_search, matched_clusters=matched_clusters)

    fits: dict[str, VariantFit] = {}
    regime_reports: dict[str, Any] = {}
    for regime, k_of in regimes.items():
        fit = fit_variant(spec, splits, tdata, k_of, random_state=random_state)
        fits[regime] = fit
        regime_reports[regime] = evaluate_fit(fit, splits, tdata)
        regime_reports[regime]["k_per_stratum"] = dict(k_of)

    report = {
        "name": spec.name,
        "title": spec.title,
        "center_by": list(spec.center_by),
        "split_by": list(spec.split_by),
        "transform": transform.meta() if transform is not None else None,
        "requires_fields_at_apply_time": sorted(set(spec.center_by) | set(spec.split_by)),
        "k_search": k_search,
        "regimes": regime_reports,
        # Геометрия считается на преобразованных векторах всего корпуса и от k
        # не зависит вовсе: это проверка, изменилось ли само пространство.
        "geometry": geometry_of(full, full_transformed),
        "seconds": round(time.monotonic() - started, 1),
    }
    return report, fits, transform


def build_model(
    fit: VariantFit,
    transform: GroupCentering | None,
    embedding_model: str,
    params: dict[str, Any],
) -> TopicModel:
    """Сохраняемая модель варианта — вместе со всем, без чего её не применить."""
    routing = (
        ClusterRouting(fields=fit.spec.split_by, cluster_groups=fit.cluster_groups)
        if fit.cluster_groups is not None
        else None
    )
    return TopicModel(
        centroids=fit.centroids,
        embedding_model=embedding_model,
        normalize=True,
        cluster_topics=fit.cluster_topics,
        params=params,
        transform=transform,
        routing=routing,
    )
