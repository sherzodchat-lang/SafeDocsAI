#!/usr/bin/env python3
"""Обучение модели тем на таджикском корпусе с рубриками.

Отличие от cluster_topics_variants.py, который обучал действующую модель, — не
в алгоритме K-means, он тот же и не менялся. Отличаются две вещи, и обе видны
пользователю.

ПЕРВОЕ: КОРПУС. Прежняя модель училась на 840 синтетических корпоративных
текстах и 1438 выдержках Википедии. Боевые документы системы — налоговый кодекс
и паёмы президента, и тем «Законодательство», «Послания» в модели не было
вовсе: каждый реальный документ приписывался к ближайшему из двадцати чужих
центров, а три паёма разъехались по трём разным темам. Здесь корпус — таджикские
новости, законы и указы, размеченные редакционными рубриками, то есть тот же
жанр и тот же язык, что у боевых документов.

ВТОРОЕ: ПРЕОБРАЗОВАНИЕ. Прежнее вычитало среднее ячейки «язык × жанр» и потому
требовало эти признаки знать. Языку в бою соответствие есть, жанру — нет, и его
приходилось назначать допущением «боевой документ всегда real», причём среднее
ветки real посчитано на Википедии. Здесь ось языка снимается проекцией на
главные оси: первые несколько направлений наибольшего разброса многоязычного
эмбеддинга — это и есть язык с регистром. Документ при этом не спрашивают ни о
чём.

ЧТО ЗДЕСЬ ВАЖНО НЕ ПЕРЕПУТАТЬ.

*Главные оси считаются на train И НА НЕРАЗМЕЧЕННОМ КОТЛЕ, но не на validation и
не на test.* Котёл — 1213 документов sputnik, чьи разделы темами не являются;
меток им не нужно, а 659 русских текстов среди них — единственное, что позволяет
первой компоненте поймать ось языка. Заглядывать при этом в отложенные выборки
нельзя: положение тестового документа относительно осей зависело бы от него
самого.

*Кластеры обучаются только на размеченном train.* Иначе кластер, целиком
состоящий из документов котла, не имел бы рубрики и назвать его было бы нечем.

*Все решения — по validation, все числа для защиты — по test.* Подбор drop,
keep, k, порога слияния и порога уверенности идёт по validation. Test читается
один раз, в конце.

    ./venv/bin/python cluster_topics_tj.py                 # полный прогон
    ./venv/bin/python cluster_topics_tj.py --quick         # без сетки вариантов
    ./venv/bin/python cluster_topics_tj.py --no-save       # ничего не писать
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from app.modules.topics.kmeans import KMeans, l2_normalize, squared_distances  # noqa: E402
from app.modules.topics.labeling import compose_labels, tokenize_all  # noqa: E402
from app.modules.topics.metrics import (  # noqa: E402
    adjusted_rand_index,
    purity,
    silhouette_score,
)
from app.modules.topics.pca import PrincipalAxes  # noqa: E402
from app.modules.topics.pipeline.dataset import load_full, load_splits  # noqa: E402
from app.modules.topics.pipeline.embeddings import EmbeddingCache  # noqa: E402
from app.modules.topics.pipeline.model_io import TopicModel, dominant_topics  # noqa: E402
from app.modules.topics.pipeline.rubrics import (  # noqa: E402
    RUBRIC_BY_CODE,
    UNLABELLED,
    rubric_names,
)

BACKEND_ROOT = Path(__file__).resolve().parent
CORPUS_ROOT = BACKEND_ROOT / "data" / "topics_tj"
DATA_DIR = CORPUS_ROOT / "data"
CACHE_PATH = CORPUS_ROOT / "embeddings.npz"
MODEL_PATH = CORPUS_ROOT / "topic_model_tj.npz"
REPORT_PATH = CORPUS_ROOT / "clustering_report_tj.json"

RANDOM_STATE = 42
N_INIT = 10

# Сетка перебора. drop — сколько первых главных осей снять (это оси языка и
# регистра), keep — в скольких измерениях работать дальше. Обе величины
# подбираются, а не назначаются: на прежнем размеченном корпусе оптимум был
# drop=3, keep=32, но там был другой язык и другой жанр, и переносить число без
# проверки значило бы повторить ровно ту ошибку, из-за которой всё и затевалось.
DROP_GRID = (0, 1, 2, 3, 4, 5)
KEEP_GRID = (16, 32, 64, 128)
K_GRID = (8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32)

# Сколько пар подвыборок берётся для оценки устойчивости. Устойчивость нужна
# потому, что силуэт на таких данных низок по модулю даже после проекции, и
# выбирать k по одному критерию — значит выбирать по шуму.
STABILITY_ROUNDS = 5
STABILITY_FRACTION = 0.8

# Косинус между центроидами, выше которого два кластера считаются одним и тем
# же. Значение подбирается по validation из этого набора; None означает «не
# сливать вовсе» и участвует в переборе наравне с остальными, чтобы слияние
# применялось только тогда, когда оно измеримо помогает.
MERGE_GRID = (None, 0.90, 0.85, 0.80, 0.75)

# Доля документов, которой модель имеет право сказать «не знаю». Порог запаса
# подбирается так, чтобы отказов было примерно столько: пустое поле честнее
# уверенной ошибки, но модель, отказывающаяся от половины корпуса, бесполезна.
TARGET_UNCLEAR_SHARE = 0.10


def say(message: str = "") -> None:
    print(message, flush=True)


# --- данные ------------------------------------------------------------------


def load_matrices(data_dir: Path, cache_path: Path):
    """Векторы и разметка по разбиениям плюс неразмеченный котёл."""
    splits = load_splits(data_dir)
    full = load_full(data_dir)
    model_name_holder: dict[str, str] = {}

    with np.load(cache_path, allow_pickle=False) as archive:
        model_name_holder["model"] = json.loads(str(archive["meta"]))["model"]
    cache = EmbeddingCache.load(cache_path, model_name_holder["model"])

    pool = full.subset(lambda document: document.split == "pool")
    data = {name: l2_normalize(cache.matrix(corpus.ids())) for name, corpus in splits.items()}
    data["pool"] = l2_normalize(cache.matrix(pool.ids())) if len(pool) else np.zeros((0, 0))
    corpora = dict(splits)
    corpora["pool"] = pool
    return corpora, data, model_name_holder["model"]


def axes_material(data: dict[str, np.ndarray]) -> np.ndarray:
    """То, на чём считаются главные оси: train плюс котёл, без отложенных выборок."""
    parts = [data["train"]]
    if data["pool"].size:
        parts.append(data["pool"])
    return np.vstack(parts)


# --- оценка ------------------------------------------------------------------


def evaluate(labels, corpus, X) -> dict:
    """Все числа по одному разбиению разом, чтобы не считать их вразнобой."""
    rubrics = corpus.labels("topic_id")
    return {
        "n": len(rubrics),
        "ari_rubric": adjusted_rand_index(rubrics, labels),
        "purity_rubric": purity(rubrics, labels),
        "ari_language": adjusted_rand_index(corpus.labels("language"), labels),
        "ari_source": adjusted_rand_index(corpus.labels("source"), labels),
        "ari_section": adjusted_rand_index(corpus.labels("subtopic_id"), labels),
        "silhouette": silhouette_score(X, labels),
        "n_clusters_used": int(len(set(np.asarray(labels).tolist()))),
    }


def stability(X: np.ndarray, k: int, *, rounds: int = STABILITY_ROUNDS) -> float:
    """Насколько разбиение воспроизводится на других подвыборках тех же данных.

    Две подвыборки по STABILITY_FRACTION документов, на каждой своя модель,
    обе размечают ВЕСЬ набор — и разметки сравниваются между собой через ARI.
    Сравнение на всём наборе, а не на пересечении подвыборок: нас интересует
    устойчивость разбиения как правила, а не совпадение на общих точках.

    Значение около 1 означает «структура есть, и она одна и та же»; около нуля —
    «каждый запуск режет облако по-своему», то есть k выбран мимо.
    """
    rng = np.random.default_rng(RANDOM_STATE + k)
    n = X.shape[0]
    size = max(k + 1, int(n * STABILITY_FRACTION))
    scores = []
    for _ in range(rounds):
        first = rng.choice(n, size=size, replace=False)
        second = rng.choice(n, size=size, replace=False)
        left = KMeans(n_clusters=k, n_init=3, random_state=RANDOM_STATE).fit(X[first])
        right = KMeans(n_clusters=k, n_init=3, random_state=RANDOM_STATE).fit(X[second])
        scores.append(adjusted_rand_index(left.predict(X), right.predict(X)))
    return float(np.mean(scores))


# --- слияние близнецов -------------------------------------------------------


def merge_close_centroids(centroids: np.ndarray, sizes: np.ndarray, threshold: float):
    """Агломеративное слияние центроидов, стоящих ближе порога.

    Своя реализация, как и всё остальное в этой работе. Шаг простой: находим
    самую близкую пару, если косинус между ними не ниже порога — сливаем в один
    центроид, взвешенный по числу документов, и нормируем заново. Повторяем,
    пока такие пары есть.

    Взвешивание по размеру, а не среднее двух: кластер из ста документов и
    кластер из десяти — не равноправные половины, и простое среднее сдвинуло бы
    объединённый центр к меньшему.

    Возвращает новые центроиды и отображение «старый номер -> новый».
    """
    current = l2_normalize(np.asarray(centroids, dtype=np.float64).copy())
    weights = np.asarray(sizes, dtype=np.float64).copy()
    mapping = {index: index for index in range(current.shape[0])}
    alive = list(range(current.shape[0]))

    while len(alive) > 1:
        similarity = current[alive] @ current[alive].T
        np.fill_diagonal(similarity, -np.inf)
        flat = int(np.argmax(similarity))
        row, column = divmod(flat, len(alive))
        if similarity[row, column] < threshold:
            break
        keep, drop = alive[min(row, column)], alive[max(row, column)]
        total = weights[keep] + weights[drop]
        if total <= 0:
            merged = current[keep] + current[drop]
        else:
            merged = (current[keep] * weights[keep] + current[drop] * weights[drop]) / total
        current[keep] = merged / max(np.linalg.norm(merged), 1e-12)
        weights[keep] = total
        alive.remove(drop)
        for old, new in list(mapping.items()):
            if new == drop:
                mapping[old] = keep

    # Номера должны идти подряд: приложение обходит темы от нуля до k-1, и
    # дырка в нумерации превратилась бы в пустую строку списка тем.
    renumber = {old: position for position, old in enumerate(alive)}
    return current[alive], {index: renumber[target] for index, target in mapping.items()}


# --- уверенность -------------------------------------------------------------


def margins(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Запас уверенности каждого документа: 1 - d1/d2.

    d1 — расстояние до ближайшего центроида, d2 — до второго. Ноль означает
    «ровно посередине между двумя темами», единица — «стоит точно в центре
    своей». Отношение, а не разность: абсолютные расстояния зависят от
    размерности пространства, и порог, подобранный на одной модели, не значил бы
    ничего для другой.
    """
    distances = np.sqrt(np.maximum(squared_distances(X, centroids), 0.0))
    if distances.shape[1] < 2:
        return np.ones(distances.shape[0])
    nearest = np.partition(distances, 1, axis=1)[:, :2]
    first, second = nearest[:, 0], nearest[:, 1]
    safe = np.where(second <= 0, 1.0, second)
    return 1.0 - first / safe


def choose_margin_threshold(values: np.ndarray, share: float = TARGET_UNCLEAR_SHARE) -> float:
    """Порог, ниже которого модель отказывается называть тему."""
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, share))


# --- подбор ------------------------------------------------------------------


def fit_and_label(train_X, k):
    return KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit(train_X)


def search_projection(data, corpora, *, drops, keeps, ks, verbose=True):
    """Перебор drop × keep × k по ARI на validation.

    Оси считаются заново для каждой пары (drop, keep), но материал один и тот
    же: train плюс котёл. Один раз построенный базис максимальной длины можно
    было бы резать, и это дало бы те же столбцы, — но тогда keep=128 и keep=16
    делили бы одно разложение, и ошибка в нём осталась бы незамеченной в обоих.
    """
    material = axes_material(data)
    best = None
    rows = []
    max_components = max(keeps) + max(drops)
    axes = PrincipalAxes.fit(material, n_components=max_components)
    if verbose:
        total = float(np.sum(axes.explained_variance))
        head = axes.explained_variance[:5] / total if total else axes.explained_variance[:5]
        say(f"главные оси: {max_components} штук на {material.shape[0]} документах")
        say("  доля разброса у первых пяти: " + ", ".join(f"{value:.3f}" for value in head))

    for drop in drops:
        for keep in keeps:
            if drop + keep > axes.n_components:
                continue
            projection = axes.to_projection(drop=drop, keep=keep)
            train_X = projection.apply(data["train"])
            validation_X = projection.apply(data["validation"])
            for k in ks:
                model = fit_and_label(train_X, k)
                labels = model.predict(validation_X)
                score = adjusted_rand_index(corpora["validation"].labels("topic_id"), labels)
                language = adjusted_rand_index(corpora["validation"].labels("language"), labels)
                rows.append(
                    {
                        "drop": drop,
                        "keep": keep,
                        "k": k,
                        "ari_rubric": score,
                        "ari_language": language,
                        "silhouette": silhouette_score(validation_X, labels),
                    }
                )
                if best is None or score > best["ari_rubric"]:
                    best = rows[-1]
            if verbose:
                top = max(
                    (row for row in rows if row["drop"] == drop and row["keep"] == keep),
                    key=lambda row: row["ari_rubric"],
                )
                say(
                    f"  drop={drop} keep={keep:3d}: лучший k={top['k']:3d} "
                    f"ARI рубрика {top['ari_rubric']:+.3f}  язык {top['ari_language']:+.3f}"
                )
    return best, rows, axes


def choose_k(train_X, validation_X, corpora, ks):
    """Таблица по k: силуэт, устойчивость, ARI. Решение принимается по ней.

    Три числа, а не одно: силуэт меряет плотность вообще, устойчивость —
    воспроизводимость разбиения, ARI на validation — попадание в рубрики.
    Совпадение всех трёх и есть основание; расхождение — повод объяснить, а не
    спрятать.
    """
    rows = []
    for k in ks:
        model = fit_and_label(train_X, k)
        labels = model.predict(validation_X)
        rows.append(
            {
                "k": k,
                "silhouette_train": silhouette_score(train_X, model.labels_),
                "silhouette_validation": silhouette_score(validation_X, labels),
                "stability": stability(train_X, k),
                "ari_rubric": adjusted_rand_index(
                    corpora["validation"].labels("topic_id"), labels
                ),
                "inertia": float(model.inertia_ or 0.0),
            }
        )
        say(
            f"  k={k:3d}  силуэт {rows[-1]['silhouette_validation']:.3f}  "
            f"устойчивость {rows[-1]['stability']:.3f}  "
            f"ARI рубрика {rows[-1]['ari_rubric']:+.3f}"
        )
    return rows


def search_merge(train_X, validation_X, corpora, model, grid=MERGE_GRID):
    """Порог слияния близнецов — по ARI на validation, включая «не сливать»."""
    sizes = np.bincount(model.labels_, minlength=model.n_clusters).astype(float)
    rows = []
    for threshold in grid:
        if threshold is None:
            centroids = model.centroids_
        else:
            centroids, _ = merge_close_centroids(model.centroids_, sizes, threshold)
        labels = _assign(validation_X, centroids)
        rows.append(
            {
                "threshold": threshold,
                "n_clusters": int(centroids.shape[0]),
                "ari_rubric": adjusted_rand_index(
                    corpora["validation"].labels("topic_id"), labels
                ),
                "purity_rubric": purity(corpora["validation"].labels("topic_id"), labels),
            }
        )
        say(
            f"  порог {str(threshold):>5}: кластеров {rows[-1]['n_clusters']:3d}  "
            f"ARI {rows[-1]['ari_rubric']:+.3f}  чистота {rows[-1]['purity_rubric']:.3f}"
        )
    return max(rows, key=lambda row: row["ari_rubric"]), rows


def _assign(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return np.argmin(squared_distances(X, centroids), axis=1)


# --- подписи -----------------------------------------------------------------


def build_labels(corpus, labels, centroid_count):
    """Подписи кластеров: рубрика плюс уточнение характерными словами.

    Рубрика берётся как преобладающая среди документов кластера — тем же
    dominant_topics, что и раньше. Уточнение считается c-TF-IDF внутри семьи
    кластеров одной рубрики: только так два кластера «Иқтисод» получают слова,
    которыми они отличаются ДРУГ ОТ ДРУГА, а не общее для обоих «иқтисод».
    """
    rubric_codes = corpus.labels("topic_id")
    topics = dominant_topics(labels, rubric_codes, rubric_codes, centroid_count)

    tokens = tokenize_all(corpus.texts())
    tokens_of_cluster: dict[int, list[list[str]]] = {index: [] for index in range(centroid_count)}
    for cluster, token_list in zip(np.asarray(labels).tolist(), tokens):
        tokens_of_cluster[int(cluster)].append(token_list)

    rubric_of_cluster = {item.cluster: (item.topic_id or UNLABELLED) for item in topics}
    names_tg = {code: rubric.tg for code, rubric in RUBRIC_BY_CODE.items()}
    names_ru = {code: rubric.ru for code, rubric in RUBRIC_BY_CODE.items()}
    # Пустому кластеру рубрику дать неоткуда — в нём нет ни одного документа.
    # Без этой пары имён он получил бы подпись «UNK», то есть строку из
    # внутреннего кода в списке тем на экране.
    names_tg[UNLABELLED], names_ru[UNLABELLED] = rubric_names(UNLABELLED)
    labels_tg = compose_labels(rubric_of_cluster, tokens_of_cluster, names_tg)
    labels_ru = compose_labels(rubric_of_cluster, tokens_of_cluster, names_ru)
    return topics, labels_tg, labels_ru


# --- прогон ------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=str(DATA_DIR))
    parser.add_argument("--cache", default=str(CACHE_PATH))
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--report", default=str(REPORT_PATH))
    parser.add_argument("--quick", action="store_true", help="узкая сетка вариантов")
    parser.add_argument("--no-save", action="store_true", help="ничего не записывать")
    args = parser.parse_args()

    started = time.monotonic()
    corpora, data, embedding_model = load_matrices(Path(args.data), Path(args.cache))
    say(
        f"корпус: train {len(corpora['train'])}, validation {len(corpora['validation'])}, "
        f"test {len(corpora['test'])}, котёл {len(corpora['pool'])}"
    )
    say(f"эмбеддинги: {embedding_model}, {data['train'].shape[1]} измерений")
    say()

    drops = (0, 3) if args.quick else DROP_GRID
    keeps = (32,) if args.quick else KEEP_GRID
    ks = (12, 16, 20) if args.quick else K_GRID

    say("=== выбор преобразования (решение по validation) ===")
    best_variant, variant_rows, axes = search_projection(
        data, corpora, drops=drops, keeps=keeps, ks=ks
    )
    say(
        f"выбрано: drop={best_variant['drop']} keep={best_variant['keep']} "
        f"(ARI рубрика {best_variant['ari_rubric']:+.3f})"
    )
    say()

    projection = axes.to_projection(drop=best_variant["drop"], keep=best_variant["keep"])
    space = {name: projection.apply(matrix) for name, matrix in data.items() if matrix.size}

    say("=== выбор k: силуэт, устойчивость, ARI ===")
    k_rows = choose_k(space["train"], space["validation"], corpora, ks)
    best_k_row = max(k_rows, key=lambda row: row["ari_rubric"])
    most_stable = max(k_rows, key=lambda row: row["stability"])
    best_silhouette = max(k_rows, key=lambda row: row["silhouette_validation"])
    say(
        f"по ARI k={best_k_row['k']}, по устойчивости k={most_stable['k']}, "
        f"по силуэту k={best_silhouette['k']}"
    )
    chosen_k = best_k_row["k"]
    say(f"взято k={chosen_k}: решает попадание в рубрики, остальные два — как объяснение")
    say()

    model = fit_and_label(space["train"], chosen_k)

    say("=== слияние близнецов (решение по validation) ===")
    best_merge, merge_rows = search_merge(space["train"], space["validation"], corpora, model)
    sizes = np.bincount(model.labels_, minlength=chosen_k).astype(float)
    if best_merge["threshold"] is None:
        centroids = model.centroids_
        say("слияние не улучшает — оставлено как есть")
    else:
        centroids, _ = merge_close_centroids(model.centroids_, sizes, best_merge["threshold"])
        say(
            f"слито при пороге {best_merge['threshold']}: "
            f"{chosen_k} -> {centroids.shape[0]} кластеров"
        )
    say()

    train_labels = _assign(space["train"], centroids)
    validation_margins = margins(space["validation"], centroids)
    margin_threshold = choose_margin_threshold(validation_margins)
    say("=== порог уверенности ===")
    kept = validation_margins >= margin_threshold
    validation_labels = _assign(space["validation"], centroids)
    say(
        f"порог {margin_threshold:.4f}: отказов {int((~kept).sum())} из {kept.size} "
        f"({100 * (~kept).mean():.1f}%)"
    )
    if kept.any():
        rubrics = np.asarray(corpora["validation"].labels("topic_id"))
        say(
            f"  ARI на всех {adjusted_rand_index(rubrics, validation_labels):+.3f}, "
            f"на уверенных {adjusted_rand_index(rubrics[kept], validation_labels[kept]):+.3f}"
        )
    say()

    topics, labels_tg, labels_ru = build_labels(
        corpora["train"], train_labels, centroids.shape[0]
    )
    say("=== темы ===")
    for item in topics:
        say(
            f"  #{item.cluster:2d}  {labels_tg.get(item.cluster, ''):44s} "
            f"{labels_ru.get(item.cluster, ''):44s} документов {item.size:4d}  "
            f"доля рубрики {item.share:.2f}"
        )
    distinct = len({labels_tg[item.cluster] for item in topics})
    say(f"различных названий: {distinct} из {centroids.shape[0]}")
    say()

    say("=== итог на test (читается один раз) ===")
    test_labels = _assign(space["test"], centroids)
    test_scores = evaluate(test_labels, corpora["test"], space["test"])
    test_margins = margins(space["test"], centroids)
    test_kept = test_margins >= margin_threshold
    rubrics_test = np.asarray(corpora["test"].labels("topic_id"))
    for name, value in test_scores.items():
        say(f"  {name}: {value:.3f}" if isinstance(value, float) else f"  {name}: {value}")
    say(f"  отказов: {100 * (~test_kept).mean():.1f}%")
    if test_kept.any():
        say(
            "  ARI на уверенных: "
            f"{adjusted_rand_index(rubrics_test[test_kept], test_labels[test_kept]):+.3f}"
        )
    say()

    if args.no_save:
        say("(--no-save: модель и отчёт не записаны)")
        return 0

    enriched = tuple(
        type(item)(
            cluster=item.cluster,
            topic_id=item.topic_id,
            topic=labels_tg.get(item.cluster, item.topic),
            topic_ru=labels_ru.get(item.cluster, ""),
            topic_tg=labels_tg.get(item.cluster, ""),
            share=item.share,
            size=item.size,
        )
        for item in topics
    )
    saved = TopicModel(
        centroids=centroids,
        embedding_model=embedding_model,
        normalize=True,
        cluster_topics=enriched,
        projection=projection,
        params={
            "corpus": "topics_tj",
            "variant": "pca_projection",
            "dropped_components": best_variant["drop"],
            "kept_components": best_variant["keep"],
            "k_requested": chosen_k,
            "merge_threshold": best_merge["threshold"],
            "margin_threshold": margin_threshold,
            "random_state": RANDOM_STATE,
            "n_init": N_INIT,
            "trained_on": "train",
            "n_train": len(corpora["train"]),
            "axes_material": "train+pool",
            "n_axes_material": int(axes_material(data).shape[0]),
            "ari_rubric_test": test_scores["ari_rubric"],
            "purity_rubric_test": test_scores["purity_rubric"],
            "ari_language_test": test_scores["ari_language"],
            "ari_source_test": test_scores["ari_source"],
            "silhouette_test": test_scores["silhouette"],
            "unclear_share_test": float((~test_kept).mean()),
            "requires_fields_at_apply_time": [],
        },
    )
    saved.save(args.model)
    say(f"модель записана: {args.model}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": embedding_model,
        "corpus": {
            name: {
                "n": len(corpus),
                "by_rubric": dict(Counter(corpus.labels("topic_id"))),
            }
            for name, corpus in corpora.items()
        },
        "variants": variant_rows,
        "chosen_variant": best_variant,
        "k_table": k_rows,
        "chosen_k": chosen_k,
        "merge_table": merge_rows,
        "chosen_merge": best_merge,
        "margin_threshold": margin_threshold,
        "test": test_scores,
        "clusters": [
            {
                "cluster": item.cluster,
                "rubric": item.topic_id,
                "label_tg": labels_tg.get(item.cluster, ""),
                "label_ru": labels_ru.get(item.cluster, ""),
                "size": item.size,
                "share": item.share,
            }
            for item in topics
        ],
        "seconds": round(time.monotonic() - started, 1),
    }
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    say(f"отчёт записан: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
