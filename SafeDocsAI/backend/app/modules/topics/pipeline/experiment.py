"""Прогоны кластеризации и сбор чисел для защиты.

Здесь нет ни одной новой формулы: и K-means, и метрики берутся из
app.modules.topics как есть. Этот модуль отвечает на другой вопрос — ПРОТИВ
ЧЕГО считать метрики.

У корпуса четыре разметки одновременно, и кластеры совпадают с ними
по-разному:

  * topic_id (20 тем) — то, ради чего работа делается;
  * language (3 языка) — то, с чем кластеры могут совпасть вместо темы, и это
    самый вероятный способ получить красивые числа, не решив задачу;
  * dataset_origin (2 класса) — синтетические корпоративные жанры против
    выдержек Википедии; наборы тем у них не пересекаются, а стилистика
    различается резче, чем темы внутри каждого слоя;
  * subtopic_id (104 подтемы) — второй уровень той же тематической разметки.

Поэтому каждый прогон считается против всех четырёх сразу. Отдельно
синтетика, отдельно реальные тексты и смесь — потому что синтетика
сгенерирована ПО ТЕМАМ и разделима по построению, и метрика на ней измеряет
качество генератора, а не качество кластеризации. Разрыв между слоями — это и
есть честный ответ на вопрос «а не потому ли разделилось, что вы сами так и
написали».
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from app.modules.topics.kmeans import KMeans, choose_k, l2_normalize
from app.modules.topics.metrics import (
    adjusted_rand_index,
    inertia,
    purity,
    silhouette_per_cluster,
    silhouette_score,
)
from app.modules.topics.pipeline.dataset import Corpus
from app.modules.topics.pipeline.embeddings import EmbeddingCache
from app.modules.topics.pipeline.model_io import ClusterTopic, TopicModel, dominant_topics

# Колонки разметки, против которых считается каждый прогон. Порядок и состав
# зафиксированы здесь, а не собираются по месту: если какой-то прогон посчитает
# на одну колонку меньше, в отчёте появится дырка, а сравнивать прогоны между
# собой станет нельзя.
LABEL_FIELDS = ("topic_id", "subtopic_id", "language", "dataset_origin")

RANDOM_STATE = 42


def _combined(first: Sequence[str], second: Sequence[str]) -> list[str]:
    """Разметка «тема И язык» одной колонкой — 60 ячеек вместо 20 тем.

    Нужна ровно для одного вопроса: не режет ли алгоритм каждую тему на
    языковые куски. Против такой разметки высокий ARI получит разбиение, где
    кластер — это пара (тема, язык), и по одному только ARI против темы это
    было бы не отличить от «тема не разделилась вовсе».
    """
    return [f"{a}|{b}" for a, b in zip(first, second)]


def external_scores(labels_pred: Sequence[int], corpus: Corpus) -> dict[str, Any]:
    """Чистота и ARI против каждой разметки корпуса.

    Обе метрики нужны вместе. Чистота отвечает на вопрос «однородны ли
    кластеры», но растёт с ростом k и при k = n равна единице. ARI за дробление
    наказывает, но его величину труднее объяснить словами. Одну без другой
    показывать нельзя: чистота 0.9 при ARI 0.2 означает, что тема разрезана на
    много чистых кусков, и это совсем не успех.
    """
    result: dict[str, Any] = {}
    for field in LABEL_FIELDS:
        truth = corpus.labels(field)
        result[field] = {
            "purity": float(purity(truth, labels_pred)),
            "ari": float(adjusted_rand_index(truth, labels_pred)),
            "n_classes": len(set(truth)),
        }
    combined = _combined(corpus.labels("topic_id"), corpus.labels("language"))
    result["topic_and_language"] = {
        "purity": float(purity(combined, labels_pred)),
        "ari": float(adjusted_rand_index(combined, labels_pred)),
        "n_classes": len(set(combined)),
    }
    return result


def internal_scores(
    X: np.ndarray,
    labels: Sequence[int],
    centroids: np.ndarray,
    *,
    normalize: bool,
) -> dict[str, Any]:
    """Инерция и силуэт в том же пространстве, где работал алгоритм.

    Нормировка повторяется здесь потому, что X приходит сырым: центроиды лежат
    в нормированном пространстве, и считать до них расстояние от ненормированной
    точки — значит получить число, не означающее ничего.

    Силуэт считается точно (sample_size=None): корпус — 2278 документов,
    матрица расстояний влезает в память, а оценка по подвыборке добавила бы к
    сравнению слоёв разброс, которого в них нет.
    """
    data = l2_normalize(np.asarray(X, dtype=np.float64)) if normalize else np.asarray(X, dtype=np.float64)
    labels_array = np.asarray(labels)
    distinct = len(np.unique(labels_array))
    if 2 <= distinct < data.shape[0]:
        silhouette = float(silhouette_score(data, labels_array, sample_size=None))
    else:
        silhouette = float("nan")
    return {
        "inertia": float(inertia(data, labels_array, centroids)),
        "silhouette": silhouette,
        "n_clusters_used": int(distinct),
        "n_samples": int(data.shape[0]),
    }


@dataclass
class Fitted:
    """Обученная модель вместе с данными, на которых её обучали."""

    kmeans: KMeans
    corpus: Corpus
    X: np.ndarray
    cluster_topics: tuple[ClusterTopic, ...]


def fit_on(corpus: Corpus, X: np.ndarray, k: int, *, random_state: int = RANDOM_STATE) -> Fitted:
    model = KMeans(n_clusters=k, random_state=random_state, normalize=True).fit(X)
    assert model.labels_ is not None
    topics = dominant_topics(
        model.labels_, corpus.labels("topic_id"), corpus.labels("topic"), k
    )
    return Fitted(kmeans=model, corpus=corpus, X=X, cluster_topics=topics)


def evaluate_split(fitted: Fitted, corpus: Corpus, X: np.ndarray) -> dict[str, Any]:
    """Все метрики одной выборки под одной обученной моделью."""
    labels = fitted.kmeans.predict(X)
    assert fitted.kmeans.centroids_ is not None
    report = internal_scores(
        X, labels, fitted.kmeans.centroids_, normalize=fitted.kmeans.normalize
    )
    report["external"] = external_scores(labels, corpus)
    return report


def cluster_breakdown(fitted: Fitted) -> list[dict[str, Any]]:
    """Что лежит в каждом кластере: тема, языки, происхождение, силуэт.

    Среднее по всем кластерам прячет главное: обычно один-два кластера собраны
    хорошо, а остальное — свалка, и разбивка показывает, какую именно тему
    алгоритм не разделил.
    """
    labels = np.asarray(fitted.kmeans.labels_)
    data = l2_normalize(fitted.X) if fitted.kmeans.normalize else fitted.X
    distinct = len(np.unique(labels))
    if 2 <= distinct < data.shape[0]:
        per_cluster = silhouette_per_cluster(data, labels, sample_size=None)
    else:
        per_cluster = {}

    languages = fitted.corpus.labels("language")
    origins = fitted.corpus.labels("dataset_origin")
    topic_ids = fitted.corpus.labels("topic_id")

    rows: list[dict[str, Any]] = []
    for item in fitted.cluster_topics:
        members = np.flatnonzero(labels == item.cluster)
        language_counts: dict[str, int] = {}
        origin_counts: dict[str, int] = {}
        topic_counts: dict[str, int] = {}
        for index in members:
            index = int(index)
            language_counts[languages[index]] = language_counts.get(languages[index], 0) + 1
            origin_counts[origins[index]] = origin_counts.get(origins[index], 0) + 1
            topic_counts[topic_ids[index]] = topic_counts.get(topic_ids[index], 0) + 1
        size = int(members.size)
        rows.append(
            {
                "cluster": item.cluster,
                "size": size,
                "dominant_topic_id": item.topic_id,
                "dominant_topic": item.topic,
                "topic_share": item.share,
                "dominant_language": max(language_counts, key=lambda key: (language_counts[key], key))
                if language_counts
                else "",
                "language_share": (max(language_counts.values()) / size) if size else 0.0,
                "languages": dict(sorted(language_counts.items())),
                "origins": dict(sorted(origin_counts.items())),
                "n_topics_present": len(topic_counts),
                "silhouette": float(per_cluster.get(item.cluster, float("nan"))),
            }
        )
    return rows


def k_search_table(
    X: np.ndarray, k_values: Sequence[int], *, random_state: int = RANDOM_STATE
) -> dict[str, Any]:
    """Таблица перебора k — то, по чему рисуют локоть и график силуэта.

    silhouette_sample_size=None: обучающая выборка меньше трёх тысяч, точный
    силуэт по ней считается за секунды, а оценка по подвыборке добавила бы к
    сравнению строк таблицы разброс, из-за которого «локоть» можно было бы
    подвинуть на соседнее k случайностью выборки.
    """
    started = time.monotonic()
    result = choose_k(
        X,
        k_values,
        random_state=random_state,
        normalize=True,
        silhouette_sample_size=None,
    )
    return {
        "k_values": [int(value) for value in sorted(set(k_values))],
        "best_k_by_silhouette": int(result.best_k),
        "elbow_k": int(result.elbow_k),
        "seconds": round(time.monotonic() - started, 1),
        "table": [
            {
                "k": score.k,
                "inertia": float(score.inertia),
                "silhouette": float(score.silhouette),
                "n_iter": int(score.n_iter),
            }
            for score in result.scores
        ],
    }


def matrices(
    splits: dict[str, Corpus], cache: EmbeddingCache
) -> dict[str, np.ndarray]:
    return {name: cache.matrix(corpus.ids()) for name, corpus in splits.items()}


def run_full_model(
    splits: dict[str, Corpus],
    data: dict[str, np.ndarray],
    k: int,
    *,
    random_state: int = RANDOM_STATE,
) -> tuple[Fitted, dict[str, Any]]:
    """Основной прогон: обучение на train, назначение на validation и test."""
    fitted = fit_on(splits["train"], data["train"], k, random_state=random_state)
    report = {
        "k": k,
        "n_iter": int(fitted.kmeans.n_iter_ or 0),
        "splits": {
            name: evaluate_split(fitted, splits[name], data[name])
            for name in ("train", "validation", "test")
        },
        "clusters": cluster_breakdown(fitted),
    }
    return fitted, report


# --- слои --------------------------------------------------------------------


def _layer(splits: dict[str, Corpus], data: dict[str, np.ndarray], predicate):
    """Подвыборка по предикату во всех разбиениях сразу, с сохранением порядка.

    Матрица режется теми же индексами, что и корпус, одним проходом: два
    независимых фильтра (по документам и по строкам) разъехались бы, и метрики
    посчитались бы по чужим векторам.
    """
    result_corpus: dict[str, Corpus] = {}
    result_data: dict[str, np.ndarray] = {}
    for name, corpus in splits.items():
        keep = [index for index, document in enumerate(corpus) if predicate(document)]
        result_corpus[name] = Corpus(tuple(corpus.documents[index] for index in keep))
        result_data[name] = data[name][keep]
    return result_corpus, result_data


LAYERS = {
    "synthetic": lambda document: document.dataset_origin == "synthetic",
    "real": lambda document: document.dataset_origin == "real",
    "mixed": lambda document: True,
}


def run_layers(
    splits: dict[str, Corpus],
    data: dict[str, np.ndarray],
    *,
    random_state: int = RANDOM_STATE,
    k_override: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Отдельная модель на каждый слой, k = число настоящих тем этого слоя.

    k берётся по разметке, а не подбирается: сравниваются НЕ модели, а слои, и
    разное k у слоёв добавило бы к разрыву между ними разницу в числе кластеров.
    При k, равном числу тем, ARI отвечает ровно на вопрос «нашлись ли эти
    темы», и слои сравнимы напрямую.
    """
    report: dict[str, Any] = {}
    for name, predicate in LAYERS.items():
        layer_splits, layer_data = _layer(splits, data, predicate)
        topics = sorted(set(layer_splits["train"].labels("topic_id")))
        k = (k_override or {}).get(name, len(topics))
        fitted = fit_on(layer_splits["train"], layer_data["train"], k, random_state=random_state)
        report[name] = {
            "k": k,
            "n_true_topics": len(topics),
            "topic_ids": topics,
            "n_documents": {
                split: len(layer_splits[split]) for split in ("train", "validation", "test")
            },
            "splits": {
                split: evaluate_split(fitted, layer_splits[split], layer_data[split])
                for split in ("train", "validation", "test")
            },
            "clusters": cluster_breakdown(fitted),
        }
    return report


LANGUAGES = ("en", "ru", "tg")


def run_cells(
    splits: dict[str, Corpus],
    data: dict[str, np.ndarray],
    *,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Тема при ЗАКРЕПЛЁННОМ языке — единственное честное сравнение слоёв.

    run_layers сравнивает синтетику с реальными текстами на трёх языках сразу, и
    это сравнение ничего не измеряет: кластеры внутри слоя уходят на язык
    раньше, чем на тему, и оба слоя получают одинаково низкий ARI по причине,
    к слоям отношения не имеющей. Разрыв в такой постановке выходит около нуля
    и читается как «слои неразличимы», хотя на деле он просто не измерен.

    Поэтому корпус режется на шесть ячеек (происхождение x язык), и в каждой
    k равно числу тем именно этой ячейки: 12 у A-тем, 8 у B-тем. Внутри ячейки
    язык постоянен и объяснить им разбиение нельзя — остаётся ровно тот
    вопрос, ради которого работа делается: находятся ли темы.
    """
    cells: dict[str, Any] = {}
    for origin in ("synthetic", "real"):
        for language in LANGUAGES:
            def predicate(document, o=origin, l=language):
                return document.dataset_origin == o and document.language == l

            cell_splits, cell_data = _layer(splits, data, predicate)
            topics = sorted(set(cell_splits["train"].labels("topic_id")))
            fitted = fit_on(
                cell_splits["train"], cell_data["train"], len(topics), random_state=random_state
            )
            cells[f"{origin}/{language}"] = {
                "origin": origin,
                "language": language,
                "k": len(topics),
                "n_true_topics": len(topics),
                "n_documents": {
                    split: len(cell_splits[split]) for split in ("train", "validation", "test")
                },
                "splits": {
                    split: evaluate_split(fitted, cell_splits[split], cell_data[split])
                    for split in ("train", "validation", "test")
                },
            }

    def mean_over(origin: str, split: str, metric: str) -> float:
        values = [
            cell["splits"][split]["external"]["topic_id"][metric]
            for cell in cells.values()
            if cell["origin"] == origin
        ]
        return float(sum(values) / len(values)) if values else float("nan")

    by_layer = {
        origin: {
            split: {
                "mean_ari_topic": mean_over(origin, split, "ari"),
                "mean_purity_topic": mean_over(origin, split, "purity"),
            }
            for split in ("train", "validation", "test")
        }
        for origin in ("synthetic", "real")
    }
    return {
        "cells": cells,
        "by_layer": by_layer,
        # Знак важнее величины. Ожидание перед работой было обратным —
        # «синтетика разделима по построению, её метрики завышены», — и
        # отрицательный разрыв означает, что упрёк «разделилось, потому что вы
        # сами это написали» данными не подтверждается.
        "gap_ari_topic": {
            split: by_layer["synthetic"][split]["mean_ari_topic"]
            - by_layer["real"][split]["mean_ari_topic"]
            for split in ("train", "validation", "test")
        },
    }


def layers_within_global(fitted: Fitted, corpus: Corpus, X: np.ndarray) -> dict[str, Any]:
    """Тот же общий прогон, но метрики посчитаны отдельно по слоям.

    Отличается от run_layers принципиально: модель здесь ОДНА, обученная на
    смеси, а слои — это лишь подмножества, на которых считаются числа. Так
    видно, насколько общая модель хороша на каждом слое, тогда как run_layers
    показывает, насколько слой разделим сам по себе.
    """
    labels = fitted.kmeans.predict(X)
    report: dict[str, Any] = {}
    for name, predicate in LAYERS.items():
        keep = [index for index, document in enumerate(corpus) if predicate(document)]
        if not keep:
            continue
        subset = Corpus(tuple(corpus.documents[index] for index in keep))
        subset_labels = labels[keep]
        report[name] = {
            "n_documents": len(subset),
            "n_clusters_used": int(len(np.unique(subset_labels))),
            "external": external_scores(subset_labels, subset),
        }
    return report


def neighbour_agreement(corpus: Corpus, X: np.ndarray) -> dict[str, Any]:
    """Доля документов, чей ближайший сосед несёт ту же метку.

    Единственное здесь число, не зависящее ни от k, ни от K-means вообще: оно
    описывает саму геометрию пространства эмбеддингов. Нужно как раз для
    защиты. Низкий ARI по теме допускает два совершенно разных объяснения —
    «модель эмбеддингов не различает темы» и «темы различимы, но k-means
    тратит кластеры на язык и жанр», — и по одному ARI они неотличимы. Согласие
    соседей разделяет эти случаи: если ближайший сосед документа обычно той же
    темы, тематический сигнал в векторах есть, и виновата процедура, а не
    признаки.

    Считается на всём корпусе сразу, потому что описывает признаки, а не
    обученную модель: делить на train и test тут нечего, никакого обучения не
    происходит.
    """
    data = l2_normalize(np.asarray(X, dtype=np.float64))
    similarity = data @ data.T
    np.fill_diagonal(similarity, -np.inf)
    nearest = np.argmax(similarity, axis=1)

    result: dict[str, Any] = {}
    for field in LABEL_FIELDS + ("language",):
        labels = np.asarray(corpus.labels(field))
        result[field] = float(np.mean(labels[nearest] == labels))
    # Случайный уровень для сравнения: без него 0.68 по теме нечем мерить.
    # Берётся сумма квадратов долей классов — вероятность совпадения меток у
    # двух независимо взятых документов.
    baseline: dict[str, Any] = {}
    for field in LABEL_FIELDS + ("language",):
        counts = np.array(list(corpus.counts(field).values()), dtype=np.float64)
        shares = counts / counts.sum()
        baseline[field] = float(np.sum(shares**2))
    return {"same_label_as_nearest_neighbour": result, "chance_level": baseline}


def build_topic_model(
    fitted: Fitted, embedding_model: str, params: dict[str, Any]
) -> TopicModel:
    assert fitted.kmeans.centroids_ is not None
    return TopicModel(
        centroids=fitted.kmeans.centroids_,
        embedding_model=embedding_model,
        normalize=fitted.kmeans.normalize,
        cluster_topics=fitted.cluster_topics,
        params=params,
    )
