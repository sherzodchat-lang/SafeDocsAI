"""Кластеризация документов по темам: собственная реализация K-means.

Раздел намеренно ничего не знает про базу, эмбеддинги и HTTP: на вход —
матрица чисел, на выход — метки кластеров. Поэтому импорт этого пакета ничего
тяжёлого за собой не тянет, а алгоритм проверяется на синтетике, где верный
ответ известен заранее.

Библиотечной кластеризации здесь нет: sklearn запрещён и запрет проверяется
тестом tests/test_topics_no_sklearn_guard.py.
"""

from app.modules.topics.kmeans import (
    DEFAULT_MAX_ITER,
    DEFAULT_N_INIT,
    DEFAULT_TOL,
    KMeans,
    KScore,
    KSearchResult,
    choose_k,
    kmeans_plus_plus_init,
    l2_normalize,
    squared_distances,
)
from app.modules.topics.metrics import (
    adjusted_rand_index,
    contingency_matrix,
    inertia,
    purity,
    silhouette_per_cluster,
    silhouette_samples,
    silhouette_score,
)

__all__ = [
    "DEFAULT_MAX_ITER",
    "DEFAULT_N_INIT",
    "DEFAULT_TOL",
    "KMeans",
    "KScore",
    "KSearchResult",
    "adjusted_rand_index",
    "choose_k",
    "contingency_matrix",
    "inertia",
    "kmeans_plus_plus_init",
    "l2_normalize",
    "purity",
    "silhouette_per_cluster",
    "silhouette_samples",
    "silhouette_score",
    "squared_distances",
]
