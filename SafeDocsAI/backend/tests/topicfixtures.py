"""Артефакт обученной модели тем, собранный руками для тестов.

Настоящий файл на диске, а не мок загрузчика. Причина простая: половина
проверок этого раздела — про то, ЧТО именно лежит в артефакте и как оно
читается (преобразование вектора, подписи кластеров, метрики), и мок загрузчика
проверял бы мок.

Размерность здесь 4, а не 4096: геометрия от этого не меняется, а числа в тесте
становятся читаемыми глазами. Имя не начинается с test_, поэтому unittest
discover не ищет здесь тестов.
"""

import json
from pathlib import Path

import numpy as np

# Модель эмбеддингов артефакта. Тесты подменяют ею действующую настройку:
# слой назначения отказывается работать, когда система считает векторы другой
# моделью, и без подмены проверки зависели бы от runtime_settings.json стенда.
ARTIFACT_EMBEDDING_MODEL = "test-embed:1b"

# Центроиды в ПРЕОБРАЗОВАННОМ пространстве: три взаимно ортогональных
# направления. С такими центроидами «ближайший» читается глазами.
CENTROIDS = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
)

LABELS = ("Налоги", "Право", "Финансы")

# Среднее по языку. Русское намеренно огромно и направлено вдоль первого
# центроида: именно так и выглядит беда, ради которой преобразование заведено —
# язык перевешивает тему, и все русские документы уезжают в один кластер.
GROUP_MEANS = {
    "en": [0.0, 0.0, 0.0, 0.0],
    "ru": [9.0, 0.0, 0.0, 0.0],
    "tg": [0.0, 0.0, 0.0, 9.0],
}

METRICS = {"ari_topic": 0.42, "purity": 0.61, "silhouette": 0.13}


def cluster_topics() -> list[dict]:
    return [
        {
            "cluster": index,
            "topic_id": f"T{index:02d}",
            "topic": name,
            "share": 0.7,
            "size": 100,
        }
        for index, name in enumerate(LABELS)
    ]


def write_artifact(
    path,
    *,
    centroids=None,
    transform=None,
    arrays=None,
    meta_overrides=None,
) -> Path:
    """Собрать .npz такой же формы, какую пишет обучающая сторона."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    matrix = CENTROIDS if centroids is None else np.asarray(centroids, dtype=np.float64)
    meta = {
        "version": 1,
        "embedding_model": ARTIFACT_EMBEDDING_MODEL,
        "normalize": True,
        "n_clusters": int(matrix.shape[0]),
        "dim": int(matrix.shape[1]),
        "cluster_topics": cluster_topics()[: matrix.shape[0]],
        "metrics": dict(METRICS),
        "params": {"k": int(matrix.shape[0]), "random_state": 42},
    }
    if transform is not None:
        meta["transform"] = transform
    meta.update(meta_overrides or {})
    payload = {"centroids": matrix, "meta": np.array(json.dumps(meta, ensure_ascii=False))}
    for name, value in (arrays or {}).items():
        payload[name] = np.asarray(value, dtype=np.float64)
    np.savez_compressed(file, **payload)
    return file


def write_language_artifact(path) -> Path:
    """Артефакт победившего вида: вычитание среднего по языку."""
    return write_artifact(
        path,
        transform={
            "kind": "group_mean_shift",
            "group_field": "language",
            "groups": list(GROUP_MEANS),
        },
        arrays={"transform_group_means": [GROUP_MEANS[key] for key in GROUP_MEANS]},
    )


def document_vector_for(language: str, cluster: int) -> list[float]:
    """Сырой вектор документа: тема плюс сдвиг своего языка.

    Ровно то, что приходит из ChromaDB: преобразование к нему ещё не
    применялось.
    """
    direction = np.zeros(CENTROIDS.shape[1])
    direction[cluster] = 1.0
    return list(np.asarray(GROUP_MEANS[language]) + direction)
