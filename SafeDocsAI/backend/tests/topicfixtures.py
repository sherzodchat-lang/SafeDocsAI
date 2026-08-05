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

# Подписи тех же кластеров на языках интерфейса. Намеренно ОТЛИЧАЮТСЯ и от
# LABELS, и друг от друга: подписи приходят из артефакта разными полями, и
# совпадающие значения не дали бы отличить «клиент показал перевод» от «клиент
# показал ключ и угадал».
LABELS_RU = ("Налоги (ру)", "Право (ру)", "Финансы (ру)")
LABELS_TG = ("Андозҳо (тҷ)", "Ҳуқуқ (тҷ)", "Молия (тҷ)")

# Среднее по языку. Русское намеренно огромно и направлено вдоль первого
# центроида: именно так и выглядит беда, ради которой преобразование заведено —
# язык перевешивает тему, и все русские документы уезжают в один кластер.
GROUP_MEANS = {
    "en": [0.0, 0.0, 0.0, 0.0],
    "ru": [9.0, 0.0, 0.0, 0.0],
    "tg": [0.0, 0.0, 0.0, 9.0],
}

METRICS = {"ari_topic": 0.42, "purity": 0.61, "silhouette": 0.13}


def cluster_topics(*, localized: bool = True) -> list[dict]:
    """Разметка «кластер -> тема» так, как её пишет обучающая сторона.

    localized=False собирает артефакт БЕЗ переводов — ровно такой, каким его
    писал код до их появления. Такие файлы лежат на стендах, и раздел обязан их
    читать: отсутствие перевода не отказ, а откат к устойчивой подписи.
    """
    return [
        {
            "cluster": index,
            "topic_id": f"T{index:02d}",
            "topic": name,
            **(
                {"topic_ru": LABELS_RU[index], "topic_tg": LABELS_TG[index]}
                if localized
                else {}
            ),
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


def write_language_artifact(path, *, localized: bool = True) -> Path:
    """Артефакт победившего вида: вычитание среднего по языку.

    localized=False собирает тот же артефакт без переводов названий тем — так
    выглядит файл, обученный до их появления.
    """
    return write_artifact(
        path,
        transform={
            "kind": "group_mean_shift",
            "group_field": "language",
            "groups": list(GROUP_MEANS),
        },
        arrays={"transform_group_means": [GROUP_MEANS[key] for key in GROUP_MEANS]},
        meta_overrides=(
            None if localized else {"cluster_topics": cluster_topics(localized=False)}
        ),
    )


# --- Артефакт победившего вида: центрирование по ячейке «язык × жанр» --------
#
# Так его пишет обучающая сторона (pipeline/transforms.py, GroupCentering):
# группа задаётся СПИСКОМ полей, ключи составные, средние лежат матрицей.
CELL_FIELDS = ("language", "dataset_origin")

# Средние по ячейкам. Реальные — маленькие и правдоподобные (языковой сдвиг
# соизмерим с тематическим), синтетические — намеренно уводят в сторону
# второго центроида. Так видно, ЧТО именно выбрала проекция: перепутанные
# ячейки дают другой кластер, а не чуть-чуть другой вектор.
CELL_MEANS = {
    "en|real": [0.0, 0.0, 0.0, 0.0],
    "en|synthetic": [0.0, 3.0, 0.0, 0.0],
    "ru|real": [0.9, 0.0, 0.0, 0.0],
    "ru|synthetic": [0.0, 3.0, 0.0, 0.0],
    "tg|real": [0.0, 0.0, 0.0, 0.9],
    "tg|synthetic": [0.0, 3.0, 0.0, 0.0],
}

# Среднее всей обучающей выборки: применяется к документу на языке, которого
# при обучении не было.
CELL_FALLBACK = [0.3, 0.0, 0.0, 0.3]

# Во сколько раз вектор документа короче единичного. Вектор документа в бою —
# СРЕДНЕЕ векторов его фрагментов, и единичным он не бывает; средние же групп
# посчитаны на нормированных эмбеддингах. Множитель здесь для того, чтобы это
# расхождение в тестах присутствовало, а не считалось несущественным.
CELL_DOCUMENT_SCALE = 0.8


def write_cell_artifact(
    path,
    *,
    fields=CELL_FIELDS,
    means=None,
    renormalize=None,
    fallback=CELL_FALLBACK,
    keys=None,
    matrix=None,
) -> Path:
    """Артефакт с преобразованием group_centering по двум полям."""
    cells = CELL_MEANS if means is None else means
    key_list = list(cells) if keys is None else list(keys)
    transform = {
        "kind": "group_centering",
        "fields": list(fields),
        "keys": key_list,
        "counts": {key: 100 for key in key_list},
        "dim": CENTROIDS.shape[1],
    }
    if renormalize is not None:
        transform["renormalize"] = renormalize
    rows = [cells[key] for key in key_list] if matrix is None else matrix
    return write_artifact(
        path,
        transform=transform,
        arrays={"transform_means": rows, "transform_fallback": fallback},
    )


def cell_document_vector_for(language: str, cluster: int) -> list[float]:
    """Сырой вектор документа: тема плюс сдвиг своей РЕАЛЬНОЙ ячейки.

    Короче единичного (CELL_DOCUMENT_SCALE) — ровно так и приходит вектор
    документа из ChromaDB: это среднее векторов его фрагментов.
    """
    direction = np.zeros(CENTROIDS.shape[1])
    direction[cluster] = 1.0
    mean = np.asarray(CELL_MEANS[f"{language}|real"])
    return list(CELL_DOCUMENT_SCALE * (mean + direction))


def document_vector_for(language: str, cluster: int) -> list[float]:
    """Сырой вектор документа: тема плюс сдвиг своего языка.

    Ровно то, что приходит из ChromaDB: преобразование к нему ещё не
    применялось.
    """
    direction = np.zeros(CENTROIDS.shape[1])
    direction[cluster] = 1.0
    return list(np.asarray(GROUP_MEANS[language]) + direction)
