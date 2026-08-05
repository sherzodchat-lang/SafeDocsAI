"""Подавление посторонних осей: центрирование по группе и маршрутизация.

Предыдущий прогон показал, что кластеры этого корпуса ложатся на язык и на
жанр, а не на тему: при k=20 ARI против языка +0.415, против темы +0.063.
Тематический сигнал в векторах при этом есть — у 68% документов ближайший
сосед той же темы при случайном уровне 6%. Значит, тема не отсутствует, а
перекрыта осями, которые сильнее.

Убрать такую ось можно двумя способами, и оба живут здесь:

  * GroupCentering — вычесть из вектора средний вектор его группы. Ось
    подавляется, но корпус остаётся единым, и кластеры по-прежнему могут
    объединять документы разных языков;
  * ClusterRouting — не смешивать группы вовсе: своя модель на каждую группу, а
    документ обслуживается только «своими» кластерами. Ось устранена по
    построению, ценой того, что одна тема на трёх языках даёт три разных
    кластера.

Оба преобразования требуют ЗНАТЬ ГРУППУ документа. Для эксперимента она лежит
в разметке, для приложения её придётся определять (язык — определителем языка,
жанр — по источнику документа). Это не деталь реализации, а требование к
встраиванию, и поэтому оно записано в сохранённую модель: см. required_fields.

Утечки. Средние считаются ТОЛЬКО по обучающей выборке и применяются к
validation и test как есть. Среднее, посчитанное по всему корпусу, — это
подглядывание в test: положение тестового документа относительно центра его
группы зависело бы от него самого.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from app.modules.topics.kmeans import l2_normalize, squared_distances

# Разделитель составного ключа группы. Символ выбран такой, которого нет ни в
# одном значении разметки (языки — двухбуквенные коды, происхождение —
# synthetic/real): иначе ключи двух разных пар групп могли бы совпасть.
KEY_SEPARATOR = "|"


def group_key(values: Iterable[str]) -> str:
    return KEY_SEPARATOR.join(str(value) for value in values)


def group_keys(documents: Iterable[Any], fields: Sequence[str]) -> list[str]:
    """Ключ группы для каждого документа: «ru», «ru|real» и так далее.

    Работает с любым объектом, у которого есть нужные атрибуты, а не только с
    Document: тем же кодом ключ считается и для документа приложения, у
    которого язык определён на лету, а разметки нет вовсе.
    """
    if not fields:
        raise ValueError("группа не может определяться пустым набором полей")
    return [group_key(str(getattr(item, field)) for field in fields) for item in documents]


@dataclass(frozen=True)
class GroupCentering:
    """Вычитание среднего вектора группы с последующей перенормировкой.

    Почему перенормировка обязательна. Кластеризация идёт по косинусу: KMeans
    получает normalize=True и приводит строки к единичной длине. После вычитания
    средних длины строк становятся разными — документ, лежащий близко к центру
    своей группы, получает короткий вектор, — и без перенормировки инерция и
    силуэт считались бы в одном пространстве, а метрики отчёта — в другом.

    fallback_mean — средний вектор всей обучающей выборки. Он применяется к
    документу, чья группа при обучении не встречалась (в приложении это
    документ на четвёртом языке). Отказ здесь был бы хуже: преобразование в
    таком случае вырождается в обычное глобальное центрирование, то есть в
    заведомо более слабое, но осмысленное действие, тогда как отказ означал бы,
    что документ вообще не классифицируется.
    """

    fields: tuple[str, ...]
    keys: tuple[str, ...]
    means: np.ndarray  # (len(keys), dim)
    fallback_mean: np.ndarray  # (dim,)
    counts: dict[str, int]
    renormalize: bool = True

    kind = "group_centering"

    @property
    def dim(self) -> int:
        return int(self.means.shape[1])

    @classmethod
    def fit(
        cls,
        documents: Sequence[Any],
        X: np.ndarray,
        fields: Sequence[str],
        *,
        renormalize: bool = True,
    ) -> "GroupCentering":
        """Средние по группам — считать ТОЛЬКО на train, см. заголовок модуля."""
        data = np.asarray(X, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("ожидалась матрица (n_samples, n_features)")
        if data.shape[0] != len(documents):
            raise ValueError(
                f"документов {len(documents)}, а строк матрицы {data.shape[0]}: "
                "разметка и векторы разъехались"
            )
        keys = group_keys(documents, fields)
        unique = sorted(set(keys))
        means = np.vstack(
            [data[[i for i, key in enumerate(keys) if key == value]].mean(axis=0) for value in unique]
        )
        counts = {value: sum(1 for key in keys if key == value) for value in unique}
        return cls(
            fields=tuple(fields),
            keys=tuple(unique),
            means=means,
            fallback_mean=data.mean(axis=0),
            counts=counts,
            renormalize=renormalize,
        )

    def apply_to_keys(self, keys: Sequence[str], X: np.ndarray) -> np.ndarray:
        data = np.asarray(X, dtype=np.float64)
        if data.shape[0] != len(keys):
            raise ValueError(
                f"ключей {len(keys)}, а строк матрицы {data.shape[0]}"
            )
        if data.shape[1] != self.dim:
            raise ValueError(f"ожидалось {self.dim} признаков, получено {data.shape[1]}")
        index = {key: position for position, key in enumerate(self.keys)}
        shift = np.vstack(
            [
                self.means[index[key]] if key in index else self.fallback_mean
                for key in keys
            ]
        )
        centered = data - shift
        return l2_normalize(centered) if self.renormalize else centered

    def apply(self, documents: Sequence[Any], X: np.ndarray) -> np.ndarray:
        return self.apply_to_keys(group_keys(documents, self.fields), X)

    def unknown_keys(self, keys: Sequence[str]) -> list[str]:
        known = set(self.keys)
        return sorted({key for key in keys if key not in known})

    def meta(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "fields": list(self.fields),
            "keys": list(self.keys),
            "counts": dict(self.counts),
            "renormalize": bool(self.renormalize),
            "dim": self.dim,
        }

    @classmethod
    def from_saved(cls, meta: dict[str, Any], means: np.ndarray, fallback: np.ndarray) -> "GroupCentering":
        keys = tuple(str(value) for value in meta["keys"])
        means = np.asarray(means, dtype=np.float64)
        if means.shape[0] != len(keys):
            raise ValueError(
                f"в модели {len(keys)} групп, а средних векторов {means.shape[0]}"
            )
        return cls(
            fields=tuple(str(value) for value in meta["fields"]),
            keys=keys,
            means=means,
            fallback_mean=np.asarray(fallback, dtype=np.float64).reshape(-1),
            counts={str(key): int(value) for key, value in (meta.get("counts") or {}).items()},
            renormalize=bool(meta.get("renormalize", True)),
        )


@dataclass(frozen=True)
class ClusterRouting:
    """Каждый кластер принадлежит своей группе; чужие кластеры недоступны.

    Так выглядит «кластеризация внутри слоя», сохранённая одной моделью:
    центроиды всех слоёв лежат в одной матрице, а cluster_groups говорит, какому
    слою какой центроид принадлежит. Без этой таблицы документ на русском мог бы
    уехать в кластер, обученный на таджикских текстах, — и именно тем способом,
    ради борьбы с которым слои и разделяли.

    Незнакомая группа здесь — отказ, а не запасной вариант. Модели для неё не
    существует; ближайший центроид среди чужих был бы ответом наугад, а по метке
    кластера этого было бы не видно.
    """

    fields: tuple[str, ...]
    cluster_groups: tuple[str, ...]

    kind = "cluster_routing"

    @property
    def groups(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.cluster_groups)))

    def mask(self, keys: Sequence[str]) -> np.ndarray:
        """(n_samples, n_clusters) — True там, где кластер документу разрешён."""
        unknown = sorted({key for key in keys if key not in set(self.cluster_groups)})
        if unknown:
            raise ValueError(
                f"нет модели для групп {unknown}; известны {list(self.groups)}"
            )
        owners = np.asarray(self.cluster_groups)
        return np.asarray(keys)[:, None] == owners[None, :]

    def meta(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "fields": list(self.fields),
            "cluster_groups": list(self.cluster_groups),
        }

    @classmethod
    def from_saved(cls, meta: dict[str, Any]) -> "ClusterRouting":
        return cls(
            fields=tuple(str(value) for value in meta["fields"]),
            cluster_groups=tuple(str(value) for value in meta["cluster_groups"]),
        )


def assign(
    X: np.ndarray,
    centroids: np.ndarray,
    *,
    normalize: bool = True,
    allowed: np.ndarray | None = None,
) -> np.ndarray:
    """Ближайший центроид, при необходимости — только среди разрешённых.

    Расстояния берутся у squared_distances из kmeans.py, а не считаются заново:
    вторая реализация той же формулы разошлась бы с первой на первой же правке,
    и разошлась бы молча — метки продолжали бы считаться, просто другие.
    """
    data = np.asarray(X, dtype=np.float64)
    data = l2_normalize(data) if normalize else data
    distances = squared_distances(data, np.asarray(centroids, dtype=np.float64))
    if allowed is not None:
        if allowed.shape != distances.shape:
            raise ValueError("маска разрешённых кластеров не совпадает по форме")
        if not allowed.any(axis=1).all():
            raise ValueError("у части документов нет ни одного разрешённого кластера")
        distances = np.where(allowed, distances, np.inf)
    return np.argmin(distances, axis=1)
