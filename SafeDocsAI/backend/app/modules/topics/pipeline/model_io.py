"""Сохранение и загрузка обученной модели тем.

Обученная модель — это не только центроиды. Чтобы применить её к новому
документу, нужно знать ещё четыре вещи, и любая забытая делает предсказание
бессмысленным, не вызвав при этом ни одной ошибки:

  * какой моделью считались эмбеддинги (вектор от другой модели лежит в другом
    пространстве, а иногда и просто другой длины);
  * нормировались ли векторы перед кластеризацией (без нормировки «ближайший
    центроид» считается по другой геометрии);
  * что означает номер кластера — номер сам по себе не тема, соответствие
    «кластер -> преобладающая тема» получено из разметки обучающей выборки и
    живёт вместе с центроидами;
  * какое преобразование применялось к векторам перед кластеризацией. Центроиды
    модели, обученной на векторах с вычтенным средним по языку, лежат в другом
    пространстве, чем сырой эмбеддинг нового документа: predict без того же
    вычитания вернёт метки, посчитанные не пойми где.

Поэтому формат — один .npz: матрица центроидов плюс строка JSON с остальным
плюс, если преобразование есть, его матрицы. Два файла рядом разъехались бы при
первом же копировании.

Версия формата 2 добавила преобразование и маршрутизацию. Файлы версии 1
читаются как «преобразования нет» — это не догадка, а факт: в них его и не
могло быть. Обратное направление закрыто: старый код на файл версии 2 ответит
отказом, а не тихо потерянным преобразованием.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app.modules.topics.kmeans import KMeans
from app.modules.topics.pipeline.transforms import (
    ClusterRouting,
    GroupCentering,
    assign,
    group_keys,
)

MODEL_FORMAT_VERSION = 2

# Какие версии формата этот код умеет читать. Версия 1 отличается только
# отсутствием преобразования, и её содержимое от этого не теряется.
READABLE_MODEL_VERSIONS = (1, 2)


@dataclass(frozen=True)
class ClusterTopic:
    """Чем оказался кластер по разметке обучающей выборки.

    share — доля преобладающей темы внутри кластера. Она не украшение: кластер
    с share = 0.95 можно называть темой, кластер с share = 0.30 — это свалка, и
    подписывать его именем большинства значит врать пользователю. Число
    сохраняется вместе с именем, чтобы приложение могло решать само.
    """

    cluster: int
    topic_id: str
    topic: str
    share: float
    size: int
    # Та же тема на языках интерфейса. Все названия пишутся В АРТЕФАКТ, а не
    # добираются приложением из корпуса: корпус лежит в репозитории обучения, а
    # артефакт уезжает на сервер один, и слой выдачи не имеет права зависеть от
    # файла, которого рядом с ним может не быть.
    #
    # Языков ровно два, и список закрыт не здесь, а в продукте: интерфейс
    # переведён на русский и таджикский, английского в нём нет вовсе. topic
    # остаётся ключом — устойчивым именем темы, по которому сходятся отчёты
    # эксперимента и назначения прошлых версий.
    #
    # Со значениями по умолчанию — чтобы артефакты, обученные до появления этих
    # полей, читались как «перевода нет», а не падали.
    topic_ru: str = ""
    topic_tg: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "cluster": self.cluster,
            "topic_id": self.topic_id,
            "topic": self.topic,
            "topic_ru": self.topic_ru,
            "topic_tg": self.topic_tg,
            "share": self.share,
            "size": self.size,
        }


@dataclass
class TopicModel:
    """Центроиды плюс всё, без чего их нельзя применить."""

    centroids: np.ndarray
    embedding_model: str
    normalize: bool
    cluster_topics: tuple[ClusterTopic, ...]
    params: dict[str, Any] = field(default_factory=dict)
    transform: GroupCentering | None = None
    routing: ClusterRouting | None = None

    def __post_init__(self) -> None:
        if (
            self.transform is not None
            and self.routing is not None
            and self.transform.fields != self.routing.fields
        ):
            # Ключ группы собирается из полей ОДНИМ порядком. Разные наборы у
            # преобразования и у маршрутизации означали бы два разных ключа для
            # одного документа, и один из них не нашёлся бы в таблице — либо,
            # что хуже, нашёлся бы чужой.
            raise ValueError(
                f"преобразование считает группу по {list(self.transform.fields)}, "
                f"а маршрутизация по {list(self.routing.fields)}"
            )

    @property
    def n_clusters(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def dim(self) -> int:
        return int(self.centroids.shape[1])

    @property
    def required_fields(self) -> tuple[str, ...]:
        """Что приложение обязано знать о документе помимо его текста.

        Пусто — модель применима к голому эмбеддингу. Непусто — перед predict
        придётся определить перечисленные признаки (язык — определителем языка,
        происхождение — по источнику загрузки). Это требование к встраиванию, и
        оно должно читаться из самой модели, а не из отчёта, который к моменту
        внедрения потеряется.
        """
        names: list[str] = []
        for part in (self.transform, self.routing):
            for name in getattr(part, "fields", ()):  # None -> пустой кортеж
                if name not in names:
                    names.append(name)
        return tuple(names)

    def _kmeans(self) -> KMeans:
        """Восстановленный KMeans с готовыми центроидами.

        Предсказание идёт через ту же KMeans.predict, что и при обучении, а не
        через отдельную копию «найти ближайший центроид». Копия разошлась бы с
        оригиналом на первой же правке нормировки, и расхождение проявилось бы
        только в проде — метками, которые вроде бы посчитались.

        Отдельного конструктора «модель из центроидов» в kmeans.py нет (см.
        отчёт), поэтому центроиды кладутся в атрибут напрямую. Атрибут
        публичный и задокументированный, приватных потрохов здесь не трогаем.
        """
        model = KMeans(n_clusters=self.n_clusters, normalize=self.normalize)
        model.centroids_ = self.centroids
        return model

    def predict(
        self,
        X: np.ndarray,
        *,
        documents: Sequence[Any] | None = None,
        groups: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Метки кластеров; при наличии преобразования — с учётом группы.

        Модели с преобразованием нельзя дать одни только векторы: молча
        применить центрирование не к тому языку — значит получить метки, по
        которым ошибку не видно. Поэтому здесь отказ, а не умолчание.

        Группу можно передать двумя способами: documents — объекты с нужными
        атрибутами (как в эксперименте), groups — уже готовые ключи вида
        "ru|real" (как в приложении, где язык определён на лету).
        """
        if groups is None and documents is not None:
            groups = group_keys(documents, self.required_fields)
        if self.required_fields and groups is None:
            raise ValueError(
                "модель применяет преобразование по группе и требует признаки "
                f"{list(self.required_fields)}: передайте documents или groups"
            )
        if groups is not None and len(groups) != np.asarray(X).shape[0]:
            raise ValueError("групп передано не столько же, сколько строк матрицы")

        data = np.asarray(X, dtype=np.float64)
        if self.transform is not None:
            data = self.transform.apply_to_keys(list(groups or []), data)
        allowed = self.routing.mask(list(groups or [])) if self.routing is not None else None
        if allowed is None:
            return self._kmeans().predict(data)
        return assign(data, self.centroids, normalize=self.normalize, allowed=allowed)

    def topic_of(self, cluster: int) -> ClusterTopic | None:
        for item in self.cluster_topics:
            if item.cluster == cluster:
                return item
        return None

    def save(self, path: str | Path) -> None:
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        meta = json.dumps(
            {
                "version": MODEL_FORMAT_VERSION,
                "embedding_model": self.embedding_model,
                "normalize": self.normalize,
                "n_clusters": self.n_clusters,
                "dim": self.dim,
                "cluster_topics": [item.as_dict() for item in self.cluster_topics],
                "params": self.params,
                "transform": self.transform.meta() if self.transform is not None else None,
                "routing": self.routing.meta() if self.routing is not None else None,
                "required_fields": list(self.required_fields),
            },
            ensure_ascii=False,
        )
        arrays: dict[str, np.ndarray] = {
            "centroids": np.asarray(self.centroids, dtype=np.float64),
            "meta": np.array(meta),
        }
        if self.transform is not None:
            # Средние по группам — часть модели, а не справочная информация:
            # без них центроиды применить не к чему.
            arrays["transform_means"] = np.asarray(self.transform.means, dtype=np.float64)
            arrays["transform_fallback"] = np.asarray(
                self.transform.fallback_mean, dtype=np.float64
            )
        temporary = file.with_suffix(file.suffix + ".tmp")
        with open(temporary, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(file)

    @classmethod
    def load(cls, path: str | Path) -> "TopicModel":
        with np.load(Path(path), allow_pickle=False) as archive:
            meta = json.loads(str(archive["meta"]))
            centroids = np.asarray(archive["centroids"], dtype=np.float64)
            transform_means = (
                np.asarray(archive["transform_means"], dtype=np.float64)
                if "transform_means" in archive
                else None
            )
            transform_fallback = (
                np.asarray(archive["transform_fallback"], dtype=np.float64)
                if "transform_fallback" in archive
                else None
            )

        version = meta.get("version")
        if version not in READABLE_MODEL_VERSIONS:
            # Читать чужую версию «как получится» нельзя: раскладка полей
            # меняется вместе с версией, и недостающий normalize превратился бы
            # в False, то есть в тихо другую геометрию.
            raise ValueError(
                f"формат модели {version!r}, читаются {list(READABLE_MODEL_VERSIONS)}"
            )
        if centroids.ndim != 2:
            raise ValueError("центроиды должны быть матрицей (n_clusters, n_features)")
        if int(meta["n_clusters"]) != centroids.shape[0]:
            raise ValueError("n_clusters в метаданных не совпадает с числом центроидов")

        transform_meta = meta.get("transform")
        if transform_meta is not None and transform_means is None:
            # Метаданные обещают преобразование, а матриц средних в файле нет.
            # Применить модель без них нельзя, а сделать вид, что преобразования
            # не было, — значит выдать метки из другого пространства.
            raise ValueError("в файле описано преобразование, но нет его средних векторов")
        transform = (
            GroupCentering.from_saved(transform_meta, transform_means, transform_fallback)
            if transform_meta is not None
            else None
        )
        routing_meta = meta.get("routing")
        routing = ClusterRouting.from_saved(routing_meta) if routing_meta is not None else None
        if routing is not None and len(routing.cluster_groups) != centroids.shape[0]:
            raise ValueError("маршрутизация описывает не столько кластеров, сколько центроидов")

        return cls(
            centroids=centroids,
            embedding_model=str(meta["embedding_model"]),
            normalize=bool(meta["normalize"]),
            cluster_topics=tuple(
                ClusterTopic(
                    cluster=int(item["cluster"]),
                    topic_id=str(item["topic_id"]),
                    topic=str(item["topic"]),
                    topic_ru=str(item.get("topic_ru") or ""),
                    topic_tg=str(item.get("topic_tg") or ""),
                    share=float(item["share"]),
                    size=int(item["size"]),
                )
                for item in meta.get("cluster_topics", [])
            ),
            params=dict(meta.get("params", {})),
            transform=transform,
            routing=routing,
        )


def dominant_topics(
    labels: Sequence[int],
    topic_ids: Sequence[str],
    topic_names: Sequence[str],
    n_clusters: int,
    topic_names_ru: Sequence[str] | None = None,
    topic_names_tg: Sequence[str] | None = None,
) -> tuple[ClusterTopic, ...]:
    """Соответствие «кластер -> преобладающая тема» по обучающей выборке.

    Пустой кластер (такое возможно, когда k больше числа различимых групп)
    получает пустую тему и size = 0, а не пропускается: приложение обязано
    уметь ответить на любой номер, который может вернуть predict.

    Переводы необязательны, и это не послабление: все колонки выровнены по
    документам, поэтому имя на каждом языке выбирается ТЕМ ЖЕ индексом, что и
    основное, — иначе кластер получил бы английское имя одной темы и русское
    другой, и расхождение было бы видно только человеку, читающему оба.
    """
    result: list[ClusterTopic] = []
    labels_array = np.asarray(labels)
    for cluster in range(n_clusters):
        members = np.flatnonzero(labels_array == cluster)
        if members.size == 0:
            result.append(ClusterTopic(cluster, "", "", 0.0, 0))
            continue
        counts: dict[str, int] = {}
        for index in members:
            key = topic_ids[int(index)]
            counts[key] = counts.get(key, 0) + 1
        # При равенстве берётся меньший topic_id — иначе повторный запуск на тех
        # же данных мог бы подписать кластер по-другому.
        best_topic = min(counts, key=lambda key: (-counts[key], key))
        name = ""
        name_ru = ""
        name_tg = ""
        for index in members:
            if topic_ids[int(index)] == best_topic:
                name = topic_names[int(index)]
                if topic_names_ru is not None:
                    name_ru = str(topic_names_ru[int(index)] or "")
                if topic_names_tg is not None:
                    name_tg = str(topic_names_tg[int(index)] or "")
                break
        result.append(
            ClusterTopic(
                cluster=cluster,
                topic_id=best_topic,
                topic=name,
                topic_ru=name_ru,
                topic_tg=name_tg,
                share=float(counts[best_topic] / members.size),
                size=int(members.size),
            )
        )
    return tuple(result)
