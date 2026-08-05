"""Чтение артефакта модели тем и применение преобразования.

Главная проверка файла — не «загрузчик читает поля», а вот эта:
ЗАБЫТОЕ ПРЕОБРАЗОВАНИЕ НЕ ПАДАЕТ, ОНО ВРЁТ. Слой назначения, сравнивающий сырой
вектор с центроидами преобразованного пространства, отвечает номером кластера
без единой ошибки в журнале — и номер этот другой. Поэтому здесь стоит тест,
который показывает расхождение на числах (см.
TransformIsNotDecorationTests), и тест, который требует отказа на незнакомом
преобразовании: молчаливое «сделаем ничего» — это ровно тот же неверный ответ,
только полученный от собственного кода.

Ни базы, ни сети: артефакт пишется на диск во временный каталог, а
кластеризация здесь своя (app/modules/topics/kmeans.py).
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from topicfixtures import (  # noqa: E402
    ARTIFACT_EMBEDDING_MODEL,
    CENTROIDS,
    GROUP_MEANS,
    LABELS,
    METRICS,
    document_vector_for,
    write_artifact,
    write_language_artifact,
)

from app.modules.topics.service import (  # noqa: E402
    TRANSFORM_GROUP_MEAN_SHIFT,
    TRANSFORM_MEAN_SHIFT,
    TRANSFORM_NONE,
    TopicEmbeddingUnavailable,
    TopicModelUnusable,
    default_label,
    document_vector,
    forget_cached_artifacts,
    load_artifact,
    load_artifact_cached,
    topic_model_path,
)


class ArtifactTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.addCleanup(forget_cached_artifacts)
        self.path = Path(self._tmpdir.name) / "topic_model.npz"


class ArtifactReadingTests(ArtifactTestCase):
    def test_reads_everything_needed_to_apply_the_model(self):
        write_language_artifact(self.path)
        artifact = load_artifact(self.path)

        self.assertEqual(artifact.embedding_model, ARTIFACT_EMBEDDING_MODEL)
        self.assertTrue(artifact.normalize)
        self.assertEqual(artifact.cluster_count, len(CENTROIDS))
        self.assertEqual(artifact.k, len(CENTROIDS))
        self.assertEqual(artifact.dim, CENTROIDS.shape[1])
        self.assertEqual(artifact.metrics, METRICS)
        self.assertEqual(
            [artifact.label_of(index) for index in range(len(LABELS))], list(LABELS)
        )
        self.assertEqual(artifact.transform.kind, TRANSFORM_GROUP_MEAN_SHIFT)

    def test_cluster_without_a_name_gets_its_number(self):
        """Безымянная строка распределения неотличима от соседней безымянной."""
        write_artifact(
            self.path,
            meta_overrides={"cluster_topics": [{"cluster": 0, "topic": "", "topic_id": ""}]},
        )
        artifact = load_artifact(self.path)
        self.assertEqual(artifact.label_of(0), default_label(0))
        self.assertEqual(artifact.label_of(2), default_label(2))

    def test_unknown_format_version_is_still_read(self):
        """Артефакт кладут отдельно от выкладки бэкенда.

        Поднятая версия формата не должна гасить раздел, который умеет
        прочитать нужные ему поля: строгость раздела вынесена туда, где ошибка
        тихая, — на преобразование.
        """
        write_language_artifact(self.path)
        with np.load(self.path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["meta"]))
            payload = {name: archive[name] for name in archive.keys() if name != "meta"}
        meta["version"] = 999
        payload["meta"] = np.array(json.dumps(meta, ensure_ascii=False))
        np.savez_compressed(self.path, **payload)
        forget_cached_artifacts()

        self.assertEqual(load_artifact(self.path).cluster_count, len(CENTROIDS))

    def test_artifact_without_embedding_model_is_refused(self):
        """Вектор от другой модели даёт правдоподобный, но чужой кластер."""
        write_artifact(self.path, meta_overrides={"embedding_model": ""})
        with self.assertRaises(TopicModelUnusable):
            load_artifact(self.path)

    def test_metrics_that_were_not_computed_come_out_as_null(self):
        """Ноль вместо «не считали» — это оценка, которой никто не получал."""
        write_artifact(
            self.path,
            meta_overrides={"metrics": {"purity": float("nan"), "ari_topic": 0.5}},
        )
        metrics = load_artifact(self.path).metrics
        self.assertEqual(metrics["ari_topic"], 0.5)
        self.assertIsNone(metrics["purity"])
        self.assertIsNone(metrics["silhouette"])

    def test_metrics_are_also_found_next_to_the_training_parameters(self):
        """Метрики считает эксперимент, и кладёт он их туда, куда удобно ему."""
        write_artifact(
            self.path,
            meta_overrides={"metrics": {}, "params": {"k": 3, "ari": 0.31, "purity": 0.7}},
        )
        metrics = load_artifact(self.path).metrics
        self.assertEqual(metrics["ari_topic"], 0.31)
        self.assertEqual(metrics["purity"], 0.7)

    def test_the_artifact_path_comes_from_the_environment(self):
        write_language_artifact(self.path)
        with patch.dict(os.environ, {"TOPIC_MODEL_PATH": str(self.path)}):
            self.assertEqual(topic_model_path(), self.path)
            self.assertEqual(load_artifact_cached().path, str(self.path))

    def test_rewritten_artifact_is_re_read(self):
        """Кэш не имеет права пережить переобучение."""
        write_language_artifact(self.path)
        first = load_artifact_cached(self.path)
        write_artifact(self.path, centroids=CENTROIDS[:2])
        second = load_artifact_cached(self.path)
        self.assertNotEqual(first.digest, second.digest)
        self.assertEqual(second.cluster_count, 2)


class TransformParsingTests(ArtifactTestCase):
    def test_no_transform_declared_means_no_transform(self):
        write_artifact(self.path)
        artifact = load_artifact(self.path)
        self.assertEqual(artifact.transform.kind, TRANSFORM_NONE)
        vector = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(
            artifact.transform.apply(vector, group="ru"), vector
        )

    def test_mean_shift_accepts_the_vector_inline(self):
        write_artifact(
            self.path,
            transform={"kind": TRANSFORM_MEAN_SHIFT, "mean": [1.0, 1.0, 1.0, 1.0]},
        )
        artifact = load_artifact(self.path)
        np.testing.assert_allclose(
            artifact.transform.apply(np.array([2.0, 2.0, 2.0, 2.0]), group=None),
            np.ones(4),
        )

    def test_mean_shift_accepts_the_vector_as_an_array(self):
        """Массив на 4096 чисел в JSON — это сотни килобайт и потерянная точность."""
        write_artifact(
            self.path,
            transform={"kind": TRANSFORM_MEAN_SHIFT},
            arrays={"transform_mean": [1.0, 1.0, 1.0, 1.0]},
        )
        artifact = load_artifact(self.path)
        np.testing.assert_allclose(
            artifact.transform.apply(np.array([3.0, 3.0, 3.0, 3.0]), group=None),
            np.full(4, 2.0),
        )

    def test_mean_shift_without_a_vector_is_refused(self):
        write_artifact(self.path, transform={"kind": TRANSFORM_MEAN_SHIFT})
        with self.assertRaises(TopicModelUnusable):
            load_artifact(self.path)

    def test_group_means_may_come_inline(self):
        write_artifact(
            self.path,
            transform={"kind": TRANSFORM_GROUP_MEAN_SHIFT, "means": GROUP_MEANS},
        )
        artifact = load_artifact(self.path)
        np.testing.assert_allclose(
            artifact.transform.apply(np.array(document_vector_for("ru", 2)), group="ru"),
            [0.0, 0.0, 1.0, 0.0],
        )

    def test_unknown_group_falls_back_instead_of_losing_the_topic(self):
        """Документ на четвёртом языке — редкость, ради которой не гасят функцию."""
        write_artifact(
            self.path,
            transform={
                "kind": TRANSFORM_GROUP_MEAN_SHIFT,
                "means": GROUP_MEANS,
                "mean": [1.0, 0.0, 0.0, 0.0],
            },
        )
        artifact = load_artifact(self.path)
        np.testing.assert_allclose(
            artifact.transform.apply(np.array([2.0, 0.0, 0.0, 0.0]), group="de"),
            [1.0, 0.0, 0.0, 0.0],
        )

    def test_unknown_transform_kind_is_refused_instead_of_ignored(self):
        """Молчаливое «сделаем ничего» — тот же неверный ответ, но от нас."""
        write_artifact(self.path, transform={"kind": "whitening"})
        with self.assertRaises(TopicModelUnusable):
            load_artifact(self.path)

    def test_grouping_by_an_unknown_field_is_refused(self):
        """У документа слой назначения знает только язык."""
        write_artifact(
            self.path,
            transform={
                "kind": TRANSFORM_GROUP_MEAN_SHIFT,
                "group_field": "dataset_origin",
                "means": GROUP_MEANS,
            },
        )
        with self.assertRaises(TopicModelUnusable):
            load_artifact(self.path)

    def test_group_mean_shift_without_any_means_is_refused(self):
        write_artifact(self.path, transform={"kind": TRANSFORM_GROUP_MEAN_SHIFT})
        with self.assertRaises(TopicModelUnusable):
            load_artifact(self.path)

    def test_group_matrix_must_match_the_declared_groups(self):
        """Разъехавшаяся пара «имена групп / матрица» назначала бы чужие средние."""
        write_artifact(
            self.path,
            transform={
                "kind": TRANSFORM_GROUP_MEAN_SHIFT,
                "groups": ["en", "ru", "tg"],
            },
            arrays={"transform_group_means": [GROUP_MEANS["en"], GROUP_MEANS["ru"]]},
        )
        with self.assertRaises(TopicModelUnusable):
            load_artifact(self.path)

    def test_transform_description_names_the_grouping_field(self):
        """Строка уходит в API: 'group_mean_shift' без поля ничего не объясняет."""
        write_language_artifact(self.path)
        self.assertEqual(
            load_artifact(self.path).transform.description,
            f"{TRANSFORM_GROUP_MEAN_SHIFT}(language)",
        )


class TransformIsNotDecorationTests(ArtifactTestCase):
    """Доказательство красноты: без преобразования ответ ДРУГОЙ, а не ошибочный.

    Русское среднее в тестовом артефакте направлено вдоль нулевого центроида —
    ровно так и выглядит беда, ради которой преобразование заведено. Документ
    третьей темы на русском языке при честном применении модели попадает в свою
    тему, а при сравнении сырого вектора — в тему «всё русское».
    """

    def test_raw_vector_lands_in_a_different_cluster(self):
        write_language_artifact(self.path)
        with_transform = load_artifact(self.path)

        raw = np.array(document_vector_for("ru", 2))
        honest = with_transform.assign(raw, group="ru")

        forget_cached_artifacts()
        write_artifact(self.path)  # тот же файл, но преобразование не объявлено
        without_transform = load_artifact(self.path)
        naive = without_transform.assign(raw, group="ru")

        self.assertEqual(honest, 2, "документ третьей темы обязан попасть в неё")
        self.assertEqual(naive, 0, "сырой вектор уезжает в кластер своего языка")
        self.assertNotEqual(
            honest,
            naive,
            "если бы ответы совпадали, тест ничего не доказывал бы: "
            "проверять надо на данных, где забытое преобразование ВИДНО",
        )

    def test_every_language_finds_its_own_topic(self):
        """Иначе совпадение могло бы оказаться случайным на одном примере."""
        artifact = load_artifact(write_language_artifact(self.path))
        for language in GROUP_MEANS:
            for cluster in range(len(CENTROIDS)):
                with self.subTest(language=language, cluster=cluster):
                    vector = np.array(document_vector_for(language, cluster))
                    self.assertEqual(artifact.assign(vector, group=language), cluster)


class DocumentVectorTests(unittest.TestCase):
    def test_document_vector_is_the_mean_of_its_chunks(self):
        vector = document_vector([np.array([0.0, 2.0]), np.array([2.0, 0.0])])
        np.testing.assert_allclose(vector, [1.0, 1.0])

    def test_no_chunks_means_no_vector(self):
        self.assertIsNone(document_vector([]))

    def test_chunks_of_different_width_are_refused(self):
        """Смесь двух коллекций в одном документе: обе трактовки выдумывают данные."""
        with self.assertRaises(TopicEmbeddingUnavailable):
            document_vector([np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0])])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
