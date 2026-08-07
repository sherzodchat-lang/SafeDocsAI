"""Проекция на главные оси в артефакте: пишет обучение — читает бой.

Главная проверка файла — сквозная: модель сохраняется обучающей стороной
(pipeline/model_io.py) и читается боевой (topics/service.py), и обе применяют
проекцию к одному вектору с одинаковым результатом. Это не формальность.
Прежняя модель имела ДВА разных читателя одного файла, и когда обучающая
сторона назвала преобразование group_centering, а боевая знала только
group_mean_shift, раздел тем молча остался на предыдущей версии модели.

Вторая по важности — про длину вектора. Обучение считает проекцию на векторах
единичной длины, а вектор боевого документа — среднее векторов его фрагментов,
и длина у него меньше единицы. Вычесть среднее единичных векторов из вектора
длины 0.8 значит сдвинуть его дальше, чем сдвигало обучение, то есть получить
другую тему. Ошибка молчаливая: номер кластера посчитается.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from topicfixtures import ARTIFACT_EMBEDDING_MODEL, write_artifact  # noqa: E402

from app.modules.topics.pca import (  # noqa: E402
    PROJECTION_KIND,
    PrincipalAxes,
    Projection,
)
from app.modules.topics.pipeline.model_io import ClusterTopic, TopicModel  # noqa: E402
from app.modules.topics.service import (  # noqa: E402
    TRANSFORM_PCA_PROJECTION,
    TopicModelUnusable,
    forget_cached_artifacts,
    load_artifact,
)

DIM = 24
KEEP = 4
DROP = 2


def corpus(seed=0, n=120, dim=DIM):
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    scales = np.full(dim, 0.2)
    scales[:6] = (30.0, 12.0, 6.0, 3.0, 2.0, 1.5)
    points = (rng.normal(size=(n, dim)) * scales) @ rotation.T
    return points / np.linalg.norm(points, axis=1, keepdims=True)


class ProjectionArtifactTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.addCleanup(forget_cached_artifacts)
        self.path = Path(self._tmpdir.name) / "topic_model.npz"
        self.points = corpus()
        self.axes = PrincipalAxes.fit(self.points, n_components=DROP + KEEP)
        self.projection = self.axes.to_projection(drop=DROP, keep=KEEP)
        self.space = self.projection.apply(self.points)
        # Центроиды берём как средние трёх кусков — их геометрия не важна,
        # важно, что они лежат в ПРОСТРАНСТВЕ ПРОЕКЦИИ, а не в исходном.
        self.centroids = np.vstack(
            [self.space[:40].mean(0), self.space[40:80].mean(0), self.space[80:].mean(0)]
        )

    def write(self, **overrides) -> Path:
        meta_overrides = {
            "version": 3,
            "projection": self.projection.meta(),
            "n_clusters": int(self.centroids.shape[0]),
            "dim": int(self.centroids.shape[1]),
        }
        meta_overrides.update(overrides.pop("meta_overrides", {}))
        arrays = {
            "projection_mean": self.projection.mean,
            "projection_basis": self.projection.basis,
        }
        arrays.update(overrides.pop("arrays", {}))
        return write_artifact(
            self.path,
            centroids=self.centroids,
            arrays=arrays,
            meta_overrides=meta_overrides,
            **overrides,
        )


class ReadingTests(ProjectionArtifactTestCase):
    def test_artifact_with_a_projection_is_read_and_applied(self):
        self.write()
        artifact = load_artifact(self.path)
        self.assertEqual(artifact.transform.kind, TRANSFORM_PCA_PROJECTION)
        self.assertEqual(artifact.transform.projection.out_dim, KEEP)
        self.assertEqual(artifact.transform.projection.dropped, DROP)
        # Центроиды живут в пространстве проекции, а не в исходном.
        self.assertEqual(artifact.dim, KEEP)

    def test_the_cluster_matches_the_one_computed_by_hand(self):
        """Артефакт обязан давать тот же ответ, что прямой расчёт по тем же
        среднему и базису. Иначе «применить модель» означает что-то своё."""
        self.write()
        artifact = load_artifact(self.path)
        for index in (0, 17, 60, 119):
            with self.subTest(row=index):
                expected = int(
                    np.argmin(
                        ((self.space[index] - self.centroids) ** 2).sum(axis=1)
                    )
                )
                self.assertEqual(artifact.assign(self.points[index], group=None), expected)

    def test_description_says_what_was_dropped_and_what_is_left(self):
        self.write()
        artifact = load_artifact(self.path)
        self.assertIn(str(DROP), artifact.transform.description)
        self.assertIn(str(KEEP), artifact.transform.description)

    def test_a_projection_model_asks_the_document_for_nothing(self):
        """Ради этого всё и затевалось: прежняя модель требовала знать язык и
        жанр, а жанр боевому документу приходилось назначать допущением."""
        self.write()
        artifact = load_artifact(self.path)
        without = artifact.assign(self.points[3], group=None)
        for group in ("ru", "tg", "en|real", "чепуха"):
            with self.subTest(group=group):
                self.assertEqual(artifact.assign(self.points[3], group=group), without)


class VectorLengthTests(ProjectionArtifactTestCase):
    def test_a_shorter_vector_lands_in_the_same_cluster(self):
        """Вектор боевого документа — среднее векторов фрагментов, его длина
        меньше единицы. Без приведения к единичной длине он сдвинулся бы
        относительно среднего сильнее, чем при обучении."""
        self.write()
        artifact = load_artifact(self.path)
        self.assertTrue(artifact.transform.unit_input)
        for index in (5, 33, 91):
            for scale in (0.35, 0.8, 1.7):
                with self.subTest(row=index, scale=scale):
                    self.assertEqual(
                        artifact.assign(self.points[index] * scale, group=None),
                        artifact.assign(self.points[index], group=None),
                    )

    def test_without_unit_input_the_answer_would_differ(self):
        """Краснота предыдущего теста показана на числах: если длину не
        приводить, часть документов уезжает в другой кластер. Значит проверка
        выше не тавтология."""
        self.write()
        artifact = load_artifact(self.path)
        raw = Projection(
            mean=self.projection.mean,
            basis=self.projection.basis,
            dropped=DROP,
            renormalize=True,
        )
        moved = 0
        for index in range(len(self.points)):
            short = self.points[index] * 0.3
            honest = artifact.assign(self.points[index], group=None)
            naive = int(
                np.argmin(((raw.apply(short) - self.centroids) ** 2).sum(axis=1))
            )
            moved += int(honest != naive)
        self.assertGreater(moved, 0, "укорочение вектора обязано менять ответ без нормировки")


class RefusalTests(ProjectionArtifactTestCase):
    def test_projection_declared_without_arrays_is_refused(self):
        write_artifact(
            self.path,
            centroids=self.centroids,
            meta_overrides={"version": 3, "projection": self.projection.meta()},
        )
        with self.assertRaises(TopicModelUnusable) as caught:
            load_artifact(self.path)
        self.assertIn("projection_mean", str(caught.exception))

    def test_projection_and_group_centering_together_are_refused(self):
        """Два преобразования сразу — это не «одно поверх другого», а
        пространство, которого при обучении не было."""
        self.write(
            meta_overrides={
                "transform": {"kind": "group_centering", "fields": ["language"], "keys": ["ru"]}
            }
        )
        with self.assertRaises(TopicModelUnusable):
            load_artifact(self.path)

    def test_basis_that_disagrees_with_its_own_metadata_is_refused(self):
        self.write(meta_overrides={"projection": {**self.projection.meta(), "components": 99}})
        with self.assertRaises(TopicModelUnusable) as caught:
            load_artifact(self.path)
        self.assertIn("несогласована", str(caught.exception))

    def test_mean_from_another_space_is_refused(self):
        self.write(arrays={"projection_mean": np.zeros(DIM + 1)})
        with self.assertRaises(TopicModelUnusable):
            load_artifact(self.path)

    def test_projection_of_the_wrong_kind_is_refused(self):
        self.write(meta_overrides={"projection": {**self.projection.meta(), "kind": "магия"}})
        with self.assertRaises(TopicModelUnusable):
            load_artifact(self.path)


class MarginTests(ProjectionArtifactTestCase):
    def raise_back(self, point: np.ndarray) -> np.ndarray:
        """Точка пространства модели, поднятая обратно в исходное.

        Базис ортонормирован, поэтому подъём — это умножение на него. Обратно
        точно тот же вектор не получится (проекция теряет всё, что лежало вне
        базиса, и нормирует остаток), и это не мешает: нужен документ, чья
        проекция стоит там, где сказано.
        """
        return self.projection.mean + self.projection.basis @ point

    def test_the_middle_between_two_topics_is_far_less_certain_than_a_centre(self):
        """Смысл запаса: он должен РАЗЛИЧАТЬ уверенный случай и спорный.

        Проверяется отношение, а не абсолютное число. Абсолютное зависит от
        того, как далеко разошлись центроиды в этом конкретном наборе, и
        зашитый в тест порог означал бы «модель обязана быть настолько же
        уверена», чего от неё никто не требует.
        """
        self.write()
        artifact = load_artifact(self.path)
        at_centre = artifact.assign_with_margin(self.raise_back(self.centroids[0]), group=None)[1]
        middle = (self.centroids[0] + self.centroids[1]) / 2
        between = artifact.assign_with_margin(self.raise_back(middle), group=None)[1]
        self.assertLess(between, 0.05)
        self.assertGreater(at_centre, between * 3)

    def test_the_threshold_refuses_the_middle_and_keeps_the_centre(self):
        """То же самое, но так, как это работает в бою: через порог."""
        self.write(meta_overrides={"params": {"k": 3, "margin_threshold": 0.10}})
        artifact = load_artifact(self.path)
        middle = (self.centroids[0] + self.centroids[1]) / 2
        _, between = artifact.assign_with_margin(self.raise_back(middle), group=None)
        _, at_centre = artifact.assign_with_margin(self.raise_back(self.centroids[0]), group=None)
        self.assertFalse(artifact.is_confident(between))
        self.assertTrue(artifact.is_confident(at_centre))

    def test_threshold_is_read_from_the_model(self):
        self.write(meta_overrides={"params": {"k": 3, "margin_threshold": 0.25}})
        artifact = load_artifact(self.path)
        self.assertAlmostEqual(artifact.margin_threshold, 0.25)
        self.assertFalse(artifact.is_confident(0.24))
        self.assertTrue(artifact.is_confident(0.25))

    def test_no_threshold_means_always_confident(self):
        """Артефакты, обученные до появления порога, обязаны работать как
        раньше, а не начать молча отказывать."""
        self.write()
        artifact = load_artifact(self.path)
        self.assertIsNone(artifact.margin_threshold)
        self.assertTrue(artifact.is_confident(0.0))

    def test_a_nonsense_threshold_is_ignored_rather_than_obeyed(self):
        """Порог вне [0, 1) — описка, а не решение. Послушаться значило бы
        отказать всему корпусу разом."""
        for value in (5.0, -1.0, "много"):
            with self.subTest(value=value):
                self.write(meta_overrides={"params": {"k": 3, "margin_threshold": value}})
                forget_cached_artifacts()
                artifact = load_artifact(self.path)
                self.assertIsNone(artifact.margin_threshold)

    def test_margin_never_leaves_its_range(self):
        self.write()
        artifact = load_artifact(self.path)
        values = [artifact.assign_with_margin(point, group=None)[1] for point in self.points]
        self.assertTrue(all(0.0 <= value <= 1.0 for value in values), min(values))


class TrainingAndProductionAgreeTests(ProjectionArtifactTestCase):
    """Сквозная проверка: пишет model_io, читает service, ответ один и тот же."""

    def test_saved_by_the_trainer_is_read_by_the_service(self):
        model = TopicModel(
            centroids=self.centroids,
            embedding_model=ARTIFACT_EMBEDDING_MODEL,
            normalize=True,
            cluster_topics=tuple(
                ClusterTopic(
                    cluster=index,
                    topic_id=f"R0{index}",
                    topic=f"тема {index}",
                    topic_ru=f"тема {index}",
                    topic_tg=f"мавзуъ {index}",
                    share=0.8,
                    size=40,
                )
                for index in range(self.centroids.shape[0])
            ),
            projection=self.projection,
            params={"k": 3, "margin_threshold": 0.05},
        )
        model.save(self.path)

        artifact = load_artifact(self.path)
        self.assertEqual(artifact.transform.kind, PROJECTION_KIND)
        self.assertAlmostEqual(artifact.margin_threshold, 0.05)
        self.assertEqual(artifact.label_in(0, "tg"), "мавзуъ 0")

        by_trainer = model.predict(self.points)
        by_service = [artifact.assign(point, group=None) for point in self.points]
        self.assertEqual(list(map(int, by_trainer)), by_service)

    def test_the_trainer_refuses_to_mix_projection_with_group_centering(self):
        from app.modules.topics.pipeline.transforms import GroupCentering

        centering = GroupCentering(
            fields=("language",),
            keys=("ru",),
            means=np.zeros((1, KEEP)),
            fallback_mean=np.zeros(KEEP),
            counts=(1,),
        )
        with self.assertRaises(ValueError):
            TopicModel(
                centroids=self.centroids,
                embedding_model=ARTIFACT_EMBEDDING_MODEL,
                normalize=True,
                cluster_topics=(),
                projection=self.projection,
                transform=centering,
            )

    def test_the_trainer_refuses_centroids_from_another_space(self):
        with self.assertRaises(ValueError):
            TopicModel(
                centroids=np.zeros((3, KEEP + 1)),
                embedding_model=ARTIFACT_EMBEDDING_MODEL,
                normalize=True,
                cluster_topics=(),
                projection=self.projection,
            )

    def test_saved_meta_names_the_projection_and_empty_required_fields(self):
        model = TopicModel(
            centroids=self.centroids,
            embedding_model=ARTIFACT_EMBEDDING_MODEL,
            normalize=True,
            cluster_topics=(),
            projection=self.projection,
        )
        model.save(self.path)
        with np.load(self.path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["meta"]))
            self.assertIn("projection_mean", archive)
            self.assertIn("projection_basis", archive)
        self.assertEqual(meta["version"], 3)
        self.assertEqual(meta["projection"]["kind"], PROJECTION_KIND)
        self.assertEqual(meta["required_fields"], [])

    def test_a_round_trip_through_the_trainer_keeps_the_projection(self):
        model = TopicModel(
            centroids=self.centroids,
            embedding_model=ARTIFACT_EMBEDDING_MODEL,
            normalize=True,
            cluster_topics=(),
            projection=self.projection,
        )
        model.save(self.path)
        restored = TopicModel.load(self.path)
        np.testing.assert_allclose(restored.projection.basis, self.projection.basis)
        np.testing.assert_allclose(restored.projection.mean, self.projection.mean)
        self.assertEqual(restored.projection.dropped, DROP)
        self.assertEqual(restored.required_fields, ())


if __name__ == "__main__":
    unittest.main()
