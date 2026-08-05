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

import dataclasses
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
    CELL_FALLBACK,
    CELL_MEANS,
    CENTROIDS,
    GROUP_MEANS,
    LABELS,
    LABELS_RU,
    LABELS_TG,
    METRICS,
    cell_document_vector_for,
    cluster_topics,
    document_vector_for,
    write_artifact,
    write_cell_artifact,
    write_language_artifact,
)

from app.modules.topics.service import (  # noqa: E402
    PRODUCTION_FIXED_FIELDS,
    TRANSFORM_GROUP_CENTERING,
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

    def test_translations_are_read_alongside_the_stable_label(self):
        """Подписи нужны сразу все: перевод — показать, ключ — сослаться."""
        write_language_artifact(self.path)
        artifact = load_artifact(self.path)

        self.assertEqual(
            [artifact.label_of(index) for index in range(len(LABELS))], list(LABELS)
        )
        self.assertEqual(
            [artifact.label_in(index, "ru") for index in range(len(LABELS))], list(LABELS_RU)
        )
        self.assertEqual(
            [artifact.label_in(index, "tg") for index in range(len(LABELS))], list(LABELS_TG)
        )

    def test_artifact_without_translations_answers_none_not_a_number(self):
        """None — сигнал клиенту откатиться к устойчивой подписи.

        default_label здесь был бы хуже пустоты: «Кластер 2» встало бы на место
        перевода, и клиент показал бы номер вместо настоящей темы, которая у
        него уже есть.
        """
        write_artifact(
            self.path, meta_overrides={"cluster_topics": cluster_topics(localized=False)}
        )
        artifact = load_artifact(self.path)

        self.assertEqual(artifact.label_of(1), LABELS[1])
        self.assertIsNone(artifact.label_in(1, "ru"))
        self.assertIsNone(artifact.label_in(1, "tg"))

    def test_blank_translation_is_the_same_as_none(self):
        topics = cluster_topics()
        topics[1]["topic_ru"] = "   "
        write_artifact(self.path, meta_overrides={"cluster_topics": topics})
        artifact = load_artifact(self.path)

        self.assertIsNone(artifact.label_in(1, "ru"))
        self.assertEqual(artifact.label_in(0, "ru"), LABELS_RU[0])
        # Таджикская подпись того же кластера на месте: языки независимы.
        self.assertEqual(artifact.label_in(1, "tg"), LABELS_TG[1])

    def test_a_language_the_product_does_not_have_is_simply_absent(self):
        """Артефакт может знать больше языков, чем показывает продукт."""
        write_language_artifact(self.path)
        artifact = load_artifact(self.path)
        self.assertIsNone(artifact.label_in(0, "en"))

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


class GroupCenteringTests(ArtifactTestCase):
    """Боевой артефакт группирует по ДВУМ полям, а документ знает одно.

    Разница закрывается допущением dataset_origin=real: синтетика существует
    только внутри обучающего корпуса, и загруженный пользователем документ
    синтетическим быть не может. Отсюда проекция шести средних на три.
    Проверяется здесь и сама проекция, и её граница: поле, за которым боевого
    значения не закреплено, обязано получать отказ.
    """

    def test_two_field_artifact_is_accepted(self):
        artifact = load_artifact(write_cell_artifact(self.path))
        self.assertEqual(artifact.transform.kind, TRANSFORM_GROUP_CENTERING)
        self.assertEqual(
            artifact.transform.description,
            f"{TRANSFORM_GROUP_CENTERING}(language; dataset_origin=real)",
            "подпись в реестре обязана называть допущение, а не только язык",
        )

    def test_six_cell_means_become_three_keyed_by_language(self):
        artifact = load_artifact(write_cell_artifact(self.path))
        means = artifact.transform.group_means or {}
        self.assertEqual(sorted(means), ["en", "ru", "tg"])
        for language in ("en", "ru", "tg"):
            np.testing.assert_allclose(means[language], CELL_MEANS[f"{language}|real"])

    def test_the_kept_half_is_the_real_one_and_it_matters(self):
        """Краснота: с синтетическим средним ответ ДРУГОЙ, а не «примерно тот же».

        Если бы проекция взяла не ту ячейку, отказа не случилось бы: номер
        кластера посчитался бы точно так же, просто другой.
        """
        artifact = load_artifact(write_cell_artifact(self.path))
        synthetic = {
            key.split("|")[0]: np.asarray(value)
            for key, value in CELL_MEANS.items()
            if key.endswith("synthetic")
        }
        wrong = dataclasses.replace(
            artifact, transform=dataclasses.replace(artifact.transform, group_means=synthetic)
        )
        for language in ("en", "ru", "tg"):
            with self.subTest(language=language):
                vector = np.array(cell_document_vector_for(language, 1))
                self.assertEqual(artifact.assign(vector, group=language), 1)
                self.assertEqual(wrong.assign(vector, group=language), 0)

    def test_every_language_finds_its_own_topic(self):
        artifact = load_artifact(write_cell_artifact(self.path))
        for language in ("en", "ru", "tg"):
            for cluster in range(len(CENTROIDS)):
                with self.subTest(language=language, cluster=cluster):
                    vector = np.array(cell_document_vector_for(language, cluster))
                    self.assertEqual(artifact.assign(vector, group=language), cluster)

    def test_tajik_documents_of_the_product_find_the_tajik_mean(self):
        """Внутри проекта таджикский — 'tj', в датасете и артефакте — 'tg'.

        Без сопоставления таджикский документ тихо получал бы запасное среднее:
        преобразование выродилось бы в глобальное центрирование ровно для того
        языка, ради которого продукт и делается.
        """
        artifact = load_artifact(write_cell_artifact(self.path))
        vector = np.array(cell_document_vector_for("tg", 2))
        np.testing.assert_allclose(
            artifact.transform.apply(vector, group="tj"),
            artifact.transform.apply(vector, group="tg"),
        )

    def test_unknown_language_takes_the_fallback_instead_of_losing_the_topic(self):
        artifact = load_artifact(write_cell_artifact(self.path))
        vector = np.array([0.5, 0.1, 0.0, 0.2])
        expected = vector / np.linalg.norm(vector) - np.asarray(CELL_FALLBACK)
        np.testing.assert_allclose(
            artifact.transform.apply(vector, group="de"),
            expected / np.linalg.norm(expected),
        )

    def test_grouping_without_language_is_refused(self):
        """Поле есть, а взять его у документа неоткуда — значит, применять нечем."""
        write_cell_artifact(
            self.path,
            fields=("dataset_origin",),
            means={"real": [0.0] * 4, "synthetic": [1.0] * 4},
        )
        with self.assertRaises(TopicModelUnusable):
            load_artifact(self.path)

    def test_grouping_by_a_field_production_cannot_fix_is_refused(self):
        """«Возьмём что-нибудь похожее» здесь запрещено.

        Отрасль документа в бою не определяется ничем, и постоянного значения за
        ней не закреплено: любое среднее для такого артефакта было бы выбрано
        наугад, а неверная тема выглядит как верная.
        """
        self.assertNotIn("industry", PRODUCTION_FIXED_FIELDS)
        write_cell_artifact(
            self.path,
            fields=("language", "industry"),
            means={"ru|banking": [0.0] * 4, "ru|retail": [1.0] * 4},
        )
        with self.assertRaises(TopicModelUnusable) as caught:
            load_artifact(self.path)
        self.assertIn("industry", str(caught.exception))

    def test_artifact_without_a_single_production_cell_is_refused(self):
        """Модель, обученная на одной синтетике, боевому документу не отвечает."""
        write_cell_artifact(
            self.path,
            means={key: value for key, value in CELL_MEANS.items() if key.endswith("synthetic")},
        )
        with self.assertRaises(TopicModelUnusable):
            load_artifact(self.path)

    def test_keys_and_matrix_must_agree(self):
        """Разъехавшаяся пара «ключи / матрица» раздавала бы чужие средние."""
        write_cell_artifact(
            self.path,
            keys=list(CELL_MEANS),
            matrix=[CELL_MEANS["en|real"], CELL_MEANS["ru|real"]],
        )
        with self.assertRaises(TopicModelUnusable):
            load_artifact(self.path)

    def test_key_that_does_not_split_into_the_declared_fields_is_refused(self):
        write_cell_artifact(self.path, means={"ru": [0.0] * 4, "en": [1.0] * 4})
        with self.assertRaises(TopicModelUnusable):
            load_artifact(self.path)

    def test_group_centering_without_any_means_is_refused(self):
        write_artifact(
            self.path,
            transform={"kind": TRANSFORM_GROUP_CENTERING, "fields": ["language"]},
        )
        with self.assertRaises(TopicModelUnusable):
            load_artifact(self.path)

    def test_means_may_come_inline(self):
        """Числа принимаются обоими путями — как и у остальных преобразований."""
        write_artifact(
            self.path,
            transform={
                "kind": TRANSFORM_GROUP_CENTERING,
                "fields": ["language", "dataset_origin"],
                "means": {key: list(value) for key, value in CELL_MEANS.items()},
                "renormalize": False,
            },
        )
        artifact = load_artifact(self.path)
        np.testing.assert_allclose(
            (artifact.transform.group_means or {})["ru"], CELL_MEANS["ru|real"]
        )


class RenormalizationTests(ArtifactTestCase):
    """Преобразование обязано повториться ЦЕЛИКОМ, включая обе нормировки.

    Обучение считало метрики на векторах, приведённых к единичной длине до
    вычитания среднего и после него. Слой назначения, пропустивший любую из
    двух, сравнивает с центроидами вектор другой длины — то есть другого
    направления — и отвечает номером кластера без единой ошибки в журнале.
    """

    def test_renormalize_declared_by_the_artifact_is_applied(self):
        artifact = load_artifact(write_cell_artifact(self.path))
        self.assertTrue(artifact.transform.renormalize)
        result = artifact.transform.apply(
            np.array(cell_document_vector_for("ru", 2)), group="ru"
        )
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0)

    def test_renormalize_switched_off_in_the_artifact_is_respected(self):
        """Читатель соблюдает объявленное, а не своё представление о нём."""
        artifact = load_artifact(write_cell_artifact(self.path, renormalize=False))
        self.assertFalse(artifact.transform.renormalize)
        result = artifact.transform.apply(
            np.array(cell_document_vector_for("ru", 2)), group="ru"
        )
        self.assertNotAlmostEqual(float(np.linalg.norm(result)), 1.0)

    def test_the_vector_is_brought_to_unit_length_before_the_mean_is_subtracted(self):
        """Краснота: без первой нормировки документ уезжает в ДРУГОЙ кластер.

        Средние групп посчитаны на нормированных эмбеддингах, а вектор документа
        в бою — среднее векторов его фрагментов, то есть короче единичного.
        Вычесть среднее единичных векторов из короткого — значит сдвинуть его
        дальше, чем сдвигало обучение. Здесь взят документ, у которого две темы
        близки: лишний сдвиг перевешивает разницу между ними.
        """
        write_cell_artifact(
            self.path,
            fields=("language", "dataset_origin"),
            means={"ru|real": [0.0, 0.4, 0.0, 0.0], "ru|synthetic": [0.0, 0.0, 0.0, 0.0]},
            fallback=[0.0, 0.0, 0.0, 0.0],
        )
        artifact = load_artifact(self.path)
        raw = np.array([0.0, 0.6, 0.25, 0.4664])  # длина 0.8, а не 1.0
        self.assertAlmostEqual(float(np.linalg.norm(raw)), 0.8, places=3)

        honest = artifact.assign(raw, group="ru")
        naive = dataclasses.replace(
            artifact, transform=dataclasses.replace(artifact.transform, unit_input=False)
        ).assign(raw, group="ru")
        self.assertEqual(honest, 1)
        self.assertEqual(naive, 2)
        self.assertNotEqual(
            honest,
            naive,
            "если бы ответы совпадали, тест ничего не доказывал бы: проверять "
            "надо там, где пропущенная нормировка ВИДНА",
        )

    def test_old_group_mean_shift_artifacts_keep_their_arithmetic(self):
        """Совместимость: у артефактов прошлого формата геометрия не менялась.

        Ни одной нормировки им не объявляли, и додумывать её за них нельзя —
        это было бы ровно то же самовольное изменение преобразования, против
        которого весь раздел и написан.
        """
        artifact = load_artifact(write_language_artifact(self.path))
        self.assertEqual(artifact.transform.kind, TRANSFORM_GROUP_MEAN_SHIFT)
        self.assertFalse(artifact.transform.unit_input)
        self.assertFalse(artifact.transform.renormalize)
        np.testing.assert_allclose(
            artifact.transform.apply(np.array(document_vector_for("ru", 2)), group="ru"),
            [0.0, 0.0, 1.0, 0.0],
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
