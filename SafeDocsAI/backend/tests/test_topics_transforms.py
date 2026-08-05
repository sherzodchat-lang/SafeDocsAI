"""Преобразования, подавляющие язык и жанр, и модель, которая их помнит.

Проверяется здесь не качество кластеризации — оно измеряется отчётом, — а то,
что ломается тихо и потому дороже всего:

  * утечка. Средние по группам обязаны считаться на train и применяться к test
    как есть. Средние, пересчитанные по test, дали бы метрики лучше настоящих,
    и заметить это по самим числам невозможно;
  * забытое преобразование. Центроиды модели, обученной на векторах с вычтенным
    средним языка, лежат в другом пространстве, чем сырой эмбеддинг нового
    документа. predict без того же вычитания вернёт метки — просто неверные;
  * перепутанные слои. У разделяющего варианта кластер принадлежит своему
    языку, и документ обязан обслуживаться только своими кластерами. Иначе
    русский документ уедет в кластер, обученный на таджикских текстах, ровно
    тем способом, ради борьбы с которым слои и разделяли;
  * сшивка меток. Кластер 0 русской модели и кластер 0 таджикской — разные
    кластеры. Сложенные под одним номером, они завысили бы ARI разделяющего
    варианта, и завысили бы молча.

Живая Ollama не нужна: векторы здесь задаются руками, чтобы верный ответ был
известен заранее.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.modules.topics.pipeline.dataset import Corpus, Document  # noqa: E402
from app.modules.topics.pipeline.model_io import (  # noqa: E402
    MODEL_FORMAT_VERSION,
    ClusterTopic,
    TopicModel,
)
from app.modules.topics.pipeline.transforms import (  # noqa: E402
    ClusterRouting,
    GroupCentering,
    assign,
    group_keys,
)
from app.modules.topics.pipeline import variants  # noqa: E402
from test_topics_no_sklearn_guard import banned_usages  # noqa: E402

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def document(doc_id, *, language="ru", topic_id="A01", origin="synthetic", split="train"):
    return Document(
        id=doc_id,
        text="текст",
        language=language,
        topic_id=topic_id,
        topic=f"тема {topic_id}",
        topic_ru=f"тема {topic_id} по-русски",
        subtopic_id=f"{topic_id}_S01",
        dataset_origin=origin,
        split=split,
        word_count=1,
    )


class GroupKeyTests(unittest.TestCase):
    def test_single_field_key_is_the_value_itself(self):
        docs = [document("a", language="ru"), document("b", language="tg")]
        self.assertEqual(group_keys(docs, ("language",)), ["ru", "tg"])

    def test_two_fields_make_one_key(self):
        docs = [document("a", language="ru", origin="real")]
        self.assertEqual(group_keys(docs, ("language", "dataset_origin")), ["ru|real"])

    def test_empty_field_list_is_refused(self):
        """Пустой набор полей означал бы одну группу на весь корпус — то есть
        глобальное центрирование под видом группового."""
        with self.assertRaises(ValueError):
            group_keys([document("a")], ())


class GroupCenteringTests(unittest.TestCase):
    def setUp(self):
        # Две группы, сдвинутые друг относительно друга по первой координате.
        # Сдвиг — это и есть «ось языка»: внутри группы тема одинаково задана
        # второй координатой, но группы разъехались, и без центрирования
        # ближайшим соседом документа будет свой по языку, а не по теме.
        self.docs = [
            document("ru1", language="ru", topic_id="A01"),
            document("ru2", language="ru", topic_id="A02"),
            document("tg1", language="tg", topic_id="A01"),
            document("tg2", language="tg", topic_id="A02"),
        ]
        self.X = np.array(
            [
                [10.0, 1.0],
                [10.0, -1.0],
                [-10.0, 1.0],
                [-10.0, -1.0],
            ]
        )

    def test_group_mean_becomes_zero(self):
        transform = GroupCentering.fit(self.docs, self.X, ("language",))
        centered = transform.apply(self.docs, self.X)
        for key, rows in (("ru", [0, 1]), ("tg", [2, 3])):
            with self.subTest(group=key):
                np.testing.assert_allclose(centered[rows].mean(axis=0), [0.0, 0.0], atol=1e-12)

    def test_groups_become_comparable(self):
        """Смысл преобразования: после него документы одной темы из разных
        групп совпадают, а до него они на разных концах пространства."""
        transform = GroupCentering.fit(self.docs, self.X, ("language",))
        centered = transform.apply(self.docs, self.X)
        np.testing.assert_allclose(centered[0], centered[2], atol=1e-12)
        np.testing.assert_allclose(centered[1], centered[3], atol=1e-12)

    def test_rows_are_renormalised(self):
        transform = GroupCentering.fit(self.docs, self.X, ("language",))
        centered = transform.apply(self.docs, self.X)
        np.testing.assert_allclose(np.linalg.norm(centered, axis=1), np.ones(4), atol=1e-12)

    def test_means_are_remembered_and_not_recomputed(self):
        """Главная проверка на утечку: средние, посчитанные на train,
        применяются к новым документам как есть.

        Тестовая группа здесь смещена относительно обучающей; если бы среднее
        пересчитывалось по ней самой, её центр оказался бы в нуле, и результат
        не отличался бы от обучающего. Он обязан отличаться."""
        transform = GroupCentering.fit(self.docs, self.X, ("language",))
        held_out = [document("ru3", language="ru"), document("ru4", language="ru")]
        Y = np.array([[14.0, 1.0], [14.0, -1.0]])
        centered = transform.apply(held_out, Y)
        # Среднее train по ru = (10, 0); после вычитания первая координата 4, а
        # не 0, как было бы при пересчёте среднего по этим двум документам.
        self.assertGreater(centered[0][0], 0.9)
        self.assertGreater(centered[1][0], 0.9)

    def test_counts_record_the_training_size_of_each_group(self):
        transform = GroupCentering.fit(self.docs, self.X, ("language",))
        self.assertEqual(transform.counts, {"ru": 2, "tg": 2})

    def test_unknown_group_falls_back_to_the_global_mean(self):
        """Документ на языке, которого при обучении не было, не должен
        оставаться необработанным: преобразование вырождается в глобальное
        центрирование, и это записано в fallback_mean."""
        transform = GroupCentering.fit(self.docs, self.X, ("language",))
        stranger = [document("de1", language="de")]
        centered = transform.apply(stranger, np.array([[1.0, 3.0]]))
        expected = np.array([1.0, 3.0]) - self.X.mean(axis=0)
        np.testing.assert_allclose(centered[0], expected / np.linalg.norm(expected), atol=1e-12)
        self.assertEqual(transform.unknown_keys(["de", "ru"]), ["de"])

    def test_mismatched_rows_and_documents_are_refused(self):
        with self.assertRaises(ValueError):
            GroupCentering.fit(self.docs, self.X[:3], ("language",))

    def test_wrong_dimension_is_refused(self):
        transform = GroupCentering.fit(self.docs, self.X, ("language",))
        with self.assertRaises(ValueError):
            transform.apply(self.docs, np.zeros((4, 5)))

    def test_two_field_grouping_keeps_cells_apart(self):
        docs = [
            document("a", language="ru", origin="real"),
            document("b", language="ru", origin="synthetic"),
        ]
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        transform = GroupCentering.fit(docs, X, ("language", "dataset_origin"))
        self.assertEqual(set(transform.keys), {"ru|real", "ru|synthetic"})
        # В каждой ячейке по одному документу, вычитание собственного среднего
        # обнуляет вектор; нулевая строка остаётся нулевой, а не делится на ноль.
        np.testing.assert_allclose(transform.apply(docs, X), np.zeros((2, 2)), atol=1e-12)


class ClusterRoutingTests(unittest.TestCase):
    def setUp(self):
        self.routing = ClusterRouting(fields=("language",), cluster_groups=("ru", "ru", "tg"))

    def test_mask_allows_only_own_clusters(self):
        mask = self.routing.mask(["ru", "tg"])
        np.testing.assert_array_equal(mask, [[True, True, False], [False, False, True]])

    def test_unknown_group_is_refused(self):
        """Запасного варианта нет намеренно: модели для этой группы не
        существует, а ближайший чужой центроид был бы ответом наугад."""
        with self.assertRaises(ValueError):
            self.routing.mask(["de"])

    def test_assignment_respects_the_mask(self):
        """Документ лежит вплотную к чужому центроиду и всё равно обязан уйти
        к своему — иначе разделение слоёв ничего не даёт."""
        centroids = np.array([[1.0, 0.0], [0.0, 1.0], [0.99, 0.01]])
        X = np.array([[1.0, 0.0]])
        without = assign(X, centroids, allowed=None)
        with_mask = assign(X, centroids, allowed=self.routing.mask(["tg"]))
        self.assertEqual(int(without[0]), 0)
        self.assertEqual(int(with_mask[0]), 2)


class ModelWithTransformTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "model.npz"
        self.addCleanup(self.directory.cleanup)
        self.docs = [
            document("ru1", language="ru"),
            document("ru2", language="ru"),
            document("tg1", language="tg"),
            document("tg2", language="tg"),
        ]
        self.X = np.array([[10.0, 1.0], [10.0, -1.0], [-10.0, 1.0], [-10.0, -1.0]])
        self.transform = GroupCentering.fit(self.docs, self.X, ("language",))

    def make_model(self, *, transform=None, routing=None, centroids=None):
        n = 2 if centroids is None else len(centroids)
        return TopicModel(
            centroids=np.array([[0.0, 1.0], [0.0, -1.0]]) if centroids is None else np.asarray(centroids),
            embedding_model="qwen3-embedding:8b",
            normalize=True,
            cluster_topics=tuple(
                ClusterTopic(i, f"A0{i + 1}", f"тема {i}", 0.8, 10) for i in range(n)
            ),
            params={"variant": "centered_language"},
            transform=transform,
            routing=routing,
        )

    def test_roundtrip_keeps_the_transform(self):
        self.make_model(transform=self.transform).save(self.path)
        loaded = TopicModel.load(self.path)
        self.assertIsNotNone(loaded.transform)
        self.assertEqual(loaded.transform.fields, ("language",))
        self.assertEqual(loaded.transform.keys, ("ru", "tg"))
        np.testing.assert_allclose(loaded.transform.means, self.transform.means)
        np.testing.assert_allclose(loaded.transform.fallback_mean, self.transform.fallback_mean)
        self.assertEqual(loaded.transform.counts, {"ru": 2, "tg": 2})
        self.assertEqual(loaded.required_fields, ("language",))

    def test_saved_model_predicts_exactly_as_before_saving(self):
        model = self.make_model(transform=self.transform)
        model.save(self.path)
        np.testing.assert_array_equal(
            TopicModel.load(self.path).predict(self.X, documents=self.docs),
            model.predict(self.X, documents=self.docs),
        )

    def test_transform_actually_changes_the_answer(self):
        """Обратная проверка к предыдущей: если бы сохранённое преобразование
        не применялось, метки всё равно посчитались бы — просто другие.

        Данные здесь такие, каким оказался настоящий корпус: тема смещена
        языком. У русских документов вторая координата положительна у обоих, у
        таджикских отрицательна, и без вычитания среднего языка метки
        повторяют язык, а не тему."""
        X = np.array([[10.0, 1.0], [10.0, 3.0], [-10.0, -1.0], [-10.0, -3.0]])
        transform = GroupCentering.fit(self.docs, X, ("language",))
        with_transform = self.make_model(transform=transform).predict(X, documents=self.docs)
        without = self.make_model().predict(X)
        self.assertEqual([int(value) for value in without], [0, 0, 1, 1])
        self.assertEqual([int(value) for value in with_transform], [1, 0, 0, 1])

    def test_model_with_transform_refuses_bare_vectors(self):
        model = self.make_model(transform=self.transform)
        with self.assertRaises(ValueError) as caught:
            model.predict(self.X)
        self.assertIn("language", str(caught.exception))

    def test_groups_may_be_passed_instead_of_documents(self):
        """В приложении разметки нет: язык определяется на лету, и модель
        обязана принимать готовый ключ группы."""
        model = self.make_model(transform=self.transform)
        np.testing.assert_array_equal(
            model.predict(self.X, groups=["ru", "ru", "tg", "tg"]),
            model.predict(self.X, documents=self.docs),
        )

    def test_group_count_must_match_the_matrix(self):
        model = self.make_model(transform=self.transform)
        with self.assertRaises(ValueError):
            model.predict(self.X, groups=["ru"])

    def test_routing_survives_saving(self):
        routing = ClusterRouting(fields=("language",), cluster_groups=("ru", "tg"))
        model = self.make_model(routing=routing)
        model.save(self.path)
        loaded = TopicModel.load(self.path)
        self.assertEqual(loaded.routing.cluster_groups, ("ru", "tg"))
        self.assertEqual(loaded.required_fields, ("language",))
        np.testing.assert_array_equal(
            loaded.predict(self.X, documents=self.docs),
            [0, 0, 1, 1],
        )

    def test_transform_and_routing_together(self):
        routing = ClusterRouting(
            fields=("language",), cluster_groups=("ru", "ru", "tg", "tg")
        )
        centroids = np.array([[0.0, 1.0], [0.0, -1.0], [0.0, 1.0], [0.0, -1.0]])
        model = self.make_model(transform=self.transform, routing=routing, centroids=centroids)
        model.save(self.path)
        loaded = TopicModel.load(self.path)
        labels = loaded.predict(self.X, documents=self.docs)
        # Русские документы обслуживаются кластерами 0-1, таджикские — 2-3.
        self.assertEqual(list(labels), [0, 1, 2, 3])
        self.assertEqual(loaded.required_fields, ("language",))

    def test_transform_and_routing_must_agree_on_the_fields(self):
        """Ключ группы собирается из полей одним порядком: два разных набора
        дали бы два разных ключа для одного документа."""
        with self.assertRaises(ValueError):
            self.make_model(
                transform=self.transform,
                routing=ClusterRouting(
                    fields=("dataset_origin",), cluster_groups=("synthetic", "real")
                ),
            )

    def test_missing_means_in_the_file_are_refused(self):
        """Метаданные обещают преобразование, а матриц нет: сделать вид, что
        преобразования не было, значит выдать метки из другого пространства."""
        self.make_model(transform=self.transform).save(self.path)
        with np.load(self.path, allow_pickle=False) as archive:
            meta = str(archive["meta"])
            centroids = archive["centroids"]
        np.savez_compressed(self.path, centroids=centroids, meta=np.array(meta))
        with self.assertRaises(ValueError):
            TopicModel.load(self.path)

    def test_routing_length_must_match_the_centroids(self):
        routing = ClusterRouting(fields=("language",), cluster_groups=("ru", "tg", "tg"))
        self.make_model(routing=routing).save(self.path)
        with self.assertRaises(ValueError):
            TopicModel.load(self.path)

    def test_old_format_still_loads_as_a_model_without_transform(self):
        """Файл версии 1 читается: преобразования в нём не могло быть, и
        отсутствие — не потеря, а факт."""
        self.make_model().save(self.path)
        with np.load(self.path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["meta"]))
            centroids = archive["centroids"]
        meta["version"] = 1
        meta.pop("transform")
        meta.pop("routing")
        np.savez_compressed(self.path, centroids=centroids, meta=np.array(json.dumps(meta)))
        loaded = TopicModel.load(self.path)
        self.assertIsNone(loaded.transform)
        self.assertEqual(loaded.required_fields, ())

    def test_future_format_is_still_refused(self):
        self.make_model().save(self.path)
        with np.load(self.path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["meta"]))
            centroids = archive["centroids"]
        meta["version"] = MODEL_FORMAT_VERSION + 1
        np.savez_compressed(self.path, centroids=centroids, meta=np.array(json.dumps(meta)))
        with self.assertRaises(ValueError):
            TopicModel.load(self.path)


def toy_splits():
    """Игрушечный корпус: две темы, два языка, обе оси видны глазом.

    Первая координата — язык (сдвиг), вторая — тема. Без вмешательства K-means
    разделит по языку: сдвиг там в десять раз больше. Ровно то, что происходит
    на настоящем корпусе, только в двух измерениях.
    """
    splits = {}
    data = {}
    for split in ("train", "validation", "test"):
        documents = []
        rows = []
        for index in range(8):
            language = "ru" if index % 2 == 0 else "tg"
            topic = "A01" if index < 4 else "A02"
            documents.append(
                document(f"{split}{index}", language=language, topic_id=topic, split=split)
            )
            rows.append(
                [10.0 if language == "ru" else -10.0, 1.0 if topic == "A01" else -1.0]
            )
        splits[split] = Corpus(tuple(documents))
        data[split] = np.array(rows)
    return splits, data


class VariantWiringTests(unittest.TestCase):
    def setUp(self):
        self.splits, self.data = toy_splits()

    def spec(self, name):
        return next(item for item in variants.VARIANTS if item.name == name)

    def test_six_variants_are_declared(self):
        names = [item.name for item in variants.VARIANTS]
        self.assertEqual(
            names,
            [
                "baseline",
                "per_language",
                "centered_language",
                "centered_origin",
                "centered_cell",
                "per_cell",
            ],
        )

    def test_baseline_leaves_vectors_untouched(self):
        transform, tdata = variants.fit_transform(self.spec("baseline"), self.splits, self.data)
        self.assertIsNone(transform)
        np.testing.assert_array_equal(tdata["test"], self.data["test"])

    def test_centering_is_fitted_on_train_only(self):
        """Средние обязаны прийти из train. Здесь test сдвинут, и если бы
        центрирование считалось по нему, сдвиг исчез бы."""
        splits, data = toy_splits()
        data["test"] = data["test"] + np.array([5.0, 0.0])
        transform, tdata = variants.fit_transform(
            self.spec("centered_language"), splits, data
        )
        self.assertEqual(set(transform.keys), {"ru", "tg"})
        # Строки test после вычитания train-средних сохраняют общий сдвиг +5.
        self.assertTrue(np.all(tdata["test"][:, 0] > 0))

    def test_split_variant_labels_do_not_collide_between_strata(self):
        """Кластер 0 русской модели и кластер 0 таджикской — разные кластеры.
        Если бы номера совпали, ARI разделяющего варианта был бы завышен."""
        spec = self.spec("per_language")
        _, tdata = variants.fit_transform(spec, self.splits, self.data)
        fit = variants.fit_variant(spec, self.splits, tdata, {"ru": 2, "tg": 2})
        self.assertEqual(fit.centroids.shape[0], 4)
        self.assertEqual(fit.cluster_groups, ("ru", "ru", "tg", "tg"))
        for name in ("train", "validation", "test"):
            labels = fit.labels[name]
            languages = np.asarray(self.splits[name].labels("language"))
            self.assertEqual(set(labels[languages == "ru"].tolist()) & {2, 3}, set())
            self.assertEqual(set(labels[languages == "tg"].tolist()) & {0, 1}, set())

    def test_splitting_by_language_finds_the_topic_that_global_kmeans_misses(self):
        """Смысл всей затеи на игрушечном примере: общий прогон уходит на язык,
        разделяющий — находит тему."""
        from app.modules.topics.pipeline.experiment import external_scores

        base_spec = self.spec("baseline")
        _, base_data = variants.fit_transform(base_spec, self.splits, self.data)
        base = variants.fit_variant(base_spec, self.splits, base_data, {"": 2})
        base_scores = external_scores(base.labels["test"], self.splits["test"])
        self.assertAlmostEqual(base_scores["language"]["ari"], 1.0, places=6)
        self.assertLess(base_scores["topic_id"]["ari"], 0.5)

        spec = self.spec("centered_language")
        _, tdata = variants.fit_transform(spec, self.splits, self.data)
        fit = variants.fit_variant(spec, self.splits, tdata, {"": 2})
        scores = external_scores(fit.labels["test"], self.splits["test"])
        self.assertAlmostEqual(scores["topic_id"]["ari"], 1.0, places=6)

    def test_stratum_missing_from_train_is_refused(self):
        """Документ, чья группа не встречалась при обучении, остался бы без
        метки, а метрики посчитали бы её отдельным кластером."""
        splits, data = toy_splits()
        splits["test"] = Corpus(
            splits["test"].documents + (document("stranger", language="de", split="test"),)
        )
        data["test"] = np.vstack([data["test"], [[0.0, 1.0]]])
        spec = self.spec("per_language")
        _, tdata = variants.fit_transform(spec, splits, data)
        with self.assertRaises(ValueError):
            variants.fit_variant(spec, splits, tdata, {"ru": 2, "tg": 2})

    def test_k_grid_is_capped_by_the_layer_size(self):
        """В маленьком слое k=40 означало бы кластеры по три документа, у
        которых силуэт меряет уже не структуру, а шум."""
        self.assertLessEqual(max(variants.k_grid_for(140, 12)), 35)
        self.assertLessEqual(max(variants.k_grid_for(40, 8)), 10)
        self.assertEqual(variants.k_grid_for(4, 2), (2,))

    def test_matched_regime_equalises_the_total_number_of_clusters(self):
        search = variants.search_k(self.spec("baseline"), self.splits, self.data)
        regimes = variants.k_regimes(
            self.spec("baseline"), self.splits, search, matched_clusters=6
        )
        self.assertEqual(regimes["true_k"], {"": 2})
        self.assertEqual(regimes["matched"], {"": 6})

    def test_split_variant_needs_no_matched_regime(self):
        """Союз слоёв уже даёт нужное общее число кластеров: отдельный режим
        добавил бы строку, повторяющую true_k."""
        spec = self.spec("per_language")
        _, tdata = variants.fit_transform(spec, self.splits, self.data)
        search = variants.search_k(spec, self.splits, tdata)
        regimes = variants.k_regimes(spec, self.splits, search, matched_clusters=4)
        self.assertNotIn("matched", regimes)

    def test_neighbour_agreement_can_be_restricted_to_a_group(self):
        """Общее согласие соседей меряет прежде всего язык: ближайшим соседом
        почти всегда оказывается документ того же языка. Внутри группы вопрос
        меняется на тот, ради которого работа делается."""
        corpus = self.splits["train"]
        X = self.data["train"]
        globally = variants.neighbour_agreement_scoped(corpus, X, ())
        inside = variants.neighbour_agreement_scoped(corpus, X, ("language",))
        self.assertAlmostEqual(
            globally["same_label_as_nearest_neighbour"]["language"], 1.0, places=6
        )
        self.assertAlmostEqual(
            inside["same_label_as_nearest_neighbour"]["topic_id"], 1.0, places=6
        )
        self.assertEqual(inside["scope"], ["language"])
        self.assertEqual(inside["n_compared"], len(corpus))

    def test_chance_level_is_computed_in_the_same_scope(self):
        """Случайный уровень внутри группы считается по составу группы: доля
        совпадений по ячейке против общекорпусной случайности была бы подменой."""
        corpus = self.splits["train"]
        inside = variants.neighbour_agreement_scoped(corpus, self.data["train"], ("language",))
        # Внутри каждого языка по две темы поровну -> случайный уровень 0.5,
        # тогда как язык внутри своей группы совпадает всегда.
        self.assertAlmostEqual(inside["chance_level"]["topic_id"], 0.5, places=6)
        self.assertAlmostEqual(inside["chance_level"]["language"], 1.0, places=6)

    def test_model_built_from_a_split_variant_carries_its_routing(self):
        spec = self.spec("per_language")
        transform, tdata = variants.fit_transform(spec, self.splits, self.data)
        fit = variants.fit_variant(spec, self.splits, tdata, {"ru": 2, "tg": 2})
        model = variants.build_model(fit, transform, "qwen3-embedding:8b", {"variant": spec.name})
        self.assertEqual(model.required_fields, ("language",))
        np.testing.assert_array_equal(
            model.predict(self.data["test"], documents=self.splits["test"].documents),
            fit.labels["test"],
        )

    def test_model_built_from_a_centering_variant_carries_its_means(self):
        spec = self.spec("centered_cell")
        transform, tdata = variants.fit_transform(spec, self.splits, self.data)
        fit = variants.fit_variant(spec, self.splits, tdata, {"": 2})
        model = variants.build_model(fit, transform, "qwen3-embedding:8b", {"variant": spec.name})
        self.assertEqual(sorted(model.required_fields), ["dataset_origin", "language"])
        np.testing.assert_array_equal(
            model.predict(self.data["test"], documents=self.splits["test"].documents),
            fit.labels["test"],
        )


class VariantsAreFreeOfLibraryClusteringTests(unittest.TestCase):
    """Запрет sklearn действует и на новые файлы этой работы."""

    def test_no_banned_imports(self):
        paths = [
            os.path.join(BACKEND_ROOT, "app", "modules", "topics", "pipeline", "transforms.py"),
            os.path.join(BACKEND_ROOT, "app", "modules", "topics", "pipeline", "variants.py"),
            os.path.join(BACKEND_ROOT, "cluster_topics_variants.py"),
        ]
        for path in paths:
            with self.subTest(file=os.path.basename(path)):
                self.assertTrue(os.path.exists(path), path)
                with open(path, encoding="utf-8") as handle:
                    self.assertEqual(banned_usages(handle.read()), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
