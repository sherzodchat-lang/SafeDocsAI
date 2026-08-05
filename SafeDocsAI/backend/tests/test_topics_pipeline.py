"""Мост «корпус -> эмбеддинги -> K-means -> сохранённая модель».

Сам алгоритм проверяется отдельно (test_topics_kmeans, test_topics_metrics), и
здесь его качество не измеряется. Здесь проверяется то, что ломается тихо:

  * порядок. Матрица векторов и колонки разметки выровнены по документам, и
    сдвиг на одну строку не вызовет ни одной ошибки — метрики посчитаются и
    ответят про случайное разбиение;
  * протухание кэша. Векторы от другой модели лежат в другом пространстве;
    подставить их вместо нужных — это не «немного другие числа», а другая
    работа с прежними выводами в отчёте;
  * полнота сохранённой модели. Центроидов без имени модели эмбеддингов и без
    признака нормировки достаточно, чтобы predict вернул метки, — просто
    неверные.

Живая Ollama здесь не нужна: embed_fn передаётся снаружи, и все прогоны идут
на подставной функции с заранее известными векторами.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.modules.topics.kmeans import KMeans  # noqa: E402
from app.modules.topics.pipeline.dataset import (  # noqa: E402
    Corpus,
    Document,
    load_full,
    load_jsonl,
    load_splits,
)
from app.modules.topics.pipeline.embeddings import (  # noqa: E402
    CACHE_FORMAT_VERSION,
    EmbeddingCache,
    embed_corpus,
)
from app.modules.topics.pipeline.experiment import (  # noqa: E402
    external_scores,
    fit_on,
    neighbour_agreement,
    run_cells,
    run_layers,
)
from app.modules.topics.pipeline.model_io import (  # noqa: E402
    MODEL_FORMAT_VERSION,
    ClusterTopic,
    TopicModel,
    dominant_topics,
)
from test_topics_no_sklearn_guard import banned_usages  # noqa: E402

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.join(BACKEND_ROOT, "app", "modules", "topics", "pipeline")


def record(
    doc_id,
    *,
    text="текст документа",
    language="ru",
    topic_id="A01",
    subtopic_id="A01_S01",
    origin="synthetic",
    split="train",
):
    return {
        "id": doc_id,
        "text": text,
        "language": language,
        "topic_id": topic_id,
        "topic": f"тема {topic_id}",
        "subtopic_id": subtopic_id,
        "dataset_origin": origin,
        "is_synthetic": origin == "synthetic",
        "split": split,
        "word_count": len(text.split()),
    }


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def document(doc_id, **kwargs):
    data = record(doc_id, **kwargs)
    data.pop("is_synthetic")
    return Document(**data)


class DatasetLoadingTests(unittest.TestCase):
    def test_reads_all_labelling_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jsonl"
            write_jsonl(path, [record("a1", language="tg", origin="real", topic_id="B02")])
            corpus = load_jsonl(path)

        self.assertEqual(len(corpus), 1)
        self.assertEqual(corpus.labels("language"), ["tg"])
        self.assertEqual(corpus.labels("topic_id"), ["B02"])
        self.assertEqual(corpus.labels("dataset_origin"), ["real"])
        self.assertFalse(corpus.documents[0].is_synthetic)

    def test_blank_lines_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jsonl"
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(record("a1")) + "\n\n")
                handle.write(json.dumps(record("a2")) + "\n")
            self.assertEqual(len(load_jsonl(path)), 2)

    def test_missing_topic_id_is_refused(self):
        """Документ без темы стал бы отдельной «темой» из пустых строк."""
        broken = record("a1")
        broken["topic_id"] = ""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jsonl"
            write_jsonl(path, [broken])
            with self.assertRaises(ValueError):
                load_jsonl(path)

    def test_broken_json_names_the_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jsonl"
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(record("a1")) + "\n")
                handle.write("{не json\n")
            with self.assertRaises(ValueError) as context:
                load_jsonl(path)
            self.assertIn(":2:", str(context.exception))


class SplitLoadingTests(unittest.TestCase):
    def make_dir(self, directory, train, validation, test):
        write_jsonl(Path(directory) / "full_train.jsonl", train)
        write_jsonl(Path(directory) / "full_validation.jsonl", validation)
        write_jsonl(Path(directory) / "full_test.jsonl", test)
        write_jsonl(
            Path(directory) / "full_multilingual.jsonl", train + validation + test
        )

    def test_loads_three_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_dir(
                directory,
                [record("t1"), record("t2")],
                [record("v1", split="validation")],
                [record("s1", split="test")],
            )
            splits = load_splits(directory)
            self.assertEqual({name: len(c) for name, c in splits.items()},
                             {"train": 2, "validation": 1, "test": 1})
            self.assertEqual(len(load_full(directory)), 4)

    def test_document_leaking_between_splits_is_refused(self):
        """Утечка train в test — самая дорогая ошибка: метрики просто станут
        хорошими, и по ним этого не увидеть."""
        with tempfile.TemporaryDirectory() as directory:
            leaked = record("t1", split="test")
            self.make_dir(directory, [record("t1")], [], [leaked])
            with self.assertRaises(ValueError) as context:
                load_splits(directory)
            self.assertIn("t1", str(context.exception))

    def test_split_field_must_match_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_dir(directory, [record("t1", split="test")], [], [])
            with self.assertRaises(ValueError):
                load_splits(directory)


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.corpus = Corpus(
            (
                document("a1", origin="synthetic", topic_id="A01"),
                document("b1", origin="real", topic_id="B01"),
                document("b2", origin="real", topic_id="B02"),
            )
        )

    def test_subsets_keep_order(self):
        self.assertEqual(self.corpus.real().ids(), ["b1", "b2"])
        self.assertEqual(self.corpus.synthetic().ids(), ["a1"])

    def test_counts(self):
        self.assertEqual(self.corpus.counts("dataset_origin"), {"real": 2, "synthetic": 1})


class EmbeddingCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "embeddings.npz"
        self.addCleanup(self.directory.cleanup)

    def test_roundtrip(self):
        cache = EmbeddingCache.empty("model-a")
        cache.put("a1", [1.0, 2.0, 3.0])
        cache.put("a2", [4.0, 5.0, 6.0])
        cache.save(self.path)

        loaded = EmbeddingCache.load(self.path, "model-a")
        self.assertEqual(len(loaded), 2)
        np.testing.assert_allclose(loaded.get("a2"), [4.0, 5.0, 6.0])
        self.assertEqual(loaded.dim, 3)

    def test_missing_file_gives_empty_cache(self):
        self.assertEqual(len(EmbeddingCache.load(self.path, "model-a")), 0)

    def test_changing_the_model_invalidates_the_cache(self):
        """Векторы другой модели — другое пространство. Не «менее точные
        числа», а чужие."""
        cache = EmbeddingCache.empty("model-a")
        cache.put("a1", [1.0, 2.0])
        cache.save(self.path)

        stale = EmbeddingCache.load(self.path, "model-b")
        self.assertEqual(len(stale), 0)
        self.assertIsNone(stale.get("a1"))
        # А своя модель тот же файл читает.
        self.assertEqual(len(EmbeddingCache.load(self.path, "model-a")), 1)

    def test_unknown_format_version_invalidates_the_cache(self):
        cache = EmbeddingCache.empty("model-a")
        cache.put("a1", [1.0, 2.0])
        cache.save(self.path)
        with np.load(self.path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["meta"]))
            ids, vectors = archive["ids"], archive["vectors"]
        meta["version"] = CACHE_FORMAT_VERSION + 1
        np.savez_compressed(self.path, ids=ids, vectors=vectors, meta=np.array(json.dumps(meta)))

        self.assertEqual(len(EmbeddingCache.load(self.path, "model-a")), 0)

    def test_dimension_mismatch_is_refused(self):
        cache = EmbeddingCache.empty("model-a")
        cache.put("a1", [1.0, 2.0])
        with self.assertRaises(ValueError):
            cache.put("a2", [1.0, 2.0, 3.0])

    def test_nan_vector_is_refused(self):
        cache = EmbeddingCache.empty("model-a")
        with self.assertRaises(ValueError):
            cache.put("a1", [1.0, float("nan")])

    def test_matrix_follows_the_requested_order(self):
        """Порядок строк задаётся снаружи, потому что рядом идут колонки
        разметки: сдвиг здесь не уронил бы ничего, а метрики стали бы про
        случайное разбиение."""
        cache = EmbeddingCache.empty("model-a")
        cache.put("a1", [1.0, 0.0])
        cache.put("a2", [0.0, 1.0])
        np.testing.assert_allclose(cache.matrix(["a2", "a1"]), [[0.0, 1.0], [1.0, 0.0]])

    def test_matrix_refuses_unknown_ids(self):
        cache = EmbeddingCache.empty("model-a")
        cache.put("a1", [1.0, 0.0])
        with self.assertRaises(KeyError):
            cache.matrix(["a1", "нет такого"])

    def test_saving_empty_cache_is_readable(self):
        EmbeddingCache.empty("model-a").save(self.path)
        self.assertEqual(len(EmbeddingCache.load(self.path, "model-a")), 0)


class FakeEmbedder:
    """Подставная модель: вектор выводится из текста, вызовы считаются."""

    def __init__(self, dim=4):
        self.dim = dim
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return [
            [float(len(text)), float(text.count("а")), 1.0, 0.0][: self.dim]
            for text in texts
        ]


class EmbedCorpusTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "embeddings.npz"
        self.addCleanup(self.directory.cleanup)
        self.documents = [document(f"d{i}", text=f"текст {i}" * (i + 1)) for i in range(7)]

    def test_computes_everything_once_and_then_reads_from_cache(self):
        embedder = FakeEmbedder()
        embed_corpus(
            self.documents,
            model="model-a",
            cache_path=self.path,
            embed_fn=embedder,
            batch_size=3,
            progress=None,
        )
        self.assertEqual(sum(len(call) for call in embedder.calls), 7)
        self.assertEqual([len(call) for call in embedder.calls], [3, 3, 1])

        again = FakeEmbedder()
        cache = embed_corpus(
            self.documents,
            model="model-a",
            cache_path=self.path,
            embed_fn=again,
            batch_size=3,
            progress=None,
        )
        self.assertEqual(again.calls, [], "второй прогон не должен ходить в модель")
        self.assertEqual(len(cache), 7)

    def test_only_missing_documents_are_computed(self):
        embedder = FakeEmbedder()
        embed_corpus(
            self.documents[:4],
            model="model-a",
            cache_path=self.path,
            embed_fn=embedder,
            batch_size=10,
            progress=None,
        )
        second = FakeEmbedder()
        embed_corpus(
            self.documents,
            model="model-a",
            cache_path=self.path,
            embed_fn=second,
            batch_size=10,
            progress=None,
        )
        self.assertEqual([len(call) for call in second.calls], [3])

    def test_changed_model_recomputes_everything(self):
        embed_corpus(
            self.documents,
            model="model-a",
            cache_path=self.path,
            embed_fn=FakeEmbedder(),
            batch_size=10,
            progress=None,
        )
        other = FakeEmbedder()
        embed_corpus(
            self.documents,
            model="model-b",
            cache_path=self.path,
            embed_fn=other,
            batch_size=10,
            progress=None,
        )
        self.assertEqual(sum(len(call) for call in other.calls), 7)

    def test_short_answer_from_the_model_is_refused(self):
        """Молчаливое расхождение длин сдвинуло бы соответствие «документ ->
        вектор» на всю оставшуюся пачку."""

        def truncating(texts):
            return [[1.0, 2.0] for _ in texts][:-1]

        with self.assertRaises(ValueError):
            embed_corpus(
                self.documents,
                model="model-a",
                cache_path=self.path,
                embed_fn=truncating,
                batch_size=10,
                progress=None,
            )

    def test_time_spent_is_remembered_in_the_file(self):
        """«Сколько заняло эмбеддирование» спрашивают на десятом запуске,
        когда всё уже читается из кэша."""
        embed_corpus(
            self.documents,
            model="model-a",
            cache_path=self.path,
            embed_fn=FakeEmbedder(),
            batch_size=3,
            progress=None,
        )
        cache = EmbeddingCache.load(self.path, "model-a")
        self.assertEqual(cache.stats["computed"], 7)
        self.assertGreaterEqual(cache.stats["seconds"], 0.0)

    def test_progress_is_reported(self):
        lines = []
        embed_corpus(
            self.documents,
            model="model-a",
            cache_path=self.path,
            embed_fn=FakeEmbedder(),
            batch_size=3,
            progress=lines.append,
        )
        self.assertTrue(any("7" in line for line in lines))
        self.assertTrue(any("осталось" in line for line in lines))


class ModelIOTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "topic_model.npz"
        self.addCleanup(self.directory.cleanup)

    def make_model(self):
        return TopicModel(
            centroids=np.array([[1.0, 0.0], [0.0, 1.0]]),
            embedding_model="qwen3-embedding:8b",
            normalize=True,
            cluster_topics=(
                ClusterTopic(0, "A01", "Кадровые документы", 0.9, 100),
                ClusterTopic(1, "B03", "История", 0.5, 40),
            ),
            params={"k": 2, "random_state": 42},
        )

    def test_roundtrip_keeps_everything_needed_to_apply_it(self):
        self.make_model().save(self.path)
        loaded = TopicModel.load(self.path)

        np.testing.assert_allclose(loaded.centroids, [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(loaded.embedding_model, "qwen3-embedding:8b")
        self.assertTrue(loaded.normalize)
        self.assertEqual(loaded.n_clusters, 2)
        self.assertEqual(loaded.topic_of(1).topic_id, "B03")
        self.assertAlmostEqual(loaded.topic_of(1).share, 0.5)
        self.assertEqual(loaded.params["random_state"], 42)

    def test_loaded_model_assigns_new_documents(self):
        self.make_model().save(self.path)
        loaded = TopicModel.load(self.path)
        labels = loaded.predict(np.array([[5.0, 0.1], [0.1, 5.0]]))
        self.assertEqual(list(labels), [0, 1])

    def test_loaded_model_matches_the_one_that_was_trained(self):
        """Сохранение не должно менять ответы: иначе метрики в отчёте
        относились бы к одной модели, а приложение работало бы другой."""
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(loc, 0.2, (20, 5)) for loc in (-3.0, 3.0)])
        trained = KMeans(n_clusters=2, random_state=1).fit(X)
        model = TopicModel(
            centroids=trained.centroids_,
            embedding_model="model-a",
            normalize=True,
            cluster_topics=(),
        )
        model.save(self.path)
        np.testing.assert_array_equal(
            TopicModel.load(self.path).predict(X), trained.predict(X)
        )

    def test_unknown_format_version_is_refused(self):
        self.make_model().save(self.path)
        with np.load(self.path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["meta"]))
            centroids = archive["centroids"]
        meta["version"] = MODEL_FORMAT_VERSION + 1
        np.savez_compressed(self.path, centroids=centroids, meta=np.array(json.dumps(meta)))
        with self.assertRaises(ValueError):
            TopicModel.load(self.path)


class DominantTopicTests(unittest.TestCase):
    def test_majority_topic_and_its_share(self):
        labels = [0, 0, 0, 1, 1]
        topic_ids = ["A01", "A01", "B02", "B02", "B02"]
        names = ["HR", "HR", "История", "История", "История"]
        result = dominant_topics(labels, topic_ids, names, 2)
        self.assertEqual(result[0].topic_id, "A01")
        self.assertAlmostEqual(result[0].share, 2 / 3)
        self.assertEqual(result[1].topic_id, "B02")
        self.assertAlmostEqual(result[1].share, 1.0)

    def test_empty_cluster_still_gets_a_row(self):
        """predict может вернуть любой номер из 0..k-1, и приложение обязано
        уметь ответить на каждый."""
        result = dominant_topics([0, 0], ["A01", "A01"], ["HR", "HR"], 3)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[2].size, 0)
        self.assertEqual(result[2].topic_id, "")


class ExperimentWiringTests(unittest.TestCase):
    """Проверяется проводка, а не качество: на игрушечном корпусе с заведомо
    разделимыми группами метрики обязаны выйти на единицу, иначе где-то
    разъехались векторы и разметка."""

    def make_corpus_and_matrix(self):
        documents = []
        rows = []
        for index in range(12):
            topic = "A01" if index % 2 == 0 else "B01"
            documents.append(
                document(
                    f"d{index}",
                    topic_id=topic,
                    subtopic_id=f"{topic}_S01",
                    origin="synthetic" if topic == "A01" else "real",
                    language="ru" if index % 2 == 0 else "en",
                )
            )
            rows.append([1.0, 0.0] if topic == "A01" else [0.0, 1.0])
        return Corpus(tuple(documents)), np.array(rows)

    def test_perfect_split_scores_one_against_every_column(self):
        corpus, X = self.make_corpus_and_matrix()
        fitted = fit_on(corpus, X, 2)
        scores = external_scores(fitted.kmeans.labels_, corpus)
        for field in ("topic_id", "language", "dataset_origin"):
            self.assertAlmostEqual(scores[field]["ari"], 1.0, places=6, msg=field)
            self.assertAlmostEqual(scores[field]["purity"], 1.0, places=6, msg=field)
        self.assertEqual(scores["topic_id"]["n_classes"], 2)

    def test_shifted_labels_do_not_score_one(self):
        """Обратная проверка к предыдущей: если бы метрики считались по чужому
        порядку строк, тест выше остался бы зелёным."""
        corpus, X = self.make_corpus_and_matrix()
        fitted = fit_on(corpus, X, 2)
        # Меняются местами метки двух документов из РАЗНЫХ тем. Сдвиг всех
        # меток (np.roll) здесь не годится: на чередующемся корпусе он даёт
        # то же разбиение с переставленными номерами, а ARI сравнивает
        # разбиения, а не номера, — и тест остался бы зелёным при любой ошибке.
        broken = np.array(fitted.kmeans.labels_, copy=True)
        broken[0], broken[1] = broken[1], broken[0]
        self.assertLess(external_scores(broken, corpus)["topic_id"]["ari"], 1.0)

    def test_layers_are_evaluated_separately(self):
        corpus, X = self.make_corpus_and_matrix()
        splits = {"train": corpus, "validation": corpus, "test": corpus}
        data = {"train": X, "validation": X, "test": X}
        report = run_layers(splits, data)
        self.assertEqual(set(report), {"synthetic", "real", "mixed"})
        self.assertEqual(report["synthetic"]["n_documents"]["train"], 6)
        self.assertEqual(report["real"]["n_documents"]["train"], 6)
        self.assertEqual(report["mixed"]["k"], 2)


class LanguageControlledLayerTests(unittest.TestCase):
    """Сравнение слоёв при закреплённом языке.

    Корпус тут устроен так, чтобы поймать ровно ту ошибку, ради которой
    run_cells и написана: язык разделим ИДЕАЛЬНО, а тема внутри языка — нет.
    Если ячейки перестанут резаться по языку, кластеры уйдут на язык, ARI по
    теме подскочит, и тест это заметит.
    """

    def make(self):
        documents = []
        rows = []
        for origin in ("synthetic", "real"):
            for language, axis in (("en", 0), ("ru", 1), ("tg", 2)):
                for index in range(4):
                    topic = f"{'A' if origin == 'synthetic' else 'B'}0{1 + index % 2}"
                    documents.append(
                        document(
                            f"{origin}-{language}-{index}",
                            topic_id=topic,
                            subtopic_id=f"{topic}_S01",
                            origin=origin,
                            language=language,
                        )
                    )
                    # Язык задаёт ось (далеко), тема — маленький сдвиг вдоль
                    # четвёртой координаты (близко). Происхождение — знак.
                    row = [0.0, 0.0, 0.0, 0.0]
                    row[axis] = 10.0 if origin == "synthetic" else -10.0
                    row[3] = 1.0 if topic.endswith("01") else -1.0
                    rows.append(row)
        return Corpus(tuple(documents)), np.array(rows)

    def test_every_cell_is_reported_and_holds_one_language(self):
        corpus, X = self.make()
        splits = {"train": corpus, "validation": corpus, "test": corpus}
        data = {"train": X, "validation": X, "test": X}
        report = run_cells(splits, data)
        self.assertEqual(
            set(report["cells"]),
            {
                f"{origin}/{language}"
                for origin in ("synthetic", "real")
                for language in ("en", "ru", "tg")
            },
        )
        for name, cell in report["cells"].items():
            self.assertEqual(cell["n_documents"]["train"], 4, msg=name)
            # k равно числу тем ИМЕННО этой ячейки, а не корпуса целиком:
            # взяв 20 на ячейку из четырёх документов, прогон посчитал бы
            # метрики пустых кластеров.
            self.assertEqual(cell["k"], 2, msg=name)

    def test_topic_is_found_inside_a_cell_where_language_cannot_explain_it(self):
        corpus, X = self.make()
        splits = {"train": corpus, "validation": corpus, "test": corpus}
        data = {"train": X, "validation": X, "test": X}
        report = run_cells(splits, data)
        for name, cell in report["cells"].items():
            scores = cell["splits"]["test"]["external"]
            self.assertAlmostEqual(scores["topic_id"]["ari"], 1.0, places=6, msg=name)
            # Язык внутри ячейки один, объяснить им разбиение нельзя.
            self.assertEqual(scores["language"]["n_classes"], 1, msg=name)

    def test_gap_is_the_difference_between_layer_means(self):
        corpus, X = self.make()
        splits = {"train": corpus, "validation": corpus, "test": corpus}
        data = {"train": X, "validation": X, "test": X}
        report = run_cells(splits, data)
        for split in ("train", "validation", "test"):
            self.assertAlmostEqual(
                report["gap_ari_topic"][split],
                report["by_layer"]["synthetic"][split]["mean_ari_topic"]
                - report["by_layer"]["real"][split]["mean_ari_topic"],
                places=9,
            )


class NeighbourAgreementTests(unittest.TestCase):
    def test_counts_the_label_of_the_nearest_neighbour(self):
        """Метки расставлены так, что ответ известен заранее: пары документов
        стоят вплотную по языку и врозь по теме."""
        documents = []
        rows = []
        for index, language in enumerate(("en", "en", "ru", "ru")):
            documents.append(
                document(
                    f"d{index}",
                    language=language,
                    topic_id=f"A0{index + 1}",
                    subtopic_id=f"A0{index + 1}_S01",
                )
            )
            rows.append([1.0, 0.0] if language == "en" else [0.0, 1.0])
        report = neighbour_agreement(Corpus(tuple(documents)), np.array(rows))
        agreement = report["same_label_as_nearest_neighbour"]
        # Ближайший сосед всегда той же языковой пары и всегда другой темы.
        self.assertAlmostEqual(agreement["language"], 1.0, places=6)
        self.assertAlmostEqual(agreement["topic_id"], 0.0, places=6)

    def test_chance_level_is_reported_next_to_the_number(self):
        """Согласие 0.5 при двух классах не значит ничего, и без случайного
        уровня рядом это не видно."""
        documents = [
            document(f"d{index}", language="en" if index < 2 else "ru")
            for index in range(4)
        ]
        rows = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        report = neighbour_agreement(Corpus(tuple(documents)), rows)
        # Два языка поровну -> вероятность совпадения у двух случайных = 0.5.
        self.assertAlmostEqual(report["chance_level"]["language"], 0.5, places=6)
        # Тема у всех одна -> совпадение гарантировано, случайный уровень 1.0.
        self.assertAlmostEqual(report["chance_level"]["topic_id"], 1.0, places=6)


class PipelineIsFreeOfLibraryClusteringTests(unittest.TestCase):
    """Запрет sklearn действует и на мост, а не только на сам алгоритм.

    Мост — самое удобное место обойти запрет: здесь и так импортируются чужие
    библиотеки (ollama, httpx), и одна лишняя строка среди них не бросается в
    глаза.
    """

    def test_the_guard_sees_the_pipeline(self):
        names = {name for name in os.listdir(PIPELINE_DIR) if name.endswith(".py")}
        self.assertLessEqual(
            {"dataset.py", "embeddings.py", "experiment.py", "model_io.py"}, names
        )

    def test_no_banned_imports(self):
        paths = [os.path.join(PIPELINE_DIR, name) for name in os.listdir(PIPELINE_DIR)]
        paths.append(os.path.join(BACKEND_ROOT, "cluster_topics.py"))
        for path in sorted(paths):
            if not path.endswith(".py"):
                continue
            with self.subTest(file=os.path.basename(path)):
                with open(path, encoding="utf-8") as handle:
                    self.assertEqual(banned_usages(handle.read()), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
