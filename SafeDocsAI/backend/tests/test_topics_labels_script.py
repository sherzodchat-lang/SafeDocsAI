"""Приведение подписей тем в готовом артефакте (cluster_topics_labels.py).

Главная проверка файла — не «подпись дописалась», а вот эта: ПОВТОРНЫЙ ПРОГОН
НЕ ТРОГАЕТ ФАЙЛ. Приложение узнаёт переобученную модель по sha256 артефакта, и
перезапись «тем же самым» завела бы новую версию модели, обесценив назначения
всех документов до следующей переразметки. То есть холостой запуск скрипта, у
которого «ничего не изменилось», стоил бы ровно столько же, сколько
переобучение.

Вторая по важности — сохранность остального содержимого. Скрипт меняет одну
строку метаданных, а рядом лежат центроиды и матрицы преобразования: потерять
их означало бы получить артефакт, который читается, но назначает темы не пойми
где.

Ни базы, ни сети: корпус и артефакт пишутся во временный каталог.
"""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from topicfixtures import CENTROIDS, cluster_topics, write_cell_artifact  # noqa: E402

from cluster_topics_labels import enrich, topic_names  # noqa: E402
from app.modules.topics.pipeline.dataset import FULL_FILE  # noqa: E402
from app.modules.topics.service import forget_cached_artifacts, load_artifact  # noqa: E402


def record(doc_id: str, language: str, topic_id: str, topic: str) -> dict:
    """Запись корпуса. Колонка topic переведена ВМЕСТЕ с документом.

    Именно поэтому имя темы на нужном языке приходится искать по языку записи,
    а не брать у первого попавшегося документа темы.
    """
    return {
        "id": doc_id,
        "text": "текст документа",
        "language": language,
        "topic_id": topic_id,
        "topic": topic,
        "topic_ru": topic if language == "ru" else "",
        "subtopic_id": f"{topic_id}_S01",
        "dataset_origin": "real",
        "split": "train",
        "word_count": 2,
    }


# Три языка на каждую из трёх тем артефакта фикстуры (T00, T01, T02).
FULL_CORPUS = [
    record("e0", "en", "T00", "Taxes"),
    record("r0", "ru", "T00", "Налоги по-русски"),
    record("t0", "tg", "T00", "Андозҳо"),
    record("e1", "en", "T01", "Law"),
    record("r1", "ru", "T01", "Право по-русски"),
    record("t1", "tg", "T01", "Ҳуқуқ"),
    record("e2", "en", "T02", "Finance"),
    record("r2", "ru", "T02", "Финансы по-русски"),
    record("t2", "tg", "T02", "Молия"),
]


class LabelEnrichmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.addCleanup(forget_cached_artifacts)
        self.root = Path(self._tmpdir.name)
        self.data_dir = self.root / "data"
        self.data_dir.mkdir()
        self.model = self.root / "topic_model.npz"

        # Корпус называет темы теми же topic_id, что и артефакт фикстуры
        # (T00, T01, T02): скрипт связывает их именно по нему.
        self.write_corpus(FULL_CORPUS)

    def write_corpus(self, records) -> None:
        with open(self.data_dir / FULL_FILE, "w", encoding="utf-8") as handle:
            for item in records:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    def write_model(self, *, localized: bool = False) -> None:
        write_cell_artifact(self.model)
        # write_cell_artifact кладёт подписи фикстуры; здесь нужен артефакт
        # ровно в том состоянии, в каком он приехал с обучения, — с переводами
        # или без них.
        with np.load(self.model, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.keys() if name != "meta"}
            meta = json.loads(str(archive["meta"]))
        meta["cluster_topics"] = cluster_topics(localized=localized)
        np.savez_compressed(
            self.model, **arrays, meta=np.array(json.dumps(meta, ensure_ascii=False))
        )

    def digest(self) -> str:
        return hashlib.sha256(self.model.read_bytes()).hexdigest()

    def topics_of_model(self) -> list[dict]:
        with np.load(self.model, allow_pickle=False) as archive:
            return json.loads(str(archive["meta"]))["cluster_topics"]

    def test_every_language_is_written_next_to_the_stable_key(self):
        self.write_model()
        summary = enrich(self.model, topic_names(self.data_dir))

        self.assertTrue(summary["written"])
        topics = {item["cluster"]: item for item in self.topics_of_model()}
        # Ключ темы приводится к английскому: в фикстуре он был русским
        # («Налоги»), ровно как в боевом артефакте, где один кластер из
        # двадцати оказался подписан по-таджикски.
        self.assertEqual(topics[0]["topic"], "Taxes")
        self.assertEqual(topics[0]["topic_ru"], "Налоги по-русски")
        self.assertEqual(topics[0]["topic_tg"], "Андозҳо")

    def test_the_rest_of_the_artifact_survives_the_rewrite(self):
        """Рядом с метаданными лежат центроиды и матрицы преобразования."""
        self.write_model()
        with np.load(self.model, allow_pickle=False) as archive:
            before = {name: np.array(archive[name]) for name in archive.keys() if name != "meta"}

        enrich(self.model, topic_names(self.data_dir))

        with np.load(self.model, allow_pickle=False) as archive:
            after = {name: np.array(archive[name]) for name in archive.keys() if name != "meta"}
        self.assertEqual(sorted(after), sorted(before))
        for name, value in before.items():
            np.testing.assert_array_equal(after[name], value)

    def test_the_enriched_artifact_is_still_readable_by_the_product(self):
        self.write_model()
        enrich(self.model, topic_names(self.data_dir))
        forget_cached_artifacts()

        artifact = load_artifact(self.model)
        self.assertEqual(artifact.cluster_count, len(CENTROIDS))
        self.assertEqual(artifact.label_of(1), "Law")
        self.assertEqual(artifact.label_in(1, "ru"), "Право по-русски")
        self.assertEqual(artifact.label_in(1, "tg"), "Ҳуқуқ")

    def test_a_second_run_does_not_touch_the_file(self):
        """Иначе холостой прогон обесценил бы все назначения документов.

        Новая версия модели заводится по несовпадению sha256 артефакта, и
        перезапись тем же содержимым — это именно несовпадение: сжатие пишет в
        файл время. После неё распределение показывает нули до переразметки.
        """
        self.write_model()
        enrich(self.model, topic_names(self.data_dir))
        after_first = self.digest()

        summary = enrich(self.model, topic_names(self.data_dir))

        self.assertFalse(summary["written"])
        self.assertEqual(summary["changed"], [])
        self.assertEqual(summary["unchanged"], len(CENTROIDS) * 3)
        self.assertEqual(self.digest(), after_first)

    def test_dry_run_reports_but_writes_nothing(self):
        self.write_model()
        before = self.digest()

        summary = enrich(self.model, topic_names(self.data_dir), dry_run=True)

        self.assertFalse(summary["written"])
        self.assertEqual(len(summary["changed"]), len(CENTROIDS) * 3)
        self.assertEqual(self.digest(), before)

    def test_a_topic_without_a_name_in_some_language_is_named_out_loud(self):
        """Молча оставленная чужеязычной тема потерялась бы среди переведённых."""
        self.write_corpus([item for item in FULL_CORPUS if item["id"] != "t2"])
        self.write_model()

        summary = enrich(self.model, topic_names(self.data_dir))

        self.assertEqual(summary["missing"], [(2, "T02", "tg")])
        # Артефакт при этом остаётся рабочим: восемь подписей из девяти лучше
        # нуля.
        self.assertTrue(summary["written"])
        self.assertEqual(len(summary["changed"]), len(CENTROIDS) * 3 - 1)

    def test_a_cluster_without_a_topic_is_not_a_loss(self):
        """Пустой кластер обучающей выборки: темы у него нет вообще."""
        self.write_model()
        with np.load(self.model, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.keys() if name != "meta"}
            meta = json.loads(str(archive["meta"]))
        meta["cluster_topics"][2] = {
            "cluster": 2, "topic_id": "", "topic": "", "share": 0.0, "size": 0
        }
        np.savez_compressed(
            self.model, **arrays, meta=np.array(json.dumps(meta, ensure_ascii=False))
        )

        summary = enrich(self.model, topic_names(self.data_dir))
        self.assertEqual(summary["missing"], [])

    def test_two_names_for_one_topic_in_one_language_are_refused(self):
        """Корпус, противоречащий сам себе, дал бы подпись «как повезёт»."""
        self.write_corpus(
            [
                record("d1", "ru", "T00", "Налоги"),
                record("d2", "ru", "T00", "Налогообложение"),
            ]
        )
        with self.assertRaises(ValueError):
            topic_names(self.data_dir)


if __name__ == "__main__":
    unittest.main()
