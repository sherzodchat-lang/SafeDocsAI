"""Удаление чистит ту коллекцию, куда документ был проиндексирован.

Имя коллекции ChromaDB выводится из embedding-модели
(ChromaGateway._collection_name), а модель меняется в админ-панели на живой
системе. До этой правки удаление ходило только в АКТИВНУЮ коллекцию:

  * документ проиндексирован моделью A → векторы в andozai_docs_a;
  * админ переключил настройку на B → активной стала andozai_docs_b;
  * DELETE источника снимал id из andozai_docs_b — там их нет, то есть no-op,
    а строки chunk после commit удалены, и id векторов больше нигде не
    хранятся;
  * админ вернул настройку на A — и «удалённый» документ снова находится в
    поиске. Зомби, переживающий откат настройки.

То же самое было у полной переиндексации (_reindex_all): она сносит старые
чанки из активной коллекции, а коллекция прежней модели оставалась полной
копией всей базы.

Чинится в шлюзе (ChromaGateway.delete_documents), а не в вызывающих: имя
коллекции выводится там, и через delete_documents ходят все пять мест, где
векторы удаляются (индексация с повтором, удаление источника, удаление
блокнота, отложенная очистка воркером, переиндексация).

ChromaDB не поднимается: chromadb.HttpClient подменяется на клиент из этого
файла, который держит коллекции в памяти. Проверяется не факт вызова, а
СОДЕРЖИМОЕ коллекций после удаления — на моке RAGService такую проверку
поставить нельзя, он бы прошёл и на старом коде.

Настоящий PostgreSQL нужен второй половине файла: удаление источника идёт
через настоящий эндпоинт, с внешними ключами, задачами и файлом на диске.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Каталог с тестами в sys.path: unittest discover кладёт его туда сам, но при
# запуске `python -m unittest tests.test_zombie_vectors_db` этого не происходит.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dbfixtures import DatabaseBackedTestCase  # noqa: E402

from app.core.exceptions import ExternalServiceError  # noqa: E402
from app.modules.rag.chroma_gateway import ChromaGateway  # noqa: E402
from app.shared.models import Chunk, Document, Notebook  # noqa: E402
from app.shared.settings.runtime_settings import RuntimeSettingsService  # noqa: E402


SOURCES = "/api/v1/sources"

OLD_MODEL = "qwen3-embedding:8b"
NEW_MODEL = "bge-m3"


class FakeCollection:
    """Коллекция ChromaDB в памяти. Из API нужны count/add/delete."""

    def __init__(self, name: str, metadata: dict | None = None) -> None:
        self.name = name
        self.metadata = metadata or {}
        self.ids: list[str] = []
        self.delete_calls: list[list[str]] = []
        self.delete_error: Exception | None = None

    def count(self) -> int:
        return len(self.ids)

    def add(self, ids=None, **kwargs) -> None:
        self.ids.extend(ids or [])

    def delete(self, ids=None) -> None:
        requested = list(ids or [])
        self.delete_calls.append(requested)
        if self.delete_error is not None:
            raise self.delete_error
        # Настоящая ChromaDB на несуществующий id не ругается — это no-op.
        removed = set(requested)
        self.ids = [stored for stored in self.ids if stored not in removed]


class FakeChromaClient:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}

    def seed_collection(self, name: str, ids: list[str]) -> FakeCollection:
        collection = self.collections.setdefault(name, FakeCollection(name))
        collection.add(ids=list(ids))
        return collection

    # --- то, что зовёт ChromaGateway ---

    def get_or_create_collection(self, name, embedding_function=None, metadata=None):
        return self.collections.setdefault(name, FakeCollection(name, metadata))

    def get_collection(self, name, embedding_function=None):
        if name not in self.collections:
            raise ValueError(f"Collection {name} does not exist")
        return self.collections[name]

    def list_collections(self, limit=None, offset=None):
        return list(self.collections.values())


class ChromaFixtureMixin:
    """Подменённые ChromaDB и файл настроек с выбранной embedding-моделью."""

    def setup_chroma(self, embedding_model: str) -> None:
        self._settings_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._settings_dir.cleanup)
        self.settings_path = Path(self._settings_dir.name) / "runtime_settings.json"

        path_patcher = patch.object(
            RuntimeSettingsService, "_settings_path", return_value=self.settings_path
        )
        path_patcher.start()
        self.addCleanup(path_patcher.stop)
        self.select_embedding_model(embedding_model)

        self.chroma = FakeChromaClient()
        client_patcher = patch("chromadb.HttpClient", return_value=self.chroma)
        client_patcher.start()
        self.addCleanup(client_patcher.stop)

    def select_embedding_model(self, embedding_model: str) -> None:
        """Переключить настройку так же, как это делает админ-панель.

        Пишется файл настроек, а не мок: ChromaGateway читает модель через
        RuntimeSettingsService.get_settings() при каждом создании.
        """
        self.settings_path.write_text(
            json.dumps({"embedding_model": embedding_model}, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def collection_name(embedding_model: str) -> str:
        return ChromaGateway._collection_name(embedding_model)


# --- Шлюз ---------------------------------------------------------------


class DeleteAcrossCollectionsTests(ChromaFixtureMixin, unittest.TestCase):
    """ChromaGateway.delete_documents — там, где выводится имя коллекции."""

    def setUp(self) -> None:
        self.setup_chroma(OLD_MODEL)
        self.old_collection = self.chroma.seed_collection(
            self.collection_name(OLD_MODEL), ["1", "2", "3"]
        )
        # Модель сменили: активной стала другая коллекция, а векторы документа
        # остались в прежней.
        self.select_embedding_model(NEW_MODEL)

    def test_vectors_are_removed_from_the_collection_of_the_previous_model(self):
        ChromaGateway().delete_documents(["1", "2"])

        self.assertEqual(
            self.old_collection.ids,
            ["3"],
            "векторы удалённого документа остались в коллекции прежней модели",
        )

    def test_active_collection_is_cleaned_too(self):
        """Обычный случай (модель не менялась) не должен пострадать."""
        self.select_embedding_model(OLD_MODEL)

        ChromaGateway().delete_documents(["1"])

        self.assertEqual(self.old_collection.ids, ["2", "3"])

    def test_foreign_collections_are_left_alone(self):
        alien = self.chroma.seed_collection("some_other_app", ["1"])

        ChromaGateway().delete_documents(["1"])

        self.assertEqual(alien.delete_calls, [], "чужая коллекция не наша забота")
        self.assertEqual(alien.ids, ["1"])

    def test_every_collection_of_ours_is_visited(self):
        third = self.chroma.seed_collection(
            self.collection_name("nomic-embed-text"), ["1", "9"]
        )

        ChromaGateway().delete_documents(["1"])

        self.assertEqual(third.ids, ["9"])
        self.assertEqual(self.old_collection.ids, ["2", "3"])
        active = self.chroma.collections[self.collection_name(NEW_MODEL)]
        self.assertEqual(active.delete_calls, [["1"]])

    def test_failure_on_a_previous_collection_is_not_swallowed(self):
        """Молча оставленные векторы — это и есть исходный дефект.

        Вызывающие с отказом жить умеют: удаление блокнота ставит задачу
        cleanup_embeddings, удаление источника отвечает 503.
        """
        self.old_collection.delete_error = RuntimeError("ChromaDB is down")

        with self.assertRaises(ExternalServiceError):
            ChromaGateway().delete_documents(["1"])

    def test_empty_id_list_touches_nothing(self):
        ChromaGateway().delete_documents([])

        self.assertEqual(self.old_collection.delete_calls, [])

    def test_unlistable_collections_do_not_block_the_active_one(self):
        """list_collections отказала — активную коллекцию всё равно чистим.

        Отказ здесь означает, что ChromaDB лежит целиком, и удаление из
        активной коллекции упадёт следом само.
        """
        self.select_embedding_model(OLD_MODEL)
        with patch.object(
            FakeChromaClient, "list_collections", side_effect=RuntimeError("boom")
        ):
            ChromaGateway().delete_documents(["1"])

        self.assertEqual(self.old_collection.ids, ["2", "3"])


# --- Эндпоинт -----------------------------------------------------------


class DeletedDocumentLeavesNoZombieVectorsTests(
    ChromaFixtureMixin, DatabaseBackedTestCase
):
    """Полный путь: DELETE /api/v1/sources/{id} после смены модели."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.setup_chroma(OLD_MODEL)

        self.user = await self.make_user("owner", "user")
        self.as_user(self.user)
        self.notebook = await self.seed(
            Notebook(name="Блокнот", domain_profile="general", owner_id=self.user.id)
        )

        self.doomed, self.doomed_ids = await self.indexed_document("удаляемый.txt")
        self.survivor, self.survivor_ids = await self.indexed_document("соседний.txt")

    async def indexed_document(self, name: str) -> tuple[Document, list[str]]:
        """Документ, чьи векторы лежат в коллекции текущей модели."""
        document = await self.seed(
            Document(
                name=name,
                path=self.make_file(name),
                size=42,
                notebook_id=self.notebook.id,
                owner_id=self.user.id,
                status="indexed",
            )
        )
        chunks = [
            Chunk(text=f"фрагмент {index} из {name}", page=1, chunk_index=index,
                  doc_id=document.id)
            for index in range(3)
        ]
        await self.seed(*chunks)
        chunk_ids = [str(chunk.id) for chunk in chunks]
        self.chroma.seed_collection(self.collection_name(OLD_MODEL), chunk_ids)
        return document, chunk_ids

    async def test_deleting_after_a_model_switch_clears_the_old_collection(self):
        self.select_embedding_model(NEW_MODEL)

        response = await self.client.delete(f"{SOURCES}/{self.doomed.id}")

        self.assertEqual(response.status_code, 200, response.text)
        old = self.chroma.collections[self.collection_name(OLD_MODEL)]
        for chunk_id in self.doomed_ids:
            self.assertNotIn(
                chunk_id,
                old.ids,
                "вектор удалённого документа остался в коллекции прежней модели: "
                "вернув настройку назад, админ снова найдёт его в поиске",
            )
        self.assertFalse(await self.exists(Document, self.doomed.id))
        self.assertEqual(
            await self.rows_where(Chunk, Chunk.doc_id == self.doomed.id), []
        )

    async def test_vectors_of_other_documents_survive(self):
        """Чистка по id, а не «снести коллекцию»: соседей трогать нельзя."""
        self.select_embedding_model(NEW_MODEL)

        await self.client.delete(f"{SOURCES}/{self.doomed.id}")

        old = self.chroma.collections[self.collection_name(OLD_MODEL)]
        self.assertEqual(sorted(old.ids, key=int), sorted(self.survivor_ids, key=int))
        self.assertTrue(await self.exists(Document, self.survivor.id))

    async def test_delete_without_a_model_switch_still_works(self):
        response = await self.client.delete(f"{SOURCES}/{self.doomed.id}")

        self.assertEqual(response.status_code, 200, response.text)
        old = self.chroma.collections[self.collection_name(OLD_MODEL)]
        self.assertEqual(sorted(old.ids, key=int), sorted(self.survivor_ids, key=int))

    async def test_failed_cleanup_of_the_old_collection_answers_503(self):
        """Документ не должен исчезнуть из БД, пока векторы не сняты."""
        self.select_embedding_model(NEW_MODEL)
        self.chroma.collections[self.collection_name(OLD_MODEL)].delete_error = (
            RuntimeError("ChromaDB is down")
        )

        response = await self.client.delete(f"{SOURCES}/{self.doomed.id}")

        self.assertEqual(response.status_code, 503, response.text)
        self.assertTrue(await self.exists(Document, self.doomed.id))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
