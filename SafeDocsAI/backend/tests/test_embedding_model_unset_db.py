"""Embedding-модель не задана: честный отказ вместо пустой коллекции.

Что закрепляем.

  * **Умолчания у embedding-модели больше нет нигде.** Имя коллекции ChromaDB
    выводится из названия модели (ChromaGateway._collection_name), а
    get_or_create_collection на незнакомое имя не отказывает, а СОЗДАЁТ пустую
    коллекцию. Пока в коде жило умолчание "nomic-embed-text", любой процесс,
    поднятый без OLLAMA_MODEL_EMBEDDING, заводил себе рядом пустую
    andozai_docs_nomic_embed_text и отвечал на поиск пустотой при полной базе —
    без единой ошибки в журнале. Ровно это и нашли на стенде, работающем на
    qwen3-embedding:8b.
  * **Порядок разрешения:** runtime_settings.json -> переменная окружения
    OLLAMA_MODEL_EMBEDDING -> отказ. Третьего шага (умолчания в коде) нет, и
    «не задано» — это пустая строка, как у contextual_embedding_model.
  * **Отказ мягкий.** Приложение стартует, GET и PUT /api/v1/settings/
    работают — иначе модель негде выбрать, — а операции, которым нужна
    векторная база, отвечают 503 с кодом settings.embedding_model_unset.
  * **Проверка стоит в конструкторе ChromaGateway**, через который проходит
    каждый путь к векторам. Поэтому здесь проверяются РАЗНЫЕ пути (удаление
    источника, разбор поиска), а не один эндпоинт: проверка в отдельных
    методах пропустила бы соседний путь.
  * **Воркер индексации задачи не берёт.** Ответить 503 ему некому, он не
    HTTP; провалить задачу значило бы перевести документы в 'error' и
    потребовать повторной загрузки из-за одной ненастроенной строки.
  * **Строка в журнале при старте** называет модель, коллекцию и число
    векторов — либо честно говорит, что модели нет.

Настоящий PostgreSQL нужен ради подмены deps.get_current_user и настоящих
строк document/chunk/job. ChromaDB подменена клиентом в памяти, Ollama не
участвует: удаление векторов ничего не считает.
"""

import json
import os
import sys
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

# Каталог с тестами в sys.path: unittest discover кладёт его туда сам, но при
# запуске `python -m unittest tests.test_embedding_model_unset_db` — нет.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dbfixtures import DatabaseBackedTestCase  # noqa: E402

from app.core.exceptions import (  # noqa: E402
    EmbeddingModelNotConfigured,
    SettingsErrors,
)
from app.core.rate_limit import chat_limiter  # noqa: E402
from app.modules.jobs.service import JOB_CLEANUP_EMBEDDINGS  # noqa: E402
from app.modules.jobs.worker import IndexingWorker  # noqa: E402
from app.modules.rag.chroma_gateway import (  # noqa: E402
    ChromaGateway,
    log_vector_store_state,
)
from app.modules.rag.constants import DEFAULT_EMBEDDING_MODEL  # noqa: E402
from app.modules.rag.service import RAGService  # noqa: E402
from app.shared.models import Chunk, Document, Job, Notebook  # noqa: E402
from app.shared.settings.config import Settings, settings as app_settings  # noqa: E402
from app.shared.settings.runtime_settings import RuntimeSettingsService  # noqa: E402


SETTINGS = "/api/v1/settings/"
SOURCES = "/api/v1/sources"
RETRIEVE = "/api/v1/chat/retrieve"

# Модель «из переменной окружения» и модель «из файла настроек» — разные
# намеренно: только так видно, кто из них победил.
ENV_MODEL = "qwen3-embedding:8b"
FILE_MODEL = "bge-m3"
CHAT_MODEL = "gemma4:26b"

FAKE_CATALOG = {
    "available_models": [CHAT_MODEL, ENV_MODEL, FILE_MODEL],
    "available_chat_models": [CHAT_MODEL],
    "available_embedding_models": [ENV_MODEL, FILE_MODEL],
    "ollama_available": True,
    "ollama_error": None,
}

GATEWAY_LOGGER = "app.modules.rag.chroma_gateway"
WORKER_LOGGER = "app.modules.jobs.worker"


class FakeCollection:
    """Коллекция ChromaDB в памяти: нужны count, add и delete."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.ids: list[str] = []

    def count(self) -> int:
        return len(self.ids)

    def add(self, ids=None, **kwargs) -> None:
        self.ids.extend(ids or [])

    def delete(self, ids=None) -> None:
        removed = set(ids or [])
        self.ids = [stored for stored in self.ids if stored not in removed]


class FakeChromaClient:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}

    def seed_collection(self, name: str, ids: list[str]) -> FakeCollection:
        collection = self.collections.setdefault(name, FakeCollection(name))
        collection.add(ids=list(ids))
        return collection

    def get_or_create_collection(self, name, embedding_function=None, metadata=None):
        # Ровно то, чем опасна настоящая ChromaDB: незнакомое имя не отказ, а
        # новая пустая коллекция.
        return self.collections.setdefault(name, FakeCollection(name))

    def get_collection(self, name, embedding_function=None):
        if name not in self.collections:
            raise ValueError(f"Collection {name} does not exist")
        return self.collections[name]

    def list_collections(self, limit=None, offset=None):
        return list(self.collections.values())


class EmbeddingModelMixin:
    """Файл настроек во временном каталоге, переменная окружения и ChromaDB."""

    def set_up_embedding_model(self, *, env_model: str = "") -> None:
        self._settings_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._settings_dir.cleanup)
        self.settings_path = Path(self._settings_dir.name) / "runtime_settings.json"

        path_patcher = patch.object(
            RuntimeSettingsService, "_settings_path", return_value=self.settings_path
        )
        path_patcher.start()
        self.addCleanup(path_patcher.stop)

        catalog_patcher = patch.object(
            RuntimeSettingsService, "model_catalog", return_value=FAKE_CATALOG
        )
        catalog_patcher.start()
        self.addCleanup(catalog_patcher.stop)

        self.set_env_model(env_model)

        self.chroma = FakeChromaClient()
        client_patcher = patch("chromadb.HttpClient", return_value=self.chroma)
        client_patcher.start()
        self.addCleanup(client_patcher.stop)

        # Жалоба о незаданной модели пишется один раз за жизнь процесса —
        # снимаем отметку, иначе порядок тестов решал бы, увидит ли её
        # assertLogs.
        ChromaGateway._unset_model_reported = False
        self.addCleanup(setattr, ChromaGateway, "_unset_model_reported", False)

    def set_env_model(self, model: str) -> None:
        """Задать (или убрать) OLLAMA_MODEL_EMBEDDING — второй шаг разрешения."""
        patcher = patch.object(app_settings, "OLLAMA_MODEL_EMBEDDING", model)
        patcher.start()
        self.addCleanup(patcher.stop)

    def set_file_model(self, model: str) -> None:
        """Записать модель в файл настроек — первый шаг разрешения."""
        self.settings_path.write_text(
            json.dumps({"embedding_model": model}, ensure_ascii=False),
            encoding="utf-8",
        )

    def collection_name(self, model: str) -> str:
        return ChromaGateway._collection_name(model)


# --- Порядок разрешения --------------------------------------------------


class ResolutionOrderTests(EmbeddingModelMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.set_up_embedding_model()

    def test_nothing_anywhere_means_unset(self):
        """Ни файла, ни переменной — и никакого имени взамен."""
        self.assertEqual(RuntimeSettingsService.embedding_model(), "")
        self.assertEqual(RuntimeSettingsService.get_settings()["embedding_model"], "")

    def test_the_environment_variable_is_the_second_step(self):
        self.set_env_model(ENV_MODEL)

        self.assertEqual(RuntimeSettingsService.embedding_model(), ENV_MODEL)

    def test_the_settings_file_wins_over_the_environment(self):
        """Выбор админа сильнее окружения: иначе его нельзя было бы сделать."""
        self.set_env_model(ENV_MODEL)
        self.set_file_model(FILE_MODEL)

        self.assertEqual(RuntimeSettingsService.embedding_model(), FILE_MODEL)

    def test_an_empty_value_in_the_file_falls_back_to_the_environment(self):
        """Пусто в файле — это «не выбрано», а не «выбрано пустое»."""
        self.set_env_model(ENV_MODEL)
        self.set_file_model("")

        self.assertEqual(RuntimeSettingsService.embedding_model(), ENV_MODEL)

    def test_no_model_name_is_hardcoded_anywhere(self):
        """Сеть на регресс: верните умолчание — и покраснеет здесь.

        Именно так дефект и выглядел: умолчание жило в трёх местах сразу
        (config.py, .env.example, start.sh), и ни одно из них не было заметно
        на глаз. Проверяется объявленное значение поля, а не текущее: текущее
        подменено этим тестом.
        """
        self.assertEqual(
            Settings.model_fields["OLLAMA_MODEL_EMBEDDING"].default,
            "",
            "у OLLAMA_MODEL_EMBEDDING снова появилось умолчание: процесс без "
            "этой переменной уйдёт в чужую пустую коллекцию ChromaDB молча",
        )
        self.assertEqual(RuntimeSettingsService.DEFAULTS["embedding_model"], "")
        self.assertEqual(DEFAULT_EMBEDDING_MODEL, "")


# --- Шлюз к ChromaDB -----------------------------------------------------


class GatewayRefusesWithoutAModelTests(EmbeddingModelMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.set_up_embedding_model()

    def test_the_gateway_refuses_instead_of_creating_an_empty_collection(self):
        with self.assertLogs(GATEWAY_LOGGER, level="ERROR"):
            with self.assertRaises(EmbeddingModelNotConfigured) as raised:
                ChromaGateway()

        self.assertEqual(
            raised.exception.error_code, SettingsErrors.EMBEDDING_MODEL_UNSET
        )
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(
            self.chroma.collections,
            {},
            "коллекция создана до отказа — именно так и заводится пустая",
        )

    def test_the_refusal_covers_rag_service_too(self):
        """RAGService — это тот же шлюз, и обходного пути мимо него нет."""
        with self.assertRaises(EmbeddingModelNotConfigured):
            RAGService()

    def test_the_environment_variable_is_enough_to_work(self):
        self.set_env_model(ENV_MODEL)

        gateway = ChromaGateway()

        self.assertEqual(gateway.embedding_model, ENV_MODEL)
        self.assertIn(self.collection_name(ENV_MODEL), self.chroma.collections)

    def test_the_file_decides_which_collection_is_used(self):
        """Настройка сильнее окружения — и это видно по имени коллекции."""
        self.set_env_model(ENV_MODEL)
        self.set_file_model(FILE_MODEL)

        gateway = ChromaGateway()

        self.assertEqual(gateway.embedding_model, FILE_MODEL)
        self.assertIn(self.collection_name(FILE_MODEL), self.chroma.collections)
        self.assertNotIn(self.collection_name(ENV_MODEL), self.chroma.collections)


# --- Строка в журнале при старте -----------------------------------------


class StartupLineTests(EmbeddingModelMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.set_up_embedding_model()

    def test_the_line_names_the_model_the_collection_and_the_count(self):
        self.set_env_model(ENV_MODEL)
        self.chroma.seed_collection(self.collection_name(ENV_MODEL), ["1", "2", "3"])

        with self.assertLogs(GATEWAY_LOGGER, level="INFO") as captured:
            self.assertIsNone(log_vector_store_state())

        line = "\n".join(captured.output)
        self.assertIn(f"embedding_model={ENV_MODEL}", line)
        self.assertIn(self.collection_name(ENV_MODEL), line)
        self.assertIn("векторов 3", line)

    def test_the_line_says_plainly_that_no_model_is_set(self):
        with self.assertLogs(GATEWAY_LOGGER, level="ERROR") as captured:
            # None, а не отказ: незаданная модель старт не роняет, иначе
            # некуда прийти и выбрать её.
            self.assertIsNone(log_vector_store_state())

        line = "\n".join(captured.output)
        self.assertIn("embedding_model не задан", line)
        self.assertIn(SettingsErrors.EMBEDDING_MODEL_UNSET, line)

    def test_an_unreachable_chromadb_does_not_break_the_line(self):
        """Число векторов — сетевой запрос, и он не имеет права ронять старт."""
        self.set_env_model(ENV_MODEL)

        # Оба клиента: в development у шлюза есть ещё и локальное
        # persist-хранилище, и без подмены проверка ушла бы писать файлы в
        # backend/data/chroma вместо того, чтобы изобразить отказ.
        with patch("chromadb.HttpClient", side_effect=RuntimeError("нет связи")), patch(
            "chromadb.PersistentClient", side_effect=RuntimeError("нет диска")
        ):
            with self.assertLogs(GATEWAY_LOGGER, level="ERROR") as captured:
                chroma_error = log_vector_store_state()

        line = "\n".join(captured.output)
        self.assertIn(f"embedding_model={ENV_MODEL}", line)
        self.assertIn(self.collection_name(ENV_MODEL), line)
        # Отказ возвращён наверх: в production он по-прежнему запрещает старт.
        self.assertIsNotNone(chroma_error)


# --- HTTP: настройки живут, RAG отвечает 503 -----------------------------


class HttpSurfaceTestCase(EmbeddingModelMixin, DatabaseBackedTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.set_up_embedding_model()

        self.admin = await self.make_user("root", "admin")
        self.as_user(self.admin)
        self.notebook = await self.seed(
            Notebook(name="Блокнот", domain_profile="general", owner_id=self.admin.id)
        )
        self.document = await self.seed(
            Document(
                name="устав.txt",
                path=self.make_file("устав.txt"),
                size=42,
                notebook_id=self.notebook.id,
                owner_id=self.admin.id,
                status="indexed",
            )
        )
        self.chunks = [
            Chunk(text=f"фрагмент {index}", page=1, chunk_index=index,
                  doc_id=self.document.id)
            for index in range(3)
        ]
        await self.seed(*self.chunks)
        self.chunk_ids = [str(chunk.id) for chunk in self.chunks]

        chat_limiter.clients.clear()
        self.addCleanup(chat_limiter.clients.clear)


class SettingsStayReachableTests(HttpSurfaceTestCase):
    """Экран, на котором это чинят, обязан открываться и сохраняться."""

    async def test_get_settings_works_without_a_model(self):
        response = await self.client.get(SETTINGS)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["embedding_model"], "")

    async def test_the_model_can_be_chosen_through_the_api(self):
        response = await self.client.put(
            SETTINGS, json={"embedding_model": FILE_MODEL}
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["embedding_model"], FILE_MODEL)
        # Подтверждение переиндексации здесь не требуется: менять не с чего —
        # без модели ничего и не было проиндексировано.
        self.assertFalse(response.json()["reindex_required"])
        self.assertEqual(RuntimeSettingsService.embedding_model(), FILE_MODEL)


class RagOperationsAnswer503Tests(HttpSurfaceTestCase):
    def assertRefused(self, response) -> None:
        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(
            response.json().get("error_code"),
            SettingsErrors.EMBEDDING_MODEL_UNSET,
            response.text,
        )

    async def test_deleting_a_source_is_refused_with_the_machine_code(self):
        response = await self.client.delete(f"{SOURCES}/{self.document.id}")

        self.assertRefused(response)
        self.assertTrue(
            await self.exists(Document, self.document.id),
            "документ исчез, а его векторы удалить было нечем",
        )

    async def test_retrieval_is_refused_with_the_same_code(self):
        """Другой путь к тому же шлюзу: проверка стоит не у одного эндпоинта."""
        response = await self.client.post(RETRIEVE, json={"question": "Какая ставка?"})

        self.assertRefused(response)

    async def test_the_very_same_request_works_once_the_model_is_set(self):
        self.set_env_model(ENV_MODEL)
        self.chroma.seed_collection(self.collection_name(ENV_MODEL), self.chunk_ids)

        response = await self.client.delete(f"{SOURCES}/{self.document.id}")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            self.chroma.collections[self.collection_name(ENV_MODEL)].ids, []
        )
        self.assertFalse(await self.exists(Document, self.document.id))

    async def test_the_settings_file_decides_which_collection_is_cleaned(self):
        """Модель из файла перекрывает переменную окружения и на этом пути."""
        self.set_env_model(ENV_MODEL)
        self.set_file_model(FILE_MODEL)
        self.chroma.seed_collection(self.collection_name(FILE_MODEL), self.chunk_ids)

        response = await self.client.delete(f"{SOURCES}/{self.document.id}")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            self.chroma.collections[self.collection_name(FILE_MODEL)].ids, []
        )


# --- Воркер индексации ---------------------------------------------------


class WorkerWaitsForTheModelTests(EmbeddingModelMixin, DatabaseBackedTestCase):
    """503 воркеру возвращать некому: он просто не берёт задачи.

    Взята задача очистки векторов, а не индексации: у неё тот же захват
    (_claim_and_process) и тот же шлюз, но обработчик целиком мокается — без
    файлов, OCR и Ollama.
    """

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.set_up_embedding_model()

        self.user = await self.make_user("owner", "user")
        self.as_user(self.user)
        self.worker = IndexingWorker()

        session_patcher = patch(
            "app.modules.jobs.worker.session_context", self._worker_session
        )
        session_patcher.start()
        self.addCleanup(session_patcher.stop)

        rag_patcher = patch("app.modules.jobs.worker.RAGService")
        self.worker_rag = rag_patcher.start()
        self.addCleanup(rag_patcher.stop)

        self.job = await self.seed(
            Job(
                job_type=JOB_CLEANUP_EMBEDDINGS,
                status="queued",
                payload_json=json.dumps({"chunk_ids": ["1", "2"]}),
                created_by=self.user.id,
            )
        )

    @asynccontextmanager
    async def _worker_session(self):
        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def test_the_queue_waits_instead_of_failing(self):
        with self.assertLogs(WORKER_LOGGER, level="ERROR") as captured:
            claimed = await self.worker._claim_and_process()

        self.assertFalse(claimed)
        self.worker_rag.assert_not_called()
        job = await self.get_row(Job, self.job.id)
        self.assertEqual(
            job.status,
            "queued",
            "задача провалена из-за настройки — её пришлось бы ставить заново",
        )
        self.assertIn(
            SettingsErrors.EMBEDDING_MODEL_UNSET, "\n".join(captured.output)
        )

    async def test_the_queue_moves_as_soon_as_the_model_is_set(self):
        """Ничего переставлять руками не нужно: очередь трогается сама."""
        self.set_env_model(ENV_MODEL)

        claimed = await self.worker._claim_and_process()

        self.assertTrue(claimed)
        job = await self.get_row(Job, self.job.id)
        self.assertEqual(job.status, "completed")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
