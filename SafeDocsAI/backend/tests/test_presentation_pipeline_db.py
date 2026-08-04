"""Пайплайн генерации презентации на настоящем PostgreSQL.

Ollama и ChromaDB не поднимаются: подменены ретривал и ModelManager — ровно
две внешние зависимости пайплайна. Всё остальное настоящее, включая рендер
python-pptx, запись файла и строку в журнале блокнота: именно на стыке «файл
на диске — строка в базе» и живут интересные ошибки.

Что проверяется:

* источники перечитываются в момент генерации, а не при постановке в очередь
  (presentation.no_sources, когда их удалили, пока задача стояла);
* файл появляется атомарно и ДО commit'а со status='ready' — строка не имеет
  права обещать файл, которого нет;
* временный файл подчищается при любом исходе;
* коды отказов: generation_failed, ollama_unavailable;
* дайджест уже написанного доезжает до второго слайд-вызова (мера 2 правила
  «не добивать»).
"""

import json
import os
import sys
import tempfile
import unittest
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dbfixtures import DatabaseBackedTestCase  # noqa: E402

from app.core.exceptions import ExternalServiceError, PresentationErrors  # noqa: E402
from app.modules.presentations.constants import (  # noqa: E402
    STATUS_ERROR,
    STATUS_GENERATING,
    STATUS_QUEUED,
    STATUS_READY,
)
from app.modules.presentations import service as presentation_service  # noqa: E402
from app.modules.presentations.worker import PresentationWorker  # noqa: E402
from app.shared.models import (  # noqa: E402
    Document,
    Log,
    Notebook,
    Presentation,
    User,
)
from app.shared.settings.config import settings as app_settings  # noqa: E402

EMBEDDING_MODEL = "qwen3-embedding:8b"

PLAN_JSON = json.dumps(
    {
        "title": "Налоговые льготы",
        "sections": [
            {"heading": "Кто имеет право", "search_query": "право на льготу"},
            {"heading": "Как оформить", "search_query": "порядок оформления"},
        ],
    },
    ensure_ascii=False,
)


def slide_json(heading: str, bullets: list[str], chunk_id: int) -> str:
    return json.dumps(
        {
            "heading": heading,
            "bullets": bullets,
            "citations": [{"source_id": 1, "chunk_id": chunk_id}],
        },
        ensure_ascii=False,
    )


class FakeModelManager:
    """Очередь заранее заготовленных ответов вместо Ollama.

    Записывает полученные сообщения: по ним проверяется, что в слайд-вызов
    действительно уехал дайджест уже написанного.
    """

    def __init__(self, responses: list, calls: list) -> None:
        self._responses = responses
        self.calls = calls

    async def chat(self, *, model=None, messages=None, num_ctx=None) -> str:
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("модель вызвали больше раз, чем ожидалось")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class PresentationPipelineTestCase(DatabaseBackedTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()

        self.user: User = await self.make_user("owner", "user")
        self.as_user(self.user)
        self.notebook: Notebook = await self.seed(
            Notebook(name="Налоги", domain_profile="tax", owner_id=self.user.id)
        )
        self.document: Document = await self.seed(
            Document(
                name="Кодекс.pdf",
                path=self.make_file("Кодекс.pdf"),
                size=10,
                status="indexed",
                notebook_id=self.notebook.id,
                owner_id=self.user.id,
            )
        )

        env_patcher = patch.object(
            app_settings, "OLLAMA_MODEL_EMBEDDING", EMBEDDING_MODEL
        )
        env_patcher.start()
        self.addCleanup(env_patcher.stop)

        for target in (
            "app.modules.presentations.worker.session_context",
            "app.modules.presentations.service.session_context",
        ):
            patcher = patch(target, self._worker_session)
            patcher.start()
            self.addCleanup(patcher.stop)

        # Каталог файлов — временный: настоящий data/presentations тесты не
        # трогают.
        self.storage = tempfile.TemporaryDirectory()
        self.addCleanup(self.storage.cleanup)
        storage_patcher = patch.object(
            presentation_service, "PRESENTATION_STORAGE_DIR", self.storage.name
        )
        storage_patcher.start()
        self.addCleanup(storage_patcher.stop)

        rag_patcher = patch.object(presentation_service, "RAGService")
        rag_patcher.start()
        self.addCleanup(rag_patcher.stop)

        settings_patcher = patch.object(
            presentation_service, "RuntimeSettingsService", MagicMock()
        )
        runtime_settings = settings_patcher.start()
        runtime_settings.get_settings.return_value = {"chat_model": "test-model"}
        self.addCleanup(settings_patcher.stop)

        self.model_calls: list = []
        self.retrieval_queries: list[str] = []
        self.worker = PresentationWorker(poll_interval=0.05)
        self.addAsyncCleanup(self.worker.stop)

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

    # --- подмены ---

    def use_model(self, responses: list) -> None:
        manager = FakeModelManager(responses, self.model_calls)
        patcher = patch.object(
            presentation_service, "ModelManager", lambda: manager
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def use_retrieval(self, chunk_ids=(1, 2, 3), page: int = 7) -> None:
        async def fake_retrieve(*, search_query, **_kwargs):
            self.retrieval_queries.append(search_query)
            chunks = [
                {
                    "chunk_id": str(chunk_id),
                    "metadata": {
                        "doc_id": self.document.id,
                        "doc_name": self.document.name,
                        "page": page,
                    },
                }
                for chunk_id in chunk_ids
            ]
            texts = {str(chunk_id): f"текст фрагмента {chunk_id}" for chunk_id in chunk_ids}
            return chunks, texts

        patcher = patch.object(presentation_service, "retrieve_for_query", fake_retrieve)
        patcher.start()
        self.addCleanup(patcher.stop)

    # --- данные ---

    async def make_presentation(self, **overrides) -> Presentation:
        fields = {
            "notebook_id": self.notebook.id,
            "owner_id": self.user.id,
            "template_key": "classic",
            "language": "ru",
            "slide_count": 4,
            "description": "Обзор льгот",
            "status": STATUS_QUEUED,
        }
        fields.update(overrides)
        return await self.seed(Presentation(**fields))

    async def run_one_job(self) -> Presentation:
        """Полный путь воркера: захват, генерация, запись исхода."""
        self.assertTrue(await self.worker._claim_and_process())
        row = await self.get_row(Presentation, self._presentation_id)
        self.assertIsNotNone(row)
        return row

    async def prepare(self, **overrides) -> None:
        self._presentation_id = (await self.make_presentation(**overrides)).id

    def storage_files(self) -> list[str]:
        return sorted(os.listdir(self.storage.name))


class SuccessfulGenerationTests(PresentationPipelineTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.use_retrieval()
        self.use_model(
            [
                PLAN_JSON,
                slide_json("Кто имеет право", ["Первый факт", "Второй факт"], 1),
                slide_json("Как оформить", ["Третий факт", "Четвёртый факт"], 2),
            ]
        )
        await self.prepare()

    async def test_ready_row_points_to_a_real_file(self):
        row = await self.run_one_job()

        self.assertEqual(row.status, STATUS_READY)
        self.assertEqual(row.progress, 100)
        self.assertIsNone(row.error_code)
        self.assertTrue(os.path.exists(row.file_path))
        self.assertEqual(row.file_size, os.path.getsize(row.file_path))
        self.assertGreater(row.file_size, 0)

    async def test_no_temporary_file_survives(self):
        row = await self.run_one_job()
        self.assertEqual(self.storage_files(), [os.path.basename(row.file_path)])

    async def test_file_exists_before_the_row_promises_it(self):
        """Порядок «файл, потом commit» — зеркальный удалению.

        Обратный порядок оставил бы после падения между шагами status='ready'
        с путём в никуда: отказ на скачивании уже после «готово».
        """
        observed: list[bool] = []
        original = presentation_service.PresentationsService.mark_ready

        async def spy(session, presentation_id, *, file_path, file_size):
            observed.append(os.path.exists(file_path))
            return await original(
                session, presentation_id, file_path=file_path, file_size=file_size
            )

        with patch.object(
            presentation_service.PresentationsService, "mark_ready", spy
        ):
            await self.run_one_job()

        self.assertEqual(observed, [True])

    async def test_deck_has_the_ordered_slides_and_a_sources_slide(self):
        from pptx import Presentation as PptxPresentation

        row = await self.run_one_job()
        deck = PptxPresentation(row.file_path)
        texts = [
            "\n".join(
                shape.text_frame.text for shape in slide.shapes if shape.has_text_frame
            )
            for slide in deck.slides
        ]

        self.assertEqual(len(texts), row.slide_count)
        self.assertIn("Налоговые льготы", texts[0])
        self.assertIn("Налоги", texts[0])  # имя блокнота
        self.assertIn("Кто имеет право", texts[1])
        self.assertIn("Как оформить", texts[2])
        self.assertIn("Источники", texts[3])
        # Уникальные документы с именами и страницами.
        self.assertIn("Кодекс.pdf", texts[3])
        self.assertIn("стр. 7", texts[3])

    async def test_digest_of_written_bullets_reaches_the_next_slide(self):
        """Мера 2: слайд-вызов видит, что уже сказано на предыдущих."""
        await self.run_one_job()

        plan_call, first_slide, second_slide = self.model_calls
        self.assertNotIn("already_written", first_slide[1]["content"])
        self.assertIn("<already_written>", second_slide[1]["content"])
        self.assertIn("- Первый факт", second_slide[1]["content"])
        self.assertIn("- Второй факт", second_slide[1]["content"])
        # Заголовки предыдущих слайдов в дайджест не входят — только буллеты.
        self.assertNotIn("Кто имеет право\n</already_written>", second_slide[1]["content"])
        # И правило про два буллета стоит в системной части.
        self.assertIn("give exactly two bullets", second_slide[0]["content"])

    async def test_each_section_is_retrieved_by_its_own_query(self):
        await self.run_one_job()
        # Первый вызов — обзорный под план, дальше по одному на секцию.
        self.assertEqual(
            self.retrieval_queries,
            ["Обзор льгот", "право на льготу", "порядок оформления"],
        )

    async def test_journal_records_the_order_and_its_outcome(self):
        row = await self.run_one_job()

        logs = await self.all_rows(Log)
        self.assertEqual(len(logs), 1)
        entry = logs[0]
        self.assertEqual(entry.user_id, self.user.id)
        self.assertEqual(entry.notebook_id, self.notebook.id)
        self.assertIn("classic", entry.question)
        self.assertIn("ru", entry.question)
        self.assertIn(str(row.slide_count), entry.question)
        self.assertIn(STATUS_READY, entry.answer)
        self.assertIn("Кодекс.pdf", entry.sources or "")


class MissingSourcesTests(PresentationPipelineTestCase):
    async def test_no_indexed_sources_left(self):
        """Источники перечитываются в момент генерации, а не при заказе."""
        self.use_retrieval()
        self.use_model([])
        await self.prepare()

        async with self.session_factory() as session:
            document = await session.get(Document, self.document.id)
            document.status = "pending"
            session.add(document)
            await session.commit()

        row = await self.run_one_job()
        self.assertEqual(row.status, STATUS_ERROR)
        self.assertEqual(row.error_code, PresentationErrors.NO_SOURCES)
        self.assertTrue(row.error_text)
        # Ни одного файла: до рендера дело не дошло.
        self.assertEqual(self.storage_files(), [])

    async def test_documents_deleted_while_the_job_was_waiting(self):
        self.use_retrieval()
        self.use_model([])
        await self.prepare()

        async with self.session_factory() as session:
            document = await session.get(Document, self.document.id)
            await session.delete(document)
            await session.commit()

        row = await self.run_one_job()
        self.assertEqual(row.error_code, PresentationErrors.NO_SOURCES)

    async def test_journal_records_the_failure_too(self):
        self.use_retrieval()
        self.use_model([])
        await self.prepare()
        async with self.session_factory() as session:
            document = await session.get(Document, self.document.id)
            document.status = "pending"
            session.add(document)
            await session.commit()

        await self.run_one_job()

        logs = await self.all_rows(Log)
        self.assertEqual(len(logs), 1)
        self.assertIn(STATUS_ERROR, logs[0].answer)
        self.assertIn(PresentationErrors.NO_SOURCES, logs[0].answer)


class FailureTests(PresentationPipelineTestCase):
    async def test_unavailable_ollama_gets_its_own_code(self):
        self.use_retrieval()
        self.use_model(
            [ExternalServiceError("Ollama is unavailable", service="Ollama", status_code=503)]
        )
        await self.prepare()

        row = await self.run_one_job()
        self.assertEqual(row.status, STATUS_ERROR)
        self.assertEqual(row.error_code, PresentationErrors.OLLAMA_UNAVAILABLE)

    async def test_two_invalid_answers_fail_the_job(self):
        """Повтор один; после второго провала — честный отказ, а не починка."""
        self.use_retrieval()
        self.use_model(["не json", "тоже не json"])
        await self.prepare()

        row = await self.run_one_job()
        self.assertEqual(row.error_code, PresentationErrors.GENERATION_FAILED)
        self.assertEqual(len(self.model_calls), 2)
        # Вторая попытка получает исходный ответ и претензию валидатора.
        self.assertEqual(self.model_calls[1][-2]["content"], "не json")
        self.assertIn("rejected by the validator", self.model_calls[1][-1]["content"])

    async def test_retry_saves_a_recoverable_answer(self):
        self.use_retrieval()
        self.use_model(
            [
                "почти json",
                PLAN_JSON,
                slide_json("Кто имеет право", ["Факт", "Ещё факт"], 1),
                slide_json("Как оформить", ["Факт", "Ещё факт"], 2),
            ]
        )
        await self.prepare()

        row = await self.run_one_job()
        self.assertEqual(row.status, STATUS_READY)

    async def test_broken_render_leaves_no_files_behind(self):
        self.use_retrieval()
        self.use_model(
            [
                PLAN_JSON,
                slide_json("Кто имеет право", ["Факт", "Ещё факт"], 1),
                slide_json("Как оформить", ["Факт", "Ещё факт"], 2),
            ]
        )
        await self.prepare()

        with patch.object(
            presentation_service, "render_presentation", side_effect=OSError("диск полон")
        ):
            row = await self.run_one_job()

        self.assertEqual(row.status, STATUS_ERROR)
        self.assertEqual(row.error_code, PresentationErrors.GENERATION_FAILED)
        self.assertIsNone(row.file_path)
        self.assertEqual(self.storage_files(), [])

    async def test_row_deleted_while_generating_is_not_resurrected(self):
        """Отменённый заказ не должен вернуться в базу строкой 'ready'."""
        self.use_retrieval()
        self.use_model([])
        await self.prepare()

        async with self.session_factory() as session:
            row = await session.get(Presentation, self._presentation_id)
            row.status = STATUS_GENERATING
            session.add(row)
            await session.commit()
            await session.delete(row)
            await session.commit()

        # Захватывать нечего: строки нет.
        self.assertFalse(await self.worker._claim_and_process())
        self.assertFalse(await self.exists(Presentation, self._presentation_id))


if __name__ == "__main__":
    unittest.main()
