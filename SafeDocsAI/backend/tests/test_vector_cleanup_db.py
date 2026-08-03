"""Отложенная очистка векторов после удаления блокнота.

Раньше отказ ChromaDB при удалении блокнота проглатывался: строки chunk уже
удалены вместе с документами, id векторов после commit не хранятся больше
нигде, и висячие векторы оставались в коллекции навсегда — в поиске они видны
как цитаты из удалённого документа.

Теперь (см. app/api/endpoints/notebooks.py, шаг 8 delete_notebook) удаление
всё равно коммитится, а осиротевшие id уезжают в задачу cleanup_embeddings,
которую дочищает воркер. В ответе появилось поле vector_cleanup:

    done     — ChromaDB отработала сразу;
    deferred — ChromaDB отказала, задача поставлена;
    failed   — не удалось даже поставить задачу (лежит и БД); id ушли в лог.

Почему настоящий PostgreSQL. Проверяется не только код ответа, но и строка в
таблице job с полным списком chunk_id в payload, а также то, что воркер эту
задачу забирает и закрывает. Вдобавок именно внешние ключи объясняют, почему
id удалённых документа и блокнота лежат в payload, а не в job.source_id и
job.notebook_id: строк, на которые они ссылаются, уже нет
(см. JobRowsRespectForeignKeysTests).

ChromaDB и Ollama не поднимаются: RAGService замокан, а lifespan через
ASGITransport не запускается, поэтому настоящий воркер в фоне не стартует —
его обработчик вызывается тестом явно.
"""

import json
import os
import sys
import unittest
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

from sqlalchemy.exc import IntegrityError

# Каталог с тестами в sys.path: unittest discover кладёт его туда сам, но при
# запуске `python -m unittest tests.test_vector_cleanup_db` этого не происходит.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dbfixtures import DatabaseBackedTestCase  # noqa: E402

from app.modules.jobs.service import (  # noqa: E402
    CLEANUP_MAX_ATTEMPTS,
    JOB_CLEANUP_EMBEDDINGS,
    JOB_INDEX_DOCUMENT,
    JobsService,
)
from app.modules.jobs.worker import IndexingWorker  # noqa: E402
from app.shared.models import Chunk, Document, Job, Notebook  # noqa: E402
from app.shared.settings.config import settings as app_settings  # noqa: E402


# Воркер не берёт задачи, пока не задана embedding-модель: имя коллекции
# ChromaDB выводится из неё, и без модели ни очистка векторов, ни индексация
# невозможны (см. IndexingWorker._embedding_model_is_set). Здесь проверяется
# не это, поэтому модель задана переменной окружения — как на рабочем стенде.
EMBEDDING_MODEL = "qwen3-embedding:8b"


class VectorCleanupTestCase(DatabaseBackedTestCase):
    """Блокнот с документами и чанками плюс замоканная ChromaDB."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()

        self.user = await self.make_user("owner", "user")
        self.as_user(self.user)

        # ChromaDB в тесте не поднимаем: интересен факт вызова и переданные id.
        # Патчим там, где delete_notebook импортирует класс.
        rag_patcher = patch("app.modules.rag.service.RAGService")
        self.rag_cls = rag_patcher.start()
        self.addCleanup(rag_patcher.stop)
        self.rag_delete = self.rag_cls.return_value.delete_documents

        env_patcher = patch.object(
            app_settings, "OLLAMA_MODEL_EMBEDDING", EMBEDDING_MODEL
        )
        env_patcher.start()
        self.addCleanup(env_patcher.stop)

    async def make_notebook(self, name: str = "Блокнот") -> Notebook:
        return await self.seed(
            Notebook(name=name, description=None, domain_profile="general",
                     owner_id=self.user.id)
        )

    async def make_document(self, name: str, notebook_id: int | None) -> Document:
        return await self.seed(
            Document(name=name, path=self.make_file(name), size=42,
                     notebook_id=notebook_id, owner_id=self.user.id)
        )

    async def make_chunks(self, document: Document, count: int) -> list[Chunk]:
        chunks = [
            Chunk(text=f"фрагмент {index} документа {document.name}", page=1,
                  chunk_index=index, embedding_id=f"emb-{document.id}-{index}",
                  doc_id=document.id)
            for index in range(count)
        ]
        await self.seed(*chunks)
        return chunks

    async def notebook_with_chunks(
        self, documents: int = 2, chunks_per_document: int = 3
    ) -> tuple[Notebook, list[str]]:
        """Блокнот, у которого чанки разложены по нескольким документам.

        Несколько документов здесь принципиальны: в задачу должен уехать
        полный список id, а не чанки одного документа.
        """
        notebook = await self.make_notebook()
        chunk_ids: list[str] = []
        for number in range(documents):
            document = await self.make_document(f"doc-{number}.txt", notebook.id)
            chunks = await self.make_chunks(document, chunks_per_document)
            chunk_ids.extend(str(chunk.id) for chunk in chunks)
        return notebook, chunk_ids

    async def cleanup_jobs(self) -> list[Job]:
        return await self.rows_where(Job, Job.job_type == JOB_CLEANUP_EMBEDDINGS)

    async def delete_notebook(self, notebook_id: int):
        return await self.client.delete(f"/api/v1/notebooks/{notebook_id}")


# --- Почему id вообще приходится класть в payload -----------------------


class JobRowsRespectForeignKeysTests(VectorCleanupTestCase):
    """Без этой проверки остальные тесты файла ничего не доказывают.

    Если внешние ключи в тестовой схеме не применяются, задача очистки могла
    бы просто сослаться на удалённый документ, и весь механизм с payload был
    бы лишним.
    """

    async def test_job_cannot_reference_a_deleted_document(self):
        with self.assertRaises(IntegrityError):
            async with self.session_factory() as session:
                session.add(Job(job_type=JOB_INDEX_DOCUMENT, source_id=10_000_019))
                await session.commit()

    async def test_job_cannot_reference_a_deleted_notebook(self):
        with self.assertRaises(IntegrityError):
            async with self.session_factory() as session:
                session.add(
                    Job(job_type=JOB_CLEANUP_EMBEDDINGS, notebook_id=10_000_019)
                )
                await session.commit()


# --- ChromaDB на месте --------------------------------------------------


class VectorCleanupDoneTests(VectorCleanupTestCase):
    async def test_working_chroma_reports_done_and_creates_no_job(self):
        notebook, chunk_ids = await self.notebook_with_chunks()

        response = await self.delete_notebook(notebook.id)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["vector_cleanup"], "done")
        self.rag_delete.assert_called_once()
        self.assertEqual(
            sorted(self.rag_delete.call_args.args[0], key=int),
            sorted(chunk_ids, key=int),
        )
        self.assertEqual(await self.cleanup_jobs(), [],
                         "при работающей ChromaDB задача очистки не нужна")
        self.assertFalse(await self.exists(Notebook, notebook.id))

    async def test_notebook_without_chunks_reports_done_without_calling_chroma(self):
        notebook = await self.make_notebook("Блокнот без чанков")
        await self.make_document("пустой.txt", notebook.id)

        response = await self.delete_notebook(notebook.id)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["vector_cleanup"], "done")
        self.rag_delete.assert_not_called()
        self.assertEqual(await self.cleanup_jobs(), [])


# --- ChromaDB лежит -----------------------------------------------------


class VectorCleanupDeferredTests(VectorCleanupTestCase):
    """Отказ ChromaDB после commit не проглатывается и не теряет id."""

    def fail_chroma(self, message: str = "ChromaDB is down") -> None:
        self.rag_delete.side_effect = RuntimeError(message)

    async def test_failing_chroma_still_deletes_notebook_and_reports_deferred(self):
        notebook, _ = await self.notebook_with_chunks()
        self.fail_chroma()

        response = await self.delete_notebook(notebook.id)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["vector_cleanup"], "deferred")
        self.assertEqual(response.json()["id"], notebook.id)
        # Удаление именно закоммичено: откатывать его нельзя, ChromaDB отказала
        # уже после commit.
        self.assertFalse(await self.exists(Notebook, notebook.id))
        self.assertEqual(await self.all_rows(Document), [])
        self.assertEqual(await self.all_rows(Chunk), [])

    async def test_deferred_job_carries_every_orphan_chunk_id(self):
        notebook, chunk_ids = await self.notebook_with_chunks(
            documents=3, chunks_per_document=4
        )
        self.assertEqual(len(chunk_ids), 12, "предусловие: чанков должно быть много")
        self.fail_chroma()

        response = await self.delete_notebook(notebook.id)

        self.assertEqual(response.json()["vector_cleanup"], "deferred")
        jobs = await self.cleanup_jobs()
        self.assertEqual(len(jobs), 1, "задача очистки не создана")
        job = jobs[0]
        self.assertEqual(job.status, "queued")
        self.assertEqual(job.created_by, self.user.id)

        payload = json.loads(job.payload_json)
        self.assertEqual(
            sorted(payload["chunk_ids"], key=int), sorted(chunk_ids, key=int),
            "в payload должен лежать ПОЛНЫЙ список id удалённых чанков",
        )
        self.assertEqual(payload["notebook_id"], notebook.id)

    async def test_deferred_job_keeps_ids_only_in_payload(self):
        """source_id и notebook_id остаются пустыми: строк уже нет.

        Заполнить их нельзя физически — внешние ключи такую задачу не примут
        (см. JobRowsRespectForeignKeysTests).
        """
        notebook, _ = await self.notebook_with_chunks(documents=1)
        self.fail_chroma()

        await self.delete_notebook(notebook.id)

        job = (await self.cleanup_jobs())[0]
        self.assertIsNone(job.source_id)
        self.assertIsNone(job.notebook_id)

    async def test_chunk_ids_are_not_lost_between_documents_of_the_notebook(self):
        """Чанки берутся по всем документам блокнота, а не по первому."""
        notebook = await self.make_notebook()
        first = await self.make_document("первый.txt", notebook.id)
        second = await self.make_document("второй.txt", notebook.id)
        first_chunks = await self.make_chunks(first, 2)
        second_chunks = await self.make_chunks(second, 2)
        self.fail_chroma()

        await self.delete_notebook(notebook.id)

        payload = json.loads((await self.cleanup_jobs())[0].payload_json)
        self.assertEqual(
            sorted(payload["chunk_ids"], key=int),
            sorted(
                [str(chunk.id) for chunk in first_chunks + second_chunks], key=int
            ),
        )

    async def test_cleanup_reports_failed_when_job_cannot_be_scheduled(self):
        """Лежит и ChromaDB, и очередь: ответ честно говорит failed."""
        notebook, _ = await self.notebook_with_chunks(documents=1)
        self.fail_chroma()

        with patch.object(
            JobsService, "enqueue", side_effect=RuntimeError("database is down")
        ):
            response = await self.delete_notebook(notebook.id)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["vector_cleanup"], "failed")
        self.assertFalse(await self.exists(Notebook, notebook.id))
        self.assertEqual(await self.cleanup_jobs(), [])

    async def test_failed_scheduling_logs_orphan_ids_as_last_resort(self):
        """Последний рубеж: id должны остаться хотя бы в логе."""
        notebook, chunk_ids = await self.notebook_with_chunks(
            documents=1, chunks_per_document=2
        )
        self.fail_chroma()

        with patch.object(JobsService, "enqueue", side_effect=RuntimeError("down")):
            with self.assertLogs("app.api.endpoints.notebooks", level="ERROR") as logs:
                await self.delete_notebook(notebook.id)

        recorded = "\n".join(logs.output)
        for chunk_id in chunk_ids:
            self.assertIn(chunk_id, recorded)


# --- Воркер дочищает ----------------------------------------------------


class DeferredCleanupWorkerTests(VectorCleanupTestCase):
    """Задачу забирает и обрабатывает настоящий код воркера.

    Цикл воркера (IndexingWorker._run) не запускается: он крутится бесконечно
    и в тесте его пришлось бы гасить по таймауту. Вызывается один проход —
    _claim_and_process/_claim_and_process_cleanup, то есть тот же захват
    задачи и тот же обработчик, что в бою.
    """

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.worker = IndexingWorker()

        # Воркер живёт вне запроса и берёт сессии сам — подставляем тестовые.
        session_patcher = patch(
            "app.modules.jobs.worker.session_context", self._worker_session
        )
        session_patcher.start()
        self.addCleanup(session_patcher.stop)

        # У воркера свой импорт RAGService, отдельный от того, что зовёт
        # эндпоинт: к моменту повтора ChromaDB может уже отвечать.
        worker_rag_patcher = patch("app.modules.jobs.worker.RAGService")
        self.worker_rag_cls = worker_rag_patcher.start()
        self.addCleanup(worker_rag_patcher.stop)
        self.worker_rag_delete = self.worker_rag_cls.return_value.delete_documents

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

    async def _defer_cleanup(self, **kwargs) -> tuple[Notebook, list[str]]:
        """Удалить блокнот при лежащей ChromaDB и получить отложенную задачу."""
        notebook, chunk_ids = await self.notebook_with_chunks(**kwargs)
        self.rag_delete.side_effect = RuntimeError("ChromaDB is down")
        response = await self.delete_notebook(notebook.id)
        self.assertEqual(response.json()["vector_cleanup"], "deferred")
        return notebook, chunk_ids

    async def test_worker_completes_deferred_cleanup_end_to_end(self):
        _, chunk_ids = await self._defer_cleanup(documents=2, chunks_per_document=3)

        processed = await self.worker._claim_and_process_cleanup()

        self.assertTrue(processed, "воркер не забрал задачу очистки")
        self.worker_rag_delete.assert_called_once()
        self.assertEqual(
            sorted(self.worker_rag_delete.call_args.args[0], key=int),
            sorted(chunk_ids, key=int),
        )
        job = (await self.cleanup_jobs())[0]
        self.assertEqual(job.status, "completed")
        self.assertEqual(job.progress, 100)
        self.assertEqual(json.loads(job.result_json)["deleted_chunks"], len(chunk_ids))
        self.assertIsNotNone(job.finished_at)

    async def test_cleanup_is_claimed_before_indexing(self):
        """Висячие векторы видны в поиске, поэтому очистка идёт вперёд."""
        await self._defer_cleanup(documents=1)
        # Отдельный документ с задачей индексации — он не должен быть взят.
        document = await self.make_document("ожидающий.txt", None)
        index_job = await self.seed(
            Job(job_type=JOB_INDEX_DOCUMENT, status="queued",
                source_id=document.id, created_by=self.user.id)
        )

        processed = await self.worker._claim_and_process()

        self.assertTrue(processed)
        self.assertEqual((await self.cleanup_jobs())[0].status, "completed")
        self.assertEqual(
            (await self.get_row(Job, index_job.id)).status, "queued",
            "задача индексации не должна была быть захвачена",
        )

    async def test_worker_requeues_cleanup_while_chroma_is_still_down(self):
        await self._defer_cleanup(documents=1, chunks_per_document=2)
        self.worker_rag_delete.side_effect = RuntimeError("ChromaDB is still down")

        processed = await self.worker._claim_and_process_cleanup()

        self.assertTrue(processed)
        job = (await self.cleanup_jobs())[0]
        self.assertEqual(job.status, "queued", "задача потеряна после отказа")
        self.assertEqual(job.attempt_count, 1)
        self.assertIn("ChromaDB is still down", job.error_text)
        # Пауза перед следующей попыткой: иначе лежащую ChromaDB долбили бы в
        # цикле и бюджет попыток сгорел бы за секунды.
        self.assertGreater(self.worker._cleanup_retry_after, 0.0)

    async def test_paused_worker_does_not_claim_cleanup_again(self):
        await self._defer_cleanup(documents=1)
        self.worker_rag_delete.side_effect = RuntimeError("still down")
        await self.worker._claim_and_process_cleanup()
        self.worker_rag_delete.reset_mock()

        processed = await self.worker._claim_and_process_cleanup()

        self.assertFalse(processed)
        self.worker_rag_delete.assert_not_called()
        self.assertEqual((await self.cleanup_jobs())[0].attempt_count, 1)

    async def test_exhausted_attempts_close_cleanup_as_failed(self):
        await self._defer_cleanup(documents=1)
        job_id = (await self.cleanup_jobs())[0].id
        # Бюджет попыток у очистки свой и большой (CLEANUP_MAX_ATTEMPTS):
        # изображаем задачу, которая его уже исчерпала.
        async with self.session_factory() as session:
            stored = await session.get(Job, job_id)
            stored.attempt_count = CLEANUP_MAX_ATTEMPTS - 1
            session.add(stored)
            await session.commit()
        self.worker_rag_delete.side_effect = RuntimeError("down forever")

        await self.worker._claim_and_process_cleanup()

        job = await self.get_row(Job, job_id)
        self.assertEqual(job.status, "failed")
        self.assertEqual(job.attempt_count, CLEANUP_MAX_ATTEMPTS)

    async def test_unusable_payload_is_not_retried_forever(self):
        await self._defer_cleanup(documents=1)
        job_id = (await self.cleanup_jobs())[0].id
        async with self.session_factory() as session:
            stored = await session.get(Job, job_id)
            stored.payload_json = "не json"
            session.add(stored)
            await session.commit()

        await self.worker._claim_and_process_cleanup()

        job = await self.get_row(Job, job_id)
        self.assertEqual(job.status, "failed")
        self.assertIn("payload", job.error_text.lower())
        self.worker_rag_delete.assert_not_called()

    async def test_worker_reports_nothing_to_do_without_cleanup_jobs(self):
        processed = await self.worker._claim_and_process_cleanup()

        self.assertFalse(processed)
        self.worker_rag_delete.assert_not_called()


# --- Осмысленность проверок ---------------------------------------------


class DeferredCleanupIsLoadBearingTests(VectorCleanupTestCase):
    """Тесты выше не должны «зеленеть» на любом поведении.

    Подменяется только поведение мока ChromaDB — файлы app/** не трогаются.
    """

    async def test_deferred_and_done_are_actually_different_outcomes(self):
        notebook_ok, _ = await self.notebook_with_chunks(documents=1)
        response_ok = await self.delete_notebook(notebook_ok.id)

        notebook_bad, chunk_ids = await self.notebook_with_chunks(documents=1)
        self.rag_delete.side_effect = RuntimeError("ChromaDB is down")
        response_bad = await self.delete_notebook(notebook_bad.id)

        self.assertEqual(response_ok.json()["vector_cleanup"], "done")
        self.assertEqual(response_bad.json()["vector_cleanup"], "deferred")
        # Задача появилась ровно от второго удаления и только от него.
        jobs = await self.cleanup_jobs()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(
            sorted(json.loads(jobs[0].payload_json)["chunk_ids"], key=int),
            sorted(chunk_ids, key=int),
        )

    async def test_ids_in_job_are_the_ids_chroma_never_received(self):
        """Задача несёт именно те id, на которых ChromaDB отказала."""
        notebook, chunk_ids = await self.notebook_with_chunks(
            documents=2, chunks_per_document=2
        )
        attempted: list[list[str]] = []

        def explode(ids):
            attempted.append(list(ids))
            raise RuntimeError("ChromaDB is down")

        self.rag_delete.side_effect = explode

        await self.delete_notebook(notebook.id)

        self.assertEqual(len(attempted), 1)
        payload = json.loads((await self.cleanup_jobs())[0].payload_json)
        self.assertEqual(
            sorted(payload["chunk_ids"], key=int), sorted(attempted[0], key=int)
        )
        self.assertEqual(sorted(attempted[0], key=int), sorted(chunk_ids, key=int))


if __name__ == "__main__":
    unittest.main()
