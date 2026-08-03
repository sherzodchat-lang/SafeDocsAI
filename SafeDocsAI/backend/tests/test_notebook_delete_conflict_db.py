"""Отказы 409 при удалении блокнота — на настоящем PostgreSQL.

У DELETE /api/v1/notebooks/{id} появились два конфликта (коды в
app/core/exceptions.py, SourceErrors):

  * source.notebook_busy_indexing — в блокноте есть задача в статусе
    'running'. Прервать её нечем: воркер живёт в другом процессе, канала
    отмены у очереди нет, а удалить документ у него из-под рук значит уронить
    индексацию на IntegrityError. Ключевая деталь: 409 отдаётся ТОЛЬКО на
    'running'. Задача в 'queued' удаляется вместе с блокнотом — её никто ещё
    не начал, и запирать удаление из-за неё было бы отказом на пустом месте;
  * source.notebook_delete_conflict — StaleDataError на commit: строка
    блокнота исчезла между нашим чтением и удалением.

Отдельный файл, а не дополнение к test_notebook_delete_db.py: там проверяется
порядок DELETE относительно внешних ключей, здесь — отказы и блокировки.
Блокировки требуют настоящей БД вдвойне: SELECT ... FOR UPDATE и
FOR UPDATE SKIP LOCKED в словаре вместо таблицы не существуют.

Про конкурентное удаление важно понимать границы проверяемого. Штатный путь
гонки — не 409, а 404: второй запрос ждёт на SELECT ... FOR UPDATE
(deps.get_owned_notebook_for_update) и после commit первого не находит строки
вовсе. Именно ради этого блокировка и добавлена. StaleDataError, а с ним и
409, остаётся страховкой на случай, когда блокировки не было или её обошли, —
детерминированно из двух параллельных HTTP-запросов он не воспроизводится
(см. ConcurrentDeleteTests). Поэтому 409 проверяется прицельно, на честно
устроенной ситуации «строка исчезла до commit», а параллельные запросы — на
то, что они не дают 500 и не оставляют мусора.
"""

import asyncio
import os
import sys
import unittest
from unittest.mock import patch

from fastapi import Depends, Path
from sqlmodel import or_, select

# Каталог с тестами в sys.path: unittest discover кладёт его туда сам, но при
# запуске `python -m unittest tests.test_notebook_delete_conflict_db` — нет.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dbfixtures import DatabaseBackedTestCase  # noqa: E402

from app.api import deps  # noqa: E402
from app.core.exceptions import ApiError, SourceErrors  # noqa: E402
from app.main import app  # noqa: E402
from app.modules.jobs.service import JOB_INDEX_DOCUMENT, JobsService  # noqa: E402
from app.shared.models import Chunk, Document, Job, Notebook, User  # noqa: E402


class NotebookDeleteConflictTestCase(DatabaseBackedTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()

        self.user = await self.make_user("owner", "user")
        self.as_user(self.user)

        rag_patcher = patch("app.modules.rag.service.RAGService")
        self.rag_cls = rag_patcher.start()
        self.addCleanup(rag_patcher.stop)
        self.rag_delete = self.rag_cls.return_value.delete_documents

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

    async def make_job(
        self,
        status: str,
        *,
        source_id: int | None = None,
        notebook_id: int | None = None,
    ) -> Job:
        return await self.seed(
            Job(job_type=JOB_INDEX_DOCUMENT, status=status, source_id=source_id,
                notebook_id=notebook_id, created_by=self.user.id)
        )

    async def notebook_with_source(
        self, name: str = "Блокнот"
    ) -> tuple[Notebook, Document, Chunk]:
        notebook = await self.make_notebook(name)
        document = await self.make_document("source.txt", notebook.id)
        chunk = await self.seed(
            Chunk(text="фрагмент", page=1, chunk_index=0,
                  embedding_id="emb-1", doc_id=document.id)
        )
        return notebook, document, chunk

    async def delete_notebook(self, notebook_id: int):
        return await self.client.delete(f"/api/v1/notebooks/{notebook_id}")

    def assertConflict(self, response, error_code: str) -> None:
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json().get("error_code"), error_code)


# --- Идёт индексация ----------------------------------------------------


class NotebookBusyIndexingTests(NotebookDeleteConflictTestCase):
    async def test_running_job_blocks_delete_with_409(self):
        notebook, document, chunk = await self.notebook_with_source()
        job = await self.make_job(
            "running", source_id=document.id, notebook_id=notebook.id
        )

        response = await self.delete_notebook(notebook.id)

        self.assertConflict(response, SourceErrors.NOTEBOOK_BUSY_INDEXING)
        # Ничего не тронуто: ни строк, ни файла, ни векторов.
        self.assertTrue(await self.exists(Notebook, notebook.id))
        self.assertTrue(await self.exists(Document, document.id))
        self.assertTrue(await self.exists(Chunk, chunk.id))
        self.assertTrue(await self.exists(Job, job.id))
        self.assertTrue(os.path.exists(document.path))
        self.rag_delete.assert_not_called()

    async def test_running_job_of_attached_document_blocks_delete(self):
        """У задачи notebook_id пустой — она находится по source_id.

        Так выглядит источник, загруженный вне блокнота и прикреплённый
        позже: фильтр только по Job.notebook_id её не увидел бы, и удаление
        снесло бы документ из-под работающего воркера.
        """
        notebook = await self.make_notebook()
        document = await self.make_document("attached.txt", notebook.id)
        await self.make_job("running", source_id=document.id, notebook_id=None)

        response = await self.delete_notebook(notebook.id)

        self.assertConflict(response, SourceErrors.NOTEBOOK_BUSY_INDEXING)
        self.assertTrue(await self.exists(Notebook, notebook.id))

    async def test_running_job_without_source_blocks_by_notebook_id(self):
        notebook = await self.make_notebook()
        await self.make_job("running", notebook_id=notebook.id)

        response = await self.delete_notebook(notebook.id)

        self.assertConflict(response, SourceErrors.NOTEBOOK_BUSY_INDEXING)

    async def test_queued_job_does_not_block_and_is_deleted(self):
        """Задача, которую никто не начал, удаляется вместе с блокнотом."""
        notebook, document, _ = await self.notebook_with_source()
        job = await self.make_job(
            "queued", source_id=document.id, notebook_id=notebook.id
        )

        response = await self.delete_notebook(notebook.id)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(await self.exists(Notebook, notebook.id))
        self.assertFalse(await self.exists(Job, job.id), "queued-задача осталась")
        self.assertFalse(await self.exists(Document, document.id))

    async def test_finished_jobs_do_not_block_delete(self):
        for status in ("completed", "failed"):
            with self.subTest(status=status):
                notebook, document, _ = await self.notebook_with_source(
                    f"Блокнот {status}"
                )
                await self.make_job(
                    status, source_id=document.id, notebook_id=notebook.id
                )

                response = await self.delete_notebook(notebook.id)

                self.assertEqual(response.status_code, 200, response.text)
                self.assertFalse(await self.exists(Notebook, notebook.id))

    async def test_running_job_of_another_notebook_does_not_block(self):
        target, _, _ = await self.notebook_with_source("Удаляемый")
        neighbour = await self.make_notebook("Соседний")
        neighbour_doc = await self.make_document("neighbour.txt", neighbour.id)
        busy = await self.make_job(
            "running", source_id=neighbour_doc.id, notebook_id=neighbour.id
        )

        response = await self.delete_notebook(target.id)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(await self.exists(Notebook, target.id))
        self.assertTrue(await self.exists(Job, busy.id))
        self.assertTrue(await self.exists(Notebook, neighbour.id))

    async def test_refusal_is_repeatable_until_job_finishes(self):
        """409 — «повторите позже», а не приговор: он снимается сам."""
        notebook, document, _ = await self.notebook_with_source()
        job = await self.make_job(
            "running", source_id=document.id, notebook_id=notebook.id
        )
        self.assertConflict(
            await self.delete_notebook(notebook.id),
            SourceErrors.NOTEBOOK_BUSY_INDEXING,
        )

        async with self.session_factory() as session:
            await JobsService.finish(session, job.id, result={"chunks": 1})

        response = await self.delete_notebook(notebook.id)
        self.assertEqual(response.status_code, 200, response.text)


# --- Гонка удалений -----------------------------------------------------


class StaleDataConflictTests(NotebookDeleteConflictTestCase):
    """409 на StaleDataError: где он достижим, а где нет.

    Ситуация одна и та же — строку блокнота уносят между чтением и commit.
    Чтобы она стала возможной, подменяется только зависимость (снимается
    FOR UPDATE); код обработчиков не трогается. В бою эту гонку блокировка и
    предотвращает, ветка 409 — страховка на случай, когда блокировки нет.

    Дальше начинается разница, которую видно только на настоящей БД:

      * PATCH шлёт UPDATE, и SQLAlchemy на «ожидали 1 строку, изменили 0»
        поднимает StaleDataError — ветка 409 срабатывает;
      * DELETE шлёт DELETE, а на нём та же несостыковка по умолчанию только
        предупреждает (SAWarning, mapper без version_id_col — см.
        orm/persistence._emit_delete_statements, ветка only_warn).
        Исключения нет, обработчик доходит до конца и отвечает 200.

    Поэтому тесты фиксируют фактическое поведение: 409 проверяется там, где
    он действительно возникает, а для удаления закреплено главное — гонка не
    превращается в 500.
    """

    def _override_dependency_without_lock(self, deleted_by_rival: bool) -> None:
        """Отдать обработчику блокнот, взятый БЕЗ FOR UPDATE.

        Подменяется зависимость, а не код обработчика: без блокировки строку
        успевает унести соперник, и обработчик доходит до commit с объектом,
        которого в БД уже нет, — ровно тот StaleDataError, ради которого
        добавлена ветка 409.
        """
        session_factory = self.session_factory

        async def dependency(
            notebook_id: int = Path(..., ge=1, le=deps.MAX_ID),
            session=Depends(deps.get_session),
            current_user: User = Depends(deps.get_current_user),
        ) -> Notebook:
            result = await session.exec(
                select(Notebook).where(Notebook.id == notebook_id)
            )
            notebook = result.first()
            if not notebook or not deps.user_owns(notebook.owner_id, current_user):
                raise ApiError(
                    404, SourceErrors.NOTEBOOK_NOT_FOUND, "Notebook not found"
                )
            if deleted_by_rival:
                # Соперник успевает удалить строку, пока она не заперта.
                async with session_factory() as rival:
                    stored = await rival.get(Notebook, notebook_id)
                    if stored is not None:
                        await rival.delete(stored)
                        await rival.commit()
            return notebook

        app.dependency_overrides[deps.get_owned_notebook_for_update] = dependency

    async def test_patch_of_vanished_notebook_gives_409_conflict(self):
        """Код source.notebook_delete_conflict живой — на UPDATE."""
        notebook = await self.make_notebook("Исчезнет до commit")
        self._override_dependency_without_lock(deleted_by_rival=True)

        response = await self.client.patch(
            f"/api/v1/notebooks/{notebook.id}", json={"name": "Новое имя"}
        )

        self.assertConflict(response, SourceErrors.NOTEBOOK_DELETE_CONFLICT)
        self.assertFalse(await self.exists(Notebook, notebook.id))

    async def test_patch_succeeds_when_nobody_intervenes(self):
        """Контроль: без соперника тот же путь даёт обычные 200."""
        notebook = await self.make_notebook("Никто не мешает")
        self._override_dependency_without_lock(deleted_by_rival=False)

        response = await self.client.patch(
            f"/api/v1/notebooks/{notebook.id}", json={"name": "Новое имя"}
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["name"], "Новое имя")

    async def test_delete_of_vanished_notebook_does_not_blow_up(self):
        """Гонка на удалении не даёт 500 — и не даёт 409.

        Ветка except StaleDataError в delete_notebook на этом пути не
        срабатывает: ORM-удаление несуществующей строки только предупреждает.
        Тест закрепляет фактическое поведение, чтобы правка в любую сторону
        (500 или осмысленный 409) была замечена.
        """
        notebook = await self.make_notebook("Исчезнет до commit")
        self._override_dependency_without_lock(deleted_by_rival=True)

        response = await self.delete_notebook(notebook.id)

        self.assertNotEqual(response.status_code, 500, response.text)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(await self.exists(Notebook, notebook.id))

    async def test_delete_succeeds_when_nobody_intervenes(self):
        notebook = await self.make_notebook("Никто не мешает")
        self._override_dependency_without_lock(deleted_by_rival=False)

        response = await self.delete_notebook(notebook.id)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(await self.exists(Notebook, notebook.id))


class ConcurrentDeleteTests(NotebookDeleteConflictTestCase):
    """Два одновременных DELETE одного блокнота.

    Проверяется не конкретный код второго ответа, а инвариант: ровно один
    запрос отчитывается об удалении, второй получает штатный отказ (404 —
    строки уже нет; 409 — если StaleDataError всё же случился), 500 не
    появляется ни при каких обстоятельствах, и блокнот со всем содержимым
    исчезает целиком. Жёстко требовать 404 нельзя: это результат гонки, а не
    контракт.
    """

    async def test_parallel_deletes_never_produce_500(self):
        notebook, document, chunk = await self.notebook_with_source()
        await self.make_job(
            "completed", source_id=document.id, notebook_id=notebook.id
        )

        first, second = await asyncio.gather(
            self.delete_notebook(notebook.id),
            self.delete_notebook(notebook.id),
        )

        statuses = sorted([first.status_code, second.status_code])
        self.assertNotIn(500, statuses, f"{first.text} / {second.text}")
        self.assertEqual(
            statuses.count(200), 1,
            f"об удалении должен отчитаться ровно один запрос: {statuses}",
        )
        loser = [r for r in (first, second) if r.status_code != 200][0]
        self.assertIn(loser.status_code, (404, 409), loser.text)
        if loser.status_code == 409:
            self.assertEqual(
                loser.json().get("error_code"),
                SourceErrors.NOTEBOOK_DELETE_CONFLICT,
            )

        self.assertFalse(await self.exists(Notebook, notebook.id))
        self.assertFalse(await self.exists(Document, document.id))
        self.assertFalse(await self.exists(Chunk, chunk.id))

    async def test_parallel_deletes_clean_up_exactly_once(self):
        """Побочная очистка не должна выполняться дважды."""
        notebook, _, _ = await self.notebook_with_source()

        await asyncio.gather(
            self.delete_notebook(notebook.id),
            self.delete_notebook(notebook.id),
        )

        self.assertEqual(
            self.rag_delete.call_count, 1,
            "векторы удалены не один раз — второй запрос дошёл до очистки",
        )


# --- Блокировка и воркер ------------------------------------------------


class LockDoesNotStallWorkerTests(NotebookDeleteConflictTestCase):
    """Удаление запирает задачи FOR UPDATE, воркер берёт их SKIP LOCKED.

    Если бы claim_next ждал на запертой строке, удаление блокнота вешало бы
    индексацию всех остальных документов до конца своей транзакции.
    """

    async def _lock_jobs(self, session, notebook_id: int, doc_ids: list[int]):
        """Повторяет запрос из delete_notebook, шаг «незавершённые задачи»."""
        job_filter = Job.notebook_id == notebook_id
        if doc_ids:
            job_filter = or_(job_filter, Job.source_id.in_(doc_ids))
        result = await session.exec(
            select(Job.id, Job.status)
            .where(job_filter)
            .where(Job.status.in_(("queued", "running")))
            .with_for_update()
        )
        return result.all()

    async def test_worker_skips_locked_queued_job_instead_of_waiting(self):
        notebook = await self.make_notebook()
        document = await self.make_document("indexing.txt", notebook.id)
        locked_job = await self.make_job(
            "queued", source_id=document.id, notebook_id=notebook.id
        )

        async with self.session_factory() as locker:
            rows = await self._lock_jobs(locker, notebook.id, [document.id])
            self.assertEqual(len(rows), 1, "предусловие: задача должна быть заперта")

            async with self.session_factory() as worker_session:
                claimed = await asyncio.wait_for(
                    JobsService.claim_next(worker_session, JOB_INDEX_DOCUMENT),
                    timeout=5,
                )

            self.assertIsNone(
                claimed, "воркер забрал задачу, которую удаление уже заперло"
            )
            # Статус не тронут: SKIP LOCKED именно пропускает строку.
            self.assertEqual((await self.get_row(Job, locked_job.id)).status, "queued")
            await locker.rollback()

    async def test_worker_takes_the_job_once_the_lock_is_released(self):
        notebook = await self.make_notebook()
        document = await self.make_document("indexing.txt", notebook.id)
        job = await self.make_job(
            "queued", source_id=document.id, notebook_id=notebook.id
        )

        async with self.session_factory() as locker:
            await self._lock_jobs(locker, notebook.id, [document.id])
            await locker.rollback()

        async with self.session_factory() as worker_session:
            claimed = await JobsService.claim_next(worker_session, JOB_INDEX_DOCUMENT)

        self.assertIsNotNone(claimed, "после снятия блокировки задача не взята")
        self.assertEqual(claimed.id, job.id)
        self.assertEqual(claimed.status, "running")

    async def test_locked_job_of_another_notebook_stays_claimable(self):
        """Запирается только то, что попало в фильтр удаляемого блокнота."""
        target = await self.make_notebook("Удаляемый")
        target_doc = await self.make_document("target.txt", target.id)
        await self.make_job("queued", source_id=target_doc.id, notebook_id=target.id)

        neighbour = await self.make_notebook("Соседний")
        neighbour_doc = await self.make_document("neighbour.txt", neighbour.id)
        free_job = await self.make_job(
            "queued", source_id=neighbour_doc.id, notebook_id=neighbour.id
        )

        async with self.session_factory() as locker:
            await self._lock_jobs(locker, target.id, [target_doc.id])

            async with self.session_factory() as worker_session:
                claimed = await JobsService.claim_next(
                    worker_session, JOB_INDEX_DOCUMENT
                )

            self.assertIsNotNone(claimed, "чужая задача заперта зря")
            self.assertEqual(claimed.id, free_job.id)
            await locker.rollback()


if __name__ == "__main__":
    unittest.main()
