"""Удаление блокнота с презентациями — на настоящем PostgreSQL.

Ловим тот же класс дефекта, что и tests/test_notebook_delete_db.py, только на
новой таблице:

    asyncpg.exceptions.ForeignKeyViolationError: update or delete on table
    "notebook" violates foreign key constraint "presentation_notebook_id_fkey"
    on table "presentation".

`presentation.notebook_id` объявлен NOT NULL и БЕЗ `ON DELETE` (confdeltype
'a', NO ACTION), поэтому блокнот, у которого есть хоть одна строка заказа,
удалить нельзя, пока эти строки не убраны той же транзакцией и раньше него.
Существующие тесты удаления этого не ловили просто потому, что презентаций не
создавали, — а цена такого пропуска в этом проекте уже известна: при отказе на
внешнем ключе транзакция откатывается, но побочная очистка, если её сделать до
commit, уже уничтожила файлы безвозвратно (docs/audit-2026-08.md).

Проверяется договор владельца целиком:

  * 'generating' -> 409, и ни одна строка, ни один файл не тронуты. Довод тот
    же, что у 'running'-индексации: генерация идёт в другом процессе, канала
    отмены нет;
  * 'queued' удаляется вместе с блокнотом — файла у него ещё нет;
  * у 'ready' (и у 'error', если файл от прошлого прогона остался) путь
    собирается ДО удаления строк, строки уходят раньше блокнота в той же
    транзакции, а файл стирается ПОСЛЕ commit;
  * сбой os.remove не превращается в 500: блокнота в БД уже нет, и отказ на
    файле означает только сироту на диске, про которую написано в журнале.

Первым идёт SchemaSanityTests: если внешнего ключа в тестовой схеме нет,
остальные проверки этого файла ничего не доказывают.
"""

import os
import sys
import unittest
from unittest.mock import patch

from sqlalchemy import text
from sqlmodel import select

# Каталог с тестами в sys.path: unittest discover кладёт его туда сам, но при
# запуске `python -m unittest tests.test_notebook_delete_presentations_db` — нет.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dbfixtures import DatabaseBackedTestCase  # noqa: E402

from app.core.exceptions import SourceErrors  # noqa: E402
from app.modules.presentations.constants import (  # noqa: E402
    STATUS_ERROR,
    STATUS_GENERATING,
    STATUS_QUEUED,
    STATUS_READY,
)
from app.modules.presentations.service import PresentationsService  # noqa: E402
from app.shared.models import Document, Notebook, Presentation, User  # noqa: E402


NOTEBOOKS_LOGGER = "app.api.endpoints.notebooks"

# Код, которым эндпоинт сейчас отвечает на «в блокноте идёт генерация».
# Своего кода у этого отказа пока нет: exceptions.py правится вместе со
# словарями фронта, и до тех пор берётся ближайший существующий — тот же 409
# «блокнот занят, повторите позже». Тест ссылается на константу, а не на
# строку, поэтому появление собственного кода правится здесь одной строкой.
BUSY_GENERATING = SourceErrors.NOTEBOOK_BUSY_GENERATING


class NotebookDeletePresentationsTestCase(DatabaseBackedTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()

        self.user: User = await self.make_user("owner", "user")
        self.as_user(self.user)

        # ChromaDB в тесте не поднимаем: интересен сам факт и момент вызова.
        rag_patcher = patch("app.modules.rag.service.RAGService")
        self.rag_cls = rag_patcher.start()
        self.addCleanup(rag_patcher.stop)
        self.rag_delete = self.rag_cls.return_value.delete_documents

    # --- данные ---

    async def make_notebook(self, name: str = "Блокнот") -> Notebook:
        return await self.seed(
            Notebook(name=name, description=None, domain_profile="general",
                     owner_id=self.user.id)
        )

    async def make_document(self, name: str, notebook_id: int) -> Document:
        return await self.seed(
            Document(name=name, path=self.make_file(name), size=42,
                     notebook_id=notebook_id, owner_id=self.user.id)
        )

    async def make_presentation(
        self,
        notebook: Notebook,
        *,
        status: str = STATUS_READY,
        with_file: bool = True,
        **overrides,
    ) -> Presentation:
        """Заказ в нужном статусе; у готового рядом лежит настоящий файл.

        Файл кладётся во временный каталог теста, а не в data/presentations:
        проверяется, что удаляется ровно то, что записано в file_path.
        """
        fields = {
            "notebook_id": notebook.id,
            "owner_id": self.user.id,
            "template_key": "classic",
            "language": "ru",
            "slide_count": 5,
            "status": status,
        }
        if with_file:
            fields["file_path"] = self.make_file(
                f"presentation_{notebook.id}_{status}.pptx", "PK\x03\x04"
            )
            fields["file_size"] = 4
        fields.update(overrides)
        return await self.seed(Presentation(**fields))

    async def delete_notebook(self, notebook_id: int):
        return await self.client.delete(f"/api/v1/notebooks/{notebook_id}")

    async def presentations_of(self, notebook_id: int) -> list[Presentation]:
        return await self.rows_where(
            Presentation, Presentation.notebook_id == notebook_id
        )


# --- Осмысленность окружения -------------------------------------------


class SchemaSanityTests(NotebookDeletePresentationsTestCase):
    async def test_presentation_notebook_fkey_exists_without_on_delete(self):
        """Без этого ключа весь файл «зеленел» бы вхолостую."""
        async with self.engine.connect() as conn:
            result = await conn.execute(
                text(
                    """
                    SELECT c.conname, src.relname, tgt.relname, c.confdeltype
                    FROM pg_constraint c
                    JOIN pg_class src ON src.oid = c.conrelid
                    JOIN pg_class tgt ON tgt.oid = c.confrelid
                    WHERE c.contype = 'f'
                      AND c.conname = 'presentation_notebook_id_fkey'
                      -- Имя ограничения уникально внутри схемы, а не в базе:
                      -- рядом живут схемы других прогонов с такой же копией
                      -- схемы проекта.
                      AND c.connamespace = current_schema()::regnamespace
                    """
                )
            )
            rows = result.all()

        self.assertEqual(
            len(rows), 1,
            "В тестовой схеме нет внешнего ключа presentation_notebook_id_fkey — "
            "воспроизвести регрессию невозможно, остальные тесты этого файла "
            "ничего не доказывают.",
        )
        _conname, src_table, tgt_table, confdeltype = rows[0]
        self.assertEqual((src_table, tgt_table), ("presentation", "notebook"))
        # pg_constraint.confdeltype — тип "char", asyncpg отдаёт его байтами.
        if isinstance(confdeltype, bytes):
            confdeltype = confdeltype.decode()
        self.assertEqual(
            confdeltype, "a",
            "у ключа появился ON DELETE — порядок удаления перестал быть "
            "единственной защитой, и комментарии в notebooks.py устарели",
        )


# --- Сама регрессия -----------------------------------------------------


class DeleteNotebookWithPresentationsTests(NotebookDeletePresentationsTestCase):
    async def test_delete_notebook_with_ready_presentation_succeeds(self):
        """Ключевой тест: до правки здесь падал presentation_notebook_id_fkey.

        Строка заказа удаляется раньше блокнота в той же транзакции, а файл
        готовой колоды стирается уже после commit.
        """
        notebook = await self.make_notebook("Блокнот с готовой колодой")
        document = await self.make_document("source.txt", notebook.id)
        presentation = await self.make_presentation(notebook, status=STATUS_READY)
        self.assertTrue(os.path.exists(presentation.file_path))

        response = await self.delete_notebook(notebook.id)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(await self.exists(Notebook, notebook.id))
        self.assertFalse(await self.exists(Document, document.id))
        self.assertFalse(
            await self.exists(Presentation, presentation.id),
            "строка presentation осталась осиротевшей",
        )
        self.assertFalse(
            os.path.exists(presentation.file_path),
            "файл готовой презентации не удалён после commit",
        )

    async def test_queued_presentation_is_deleted_with_the_notebook(self):
        """Заказ, который никто не начал: файла нет, удалять с диска нечего."""
        notebook = await self.make_notebook("Блокнот с заказом в очереди")
        presentation = await self.make_presentation(
            notebook, status=STATUS_QUEUED, with_file=False
        )

        response = await self.delete_notebook(notebook.id)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(await self.exists(Notebook, notebook.id))
        self.assertFalse(
            await self.exists(Presentation, presentation.id),
            "queued-заказ остался",
        )

    async def test_failed_presentation_with_leftover_file_is_cleaned_up(self):
        """У 'error' файл мог остаться от прошлого прогона — стираем и его.

        Решение намеренно опирается на колонку file_path, а не на статус:
        заполненный путь и есть запись о существовании файла.
        """
        notebook = await self.make_notebook("Блокнот с неудачной колодой")
        presentation = await self.make_presentation(notebook, status=STATUS_ERROR)

        response = await self.delete_notebook(notebook.id)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(await self.exists(Presentation, presentation.id))
        self.assertFalse(
            os.path.exists(presentation.file_path),
            "файл презентации в статусе error остался на диске",
        )

    async def test_all_presentations_of_the_notebook_go_at_once(self):
        """Заказов у блокнота обычно несколько, и уйти обязаны все."""
        notebook = await self.make_notebook("Много колод")
        ready = await self.make_presentation(notebook, status=STATUS_READY)
        second = await self.make_presentation(
            notebook, status=STATUS_READY, slide_count=7
        )
        queued = await self.make_presentation(
            notebook, status=STATUS_QUEUED, with_file=False
        )

        response = await self.delete_notebook(notebook.id)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(await self.presentations_of(notebook.id), [])
        for presentation in (ready, second):
            self.assertFalse(os.path.exists(presentation.file_path))
        self.assertFalse(await self.exists(Presentation, queued.id))

    async def test_presentations_of_another_notebook_are_untouched(self):
        """Чужой заказ и его файл не задеты."""
        target = await self.make_notebook("Удаляемый")
        await self.make_presentation(target, status=STATUS_READY)
        keeper = await self.make_notebook("Оставляемый")
        keeper_deck = await self.make_presentation(keeper, status=STATUS_READY)

        response = await self.delete_notebook(target.id)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(await self.exists(Notebook, keeper.id))
        self.assertTrue(await self.exists(Presentation, keeper_deck.id))
        self.assertTrue(
            os.path.exists(keeper_deck.file_path),
            "удалён файл презентации соседнего блокнота",
        )

    async def test_file_removal_failure_does_not_become_500(self):
        """os.remove упал (права, read-only монтирование) — это не отказ.

        Блокнота в БД уже нет: превращать сироту на диске в 500 значило бы
        врать клиенту про несделанную работу. Единственное требование — путь
        должен остаться в журнале, чтобы файл можно было убрать руками.
        """
        notebook = await self.make_notebook("Файл не стереть")
        presentation = await self.make_presentation(notebook, status=STATUS_READY)
        blocked = presentation.file_path
        real_remove = os.remove

        def refuse(path, *args, **kwargs):
            if path == blocked:
                raise PermissionError(13, "Permission denied")
            return real_remove(path, *args, **kwargs)

        with patch(f"{NOTEBOOKS_LOGGER}.os.remove", side_effect=refuse):
            with self.assertLogs(NOTEBOOKS_LOGGER, level="WARNING") as logs:
                response = await self.delete_notebook(notebook.id)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(await self.exists(Notebook, notebook.id))
        self.assertFalse(await self.exists(Presentation, presentation.id))
        recorded = "\n".join(logs.output)
        self.assertIn(blocked, recorded)
        self.assertIn(str(presentation.id), recorded)
        self.assertTrue(os.path.exists(blocked), "файл всё-таки исчез")


# --- Идёт генерация -----------------------------------------------------


class NotebookBusyGeneratingTests(NotebookDeletePresentationsTestCase):
    async def test_generating_presentation_blocks_delete_with_409(self):
        notebook = await self.make_notebook("Идёт генерация")
        document = await self.make_document("source.txt", notebook.id)
        presentation = await self.make_presentation(
            notebook, status=STATUS_GENERATING, with_file=False
        )

        response = await self.delete_notebook(notebook.id)

        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json().get("error_code"), BUSY_GENERATING)
        # Ничего не тронуто: ни строк, ни файлов, ни векторов.
        self.assertTrue(await self.exists(Notebook, notebook.id))
        self.assertTrue(await self.exists(Document, document.id))
        self.assertTrue(await self.exists(Presentation, presentation.id))
        self.assertTrue(os.path.exists(document.path))
        self.rag_delete.assert_not_called()

    async def test_generating_presentation_of_another_notebook_does_not_block(self):
        target = await self.make_notebook("Удаляемый")
        neighbour = await self.make_notebook("Соседний")
        busy = await self.make_presentation(
            neighbour, status=STATUS_GENERATING, with_file=False
        )

        response = await self.delete_notebook(target.id)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(await self.exists(Notebook, target.id))
        self.assertTrue(await self.exists(Presentation, busy.id))

    async def test_queued_and_ready_presentations_do_not_block(self):
        """409 отдаётся ТОЛЬКО на 'generating'."""
        for status in (STATUS_QUEUED, STATUS_READY, STATUS_ERROR):
            with self.subTest(status=status):
                notebook = await self.make_notebook(f"Блокнот {status}")
                await self.make_presentation(
                    notebook, status=status, with_file=status != STATUS_QUEUED
                )

                response = await self.delete_notebook(notebook.id)

                self.assertEqual(response.status_code, 200, response.text)
                self.assertFalse(await self.exists(Notebook, notebook.id))

    async def test_refusal_lifts_once_generation_finishes(self):
        """409 — «повторите позже», а не приговор.

        Строку в 'ready' переводит настоящий PresentationsService.mark_ready:
        так проверяется и то, что удаление снимает отказ, и то, что оно потом
        убирает записанный сервисом файл.
        """
        notebook = await self.make_notebook("Сначала занят")
        presentation = await self.make_presentation(
            notebook, status=STATUS_GENERATING, with_file=False
        )
        first = await self.delete_notebook(notebook.id)
        self.assertEqual(first.status_code, 409, first.text)

        file_path = self.make_file("presentation_done.pptx", "PK\x03\x04")
        async with self.session_factory() as session:
            await PresentationsService.mark_ready(
                session, presentation.id, file_path=file_path, file_size=4
            )

        response = await self.delete_notebook(notebook.id)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(await self.exists(Notebook, notebook.id))
        self.assertFalse(os.path.exists(file_path), "файл готовой колоды остался")


# --- Блокировка и воркер презентаций ------------------------------------


class LockDoesNotStallPresentationWorkerTests(NotebookDeletePresentationsTestCase):
    """Удаление запирает заказы FOR UPDATE, воркер берёт их SKIP LOCKED.

    Тот же приём, что с очередью индексации (tests/
    test_notebook_delete_conflict_db.py): если бы claim_next ждал на запертой
    строке, удаление блокнота вешало бы генерацию всех остальных колод до
    конца своей транзакции.
    """

    async def _lock_presentations(self, session, notebook_id: int):
        """Повторяет запрос из delete_notebook, шаг «презентации блокнота»."""
        result = await session.exec(
            select(Presentation.id, Presentation.status, Presentation.file_path)
            .where(Presentation.notebook_id == notebook_id)
            .with_for_update()
        )
        return result.all()

    async def test_worker_skips_locked_queued_order_instead_of_waiting(self):
        notebook = await self.make_notebook()
        queued = await self.make_presentation(
            notebook, status=STATUS_QUEUED, with_file=False
        )

        async with self.session_factory() as locker:
            rows = await self._lock_presentations(locker, notebook.id)
            self.assertEqual(len(rows), 1, "предусловие: заказ должен быть заперт")

            async with self.session_factory() as worker_session:
                claimed = await PresentationsService.claim_next(worker_session)

            self.assertIsNone(
                claimed, "воркер забрал заказ, который удаление уже заперло"
            )
            # Статус не тронут: SKIP LOCKED именно пропускает строку.
            self.assertEqual(
                (await self.get_row(Presentation, queued.id)).status, STATUS_QUEUED
            )
            await locker.rollback()

    async def test_worker_takes_the_order_once_the_lock_is_released(self):
        notebook = await self.make_notebook()
        queued = await self.make_presentation(
            notebook, status=STATUS_QUEUED, with_file=False
        )

        async with self.session_factory() as locker:
            await self._lock_presentations(locker, notebook.id)
            await locker.rollback()

        async with self.session_factory() as worker_session:
            claimed = await PresentationsService.claim_next(worker_session)

        self.assertEqual(claimed, queued.id)

    async def test_locked_order_of_another_notebook_stays_claimable(self):
        """Запирается только то, что попало в фильтр удаляемого блокнота."""
        target = await self.make_notebook("Удаляемый")
        await self.make_presentation(target, status=STATUS_QUEUED, with_file=False)
        neighbour = await self.make_notebook("Соседний")
        free = await self.make_presentation(
            neighbour, status=STATUS_QUEUED, with_file=False
        )

        async with self.session_factory() as locker:
            await self._lock_presentations(locker, target.id)

            async with self.session_factory() as worker_session:
                claimed = await PresentationsService.claim_next(worker_session)

            self.assertEqual(claimed, free.id, "чужой заказ заперт зря")
            await locker.rollback()


if __name__ == "__main__":
    unittest.main()
