"""Регрессия на DELETE /api/v1/notebooks/{id} — на настоящем PostgreSQL.

Зачем отдельный файл, если удаление блокнота уже проверяет
tests/test_ownership.py (NotebookOwnershipTests.test_delete_own_notebook_succeeds).
Тот тест работает на самодельной in-memory сессии (FakeAsyncSession) — словаре
объектов, в котором внешних ключей не существует в принципе. Поэтому он
зелёный и на сломанном коде: любой порядок DELETE там одинаково «успешен».

Ловим конкретный дефект:

    asyncpg.exceptions.ForeignKeyViolationError: update or delete on table
    "document" violates foreign key constraint "job_source_id_fkey" on table
    "job". DETAIL: Key (id)=(1) is still referenced from table "job".

Две причины:
  1. документы удалялись раньше задач, а job.source_id -> document.id объявлен
     без ON DELETE (confdeltype = 'a', NO ACTION);
  2. задачи выбирались только по Job.notebook_id, поэтому задача документа,
     прикреплённого к блокноту позже (attach) или перенесённого из другого
     блокнота, в выборку не попадала вовсе.

Отсюда требование к окружению: тест обязан идти по настоящей схеме с
включёнными внешними ключами. Схему создаёт код проекта (app.core.database
.init_db) в отдельной базе — рабочая andozai_db не трогается. Первым делом
идут тесты SchemaSanityTests: если job_source_id_fkey в тестовой схеме нет
или внешние ключи не применяются, остальные проверки бессмысленны, и это
должно быть видно явно, а не выглядеть как «всё прошло».

Тестовая база не создаётся автоматически (у andozai_user нет CREATEDB):

    sudo -u postgres psql -c "CREATE DATABASE andozai_test OWNER andozai_user;"

Имя перекрывается переменной окружения SAFEDOCS_TEST_DB.
"""

import asyncio
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

import app.core.database as database_module
from app.api import deps
from app.core.database import get_session
from app.main import app
from app.shared.models import Chunk, Document, Insight, Job, Log, Note, Notebook, User
from app.shared.settings import settings


# IsolatedAsyncioTestCase поднимает цикл событий в debug-режиме, а
# app.core.logging в development выставляет корневой уровень DEBUG: вдвоём они
# засыпают вывод трассировкой каждого соединения asyncpg, и результат прогона
# в ней тонет.
logging.getLogger("asyncio").setLevel(logging.WARNING)


# --- Тестовая база ------------------------------------------------------

TEST_DB_NAME = os.environ.get("SAFEDOCS_TEST_DB", "andozai_test")

# Страховка от опечатки в переменной окружения: тест удаляет и обрезает
# таблицы, и запускать его по рабочей базе нельзя ни при каких условиях.
if TEST_DB_NAME == settings.POSTGRES_DB:
    raise RuntimeError(
        f"SAFEDOCS_TEST_DB={TEST_DB_NAME!r} совпадает с рабочей базой "
        f"POSTGRES_DB={settings.POSTGRES_DB!r}. Заведите отдельную базу."
    )

TEST_DATABASE_URI = (
    f"postgresql+asyncpg://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
    f"@{settings.POSTGRES_SERVER}:{settings.POSTGRES_PORT}/{TEST_DB_NAME}"
)

# Порядок не важен: TRUNCATE перечисленных разом снимает вопрос ссылок между
# ними, CASCADE подхватывает всё, что могло появиться в схеме позже.
_TABLES = (
    "chunk",
    "job",
    "insight",
    "note",
    "log",
    "document",
    "notebook",
    "refreshtoken",
    '"user"',
)

_SETUP_ERROR: str | None = None
_SCHEMA_READY = False


def _new_engine():
    # NullPool обязателен: IsolatedAsyncioTestCase заводит новый цикл событий
    # на каждый тест, а соединение asyncpg привязано к циклу, в котором
    # открыто. С обычным пулом второй тест получил бы соединение из закрытого
    # цикла.
    return create_async_engine(TEST_DATABASE_URI, poolclass=NullPool, future=True)


async def _ensure_schema(engine) -> None:
    """Схему создаёт код проекта, а не CREATE TABLE в тесте.

    init_db() работает с модульным engine приложения — подменяем его на
    тестовый на время вызова. Так тест идёт по той же схеме и тем же
    миграциям, что и рабочая база, включая индексы и ALTER.
    """
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    with patch.object(database_module, "engine", engine):
        await database_module.init_db()
    _SCHEMA_READY = True


async def _truncate(engine) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(f"TRUNCATE TABLE {', '.join(_TABLES)} RESTART IDENTITY CASCADE")
        )


async def _probe() -> str | None:
    engine = _new_engine()
    try:
        await _ensure_schema(engine)
        await _truncate(engine)
    except Exception as exc:  # pragma: no cover - зависит от окружения
        return (
            f"Тестовая база {TEST_DB_NAME!r} недоступна ({exc.__class__.__name__}: "
            f"{exc}). Создайте её: sudo -u postgres psql -c "
            f'"CREATE DATABASE {TEST_DB_NAME} OWNER {settings.POSTGRES_USER};"'
        )
    finally:
        await engine.dispose()
    return None


_SETUP_ERROR = asyncio.run(_probe())


# --- Общая обвязка ------------------------------------------------------


class NotebookDeleteDbTestCase(unittest.IsolatedAsyncioTestCase):
    """Настоящая сессия к PostgreSQL, подставленная в зависимости FastAPI."""

    async def asyncSetUp(self) -> None:
        if _SETUP_ERROR:
            self.skipTest(_SETUP_ERROR)

        self.engine = _new_engine()
        self.addAsyncCleanup(self.engine.dispose)
        await _ensure_schema(self.engine)
        await _truncate(self.engine)

        self.session_factory = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

        self.user = await self._seed(
            User(username="owner", password_hash="not-a-real-hash", role="user")
        )

        # ChromaDB в тесте не поднимаем: интересен сам факт и момент вызова.
        rag_patcher = patch("app.modules.rag.service.RAGService")
        self.rag_cls = rag_patcher.start()
        self.addCleanup(rag_patcher.stop)
        self.rag_delete = self.rag_cls.return_value.delete_documents

        app.dependency_overrides[get_session] = self._session_dependency
        app.dependency_overrides[deps.get_current_user] = self._current_user_dependency
        self.addCleanup(app.dependency_overrides.clear)

        # raise_app_exceptions=True: если регрессия вернулась, тест упадёт с
        # настоящим ForeignKeyViolationError в трейсбеке, а не с безликим 500.
        self.client = AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        )
        self.addAsyncCleanup(self.client.aclose)

    # --- зависимости ---

    async def _session_dependency(self):
        # Повторяет app.core.database.session_context: откат на исключении
        # важен именно здесь — на нём держится проверка отката.
        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def _current_user_dependency(self) -> User:
        return self.user

    # --- работа с данными ---

    async def _seed(self, *rows):
        async with self.session_factory() as session:
            for row in rows:
                session.add(row)
            await session.commit()
            for row in rows:
                await session.refresh(row)
        return rows[0] if len(rows) == 1 else rows

    async def _exists(self, model, pk) -> bool:
        async with self.session_factory() as session:
            return await session.get(model, pk) is not None

    async def _count_where(self, model, clause) -> int:
        async with self.session_factory() as session:
            result = await session.exec(select(model).where(clause))
            return len(result.all())

    def _make_file(self, name: str) -> str:
        path = Path(self._tmpdir.name) / name
        path.write_text("содержимое источника", encoding="utf-8")
        return str(path)

    async def _make_notebook(self, name: str, owner: User | None = None) -> Notebook:
        owner = owner or self.user
        return await self._seed(
            Notebook(name=name, description=None, domain_profile="general",
                     owner_id=owner.id)
        )

    async def _make_document(self, name: str, notebook_id: int | None) -> Document:
        return await self._seed(
            Document(
                name=name,
                path=self._make_file(name),
                size=42,
                notebook_id=notebook_id,
                owner_id=self.user.id,
            )
        )

    async def _assert_notebook_fully_gone(self, notebook_id: int, doc_ids: list[int]):
        """Ни одной осиротевшей строки ни в одной связанной таблице."""
        self.assertFalse(await self._exists(Notebook, notebook_id), "notebook остался")
        for doc_id in doc_ids:
            self.assertFalse(
                await self._exists(Document, doc_id), f"document {doc_id} остался"
            )
            self.assertEqual(
                await self._count_where(Chunk, Chunk.doc_id == doc_id), 0,
                f"chunk документа {doc_id} остался",
            )
            self.assertEqual(
                await self._count_where(Job, Job.source_id == doc_id), 0,
                f"job по source_id={doc_id} остался",
            )
        for model in (Note, Insight, Log, Job):
            self.assertEqual(
                await self._count_where(model, model.notebook_id == notebook_id), 0,
                f"{model.__name__} блокнота {notebook_id} остался",
            )


# --- Осмысленность окружения -------------------------------------------


class SchemaSanityTests(NotebookDeleteDbTestCase):
    """Без этих двух проверок весь файл может «зеленеть» вхолостую."""

    async def test_job_source_id_fkey_exists(self):
        async with self.engine.connect() as conn:
            result = await conn.execute(
                text(
                    """
                    SELECT c.conname, src.relname, tgt.relname, c.confdeltype
                    FROM pg_constraint c
                    JOIN pg_class src ON src.oid = c.conrelid
                    JOIN pg_class tgt ON tgt.oid = c.confrelid
                    WHERE c.contype = 'f' AND c.conname = 'job_source_id_fkey'
                      -- Имя ограничения уникально внутри схемы, а не в базе:
                      -- в той же базе живёт схема других тестов
                      -- (tests/dbfixtures.py) с такой же копией схемы проекта.
                      -- Без этого условия проверка считала бы чужие строки.
                      AND c.connamespace = current_schema()::regnamespace
                    """
                )
            )
            rows = result.all()

        self.assertEqual(
            len(rows), 1,
            "В тестовой схеме нет внешнего ключа job_source_id_fkey — "
            "воспроизвести регрессию невозможно, остальные тесты этого файла "
            "ничего не доказывают.",
        )
        conname, src_table, tgt_table, _confdeltype = rows[0]
        self.assertEqual((src_table, tgt_table), ("job", "document"))

    async def test_foreign_keys_are_actually_enforced(self):
        """Ключ мало объявить — БД должна его применять (не NOT VALID и т.п.)."""
        with self.assertRaises(IntegrityError):
            async with self.session_factory() as session:
                session.add(Job(job_type="index_document", source_id=10_000_019))
                await session.commit()


# --- Сама регрессия -----------------------------------------------------


class DeleteNotebookWithSourceTests(NotebookDeleteDbTestCase):
    async def test_delete_notebook_with_job_bound_to_same_notebook(self):
        """Сценарий 1: обычная загрузка — у задачи notebook_id тот же."""
        notebook = await self._make_notebook("Блокнот с источником")
        document = await self._make_document("source.txt", notebook.id)
        await self._seed(
            Chunk(text="фрагмент", page=1, chunk_index=0,
                  embedding_id="emb-1", doc_id=document.id),
            Job(job_type="index_document", status="done",
                source_id=document.id, notebook_id=notebook.id,
                created_by=self.user.id),
            Note(title="Заметка", body="текст", notebook_id=notebook.id,
                 created_by=self.user.id),
            Insight(title="Инсайт", body="текст", notebook_id=notebook.id,
                    created_by=self.user.id),
            Log(question="Вопрос", answer="Ответ", time_ms=1,
                user_id=self.user.id, notebook_id=notebook.id),
        )

        response = await self.client.delete(f"/api/v1/notebooks/{notebook.id}")

        self.assertEqual(response.status_code, 200, response.text)
        await self._assert_notebook_fully_gone(notebook.id, [document.id])

    async def test_delete_notebook_with_job_attached_later(self):
        """Сценарий 2: источник загружен вне блокнота и прикреплён потом.

        У задачи notebook_id так и остался NULL. Выборка задач только по
        Job.notebook_id её не находит, и DELETE документа упирается в
        job_source_id_fkey.
        """
        notebook = await self._make_notebook("Блокнот, куда прикрепили позже")
        document = await self._make_document("attached.txt", notebook_id=None)
        job = await self._seed(
            Job(job_type="index_document", status="done",
                source_id=document.id, notebook_id=None,
                created_by=self.user.id)
        )
        # Сам attach: документ переезжает в блокнот, задача не трогается.
        async with self.session_factory() as session:
            stored = await session.get(Document, document.id)
            stored.notebook_id = notebook.id
            session.add(stored)
            await session.commit()
        self.assertIsNone(
            (await self._get_job(job.id)).notebook_id,
            "предусловие: у задачи notebook_id должен остаться NULL",
        )

        response = await self.client.delete(f"/api/v1/notebooks/{notebook.id}")

        self.assertEqual(response.status_code, 200, response.text)
        await self._assert_notebook_fully_gone(notebook.id, [document.id])
        self.assertFalse(await self._exists(Job, job.id), "job с NULL notebook_id остался")

    async def test_delete_notebook_after_document_moved_from_another(self):
        """Сценарий 3: документ перенесён из блокнота A в B, удаляем B.

        У задачи notebook_id указывает на A — фильтр по Job.notebook_id == B
        её не выбирает.
        """
        notebook_a = await self._make_notebook("Блокнот A")
        notebook_b = await self._make_notebook("Блокнот B")
        document = await self._make_document("moved.txt", notebook_a.id)
        job = await self._seed(
            Job(job_type="index_document", status="done",
                source_id=document.id, notebook_id=notebook_a.id,
                created_by=self.user.id)
        )
        async with self.session_factory() as session:
            stored = await session.get(Document, document.id)
            stored.notebook_id = notebook_b.id
            session.add(stored)
            await session.commit()

        response = await self.client.delete(f"/api/v1/notebooks/{notebook_b.id}")

        self.assertEqual(response.status_code, 200, response.text)
        await self._assert_notebook_fully_gone(notebook_b.id, [document.id])
        self.assertFalse(await self._exists(Job, job.id), "job чужого блокнота остался")
        # Блокнот A удалять не просили.
        self.assertTrue(await self._exists(Notebook, notebook_a.id))

    async def test_deleting_notebook_leaves_foreign_notebook_data_intact(self):
        """Чужой блокнот с собственным документом и задачей не задет."""
        target = await self._make_notebook("Удаляемый")
        target_doc = await self._make_document("target.txt", target.id)
        await self._seed(
            Job(job_type="index_document", source_id=target_doc.id,
                notebook_id=target.id, created_by=self.user.id)
        )
        keeper = await self._make_notebook("Оставляемый")
        keeper_doc = await self._make_document("keeper.txt", keeper.id)
        keeper_job = await self._seed(
            Job(job_type="index_document", source_id=keeper_doc.id,
                notebook_id=keeper.id, created_by=self.user.id)
        )

        response = await self.client.delete(f"/api/v1/notebooks/{target.id}")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(await self._exists(Notebook, keeper.id))
        self.assertTrue(await self._exists(Document, keeper_doc.id))
        self.assertTrue(await self._exists(Job, keeper_job.id))

    async def _get_job(self, job_id: int) -> Job:
        async with self.session_factory() as session:
            return await session.get(Job, job_id)


# --- Побочные эффекты: только после commit ------------------------------


class DeleteNotebookSideEffectsTests(NotebookDeleteDbTestCase):
    """Второе требование фикса: файл и векторы уничтожаются только после
    успешного commit. При откате транзакции блокнот остаётся в БД, и удалённые
    заранее файл с эмбеддингами восстановить было бы нечем."""

    async def test_successful_delete_removes_file_and_vectors(self):
        notebook = await self._make_notebook("С побочными эффектами")
        document = await self._make_document("payload.txt", notebook.id)
        await self._seed(
            Chunk(text="фрагмент", page=1, chunk_index=0,
                  embedding_id="emb-1", doc_id=document.id),
            Job(job_type="index_document", source_id=document.id,
                notebook_id=None, created_by=self.user.id),
        )
        self.assertTrue(os.path.exists(document.path))

        response = await self.client.delete(f"/api/v1/notebooks/{notebook.id}")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(os.path.exists(document.path), "файл не удалён после commit")
        self.rag_delete.assert_called_once()

    async def test_failed_delete_keeps_file_and_vectors(self):
        """Транзакцию рушит реальная ссылка, а не мок.

        insight.note_id -> note.id: инсайт соседнего блокнота ссылается на
        заметку удаляемого. Заметки удаляются, ссылка остаётся — commit падает.
        Ожидание: 500, блокнот на месте, файл на диске цел, ChromaDB не тронута.
        """
        notebook = await self._make_notebook("Удаление сорвётся")
        document = await self._make_document("survivor.txt", notebook.id)
        await self._seed(
            Chunk(text="фрагмент", page=1, chunk_index=0,
                  embedding_id="emb-1", doc_id=document.id),
            Job(job_type="index_document", source_id=document.id,
                notebook_id=notebook.id, created_by=self.user.id),
        )
        note = await self._seed(
            Note(title="Заметка удаляемого", notebook_id=notebook.id,
                 created_by=self.user.id)
        )
        neighbour = await self._make_notebook("Соседний блокнот")
        await self._seed(
            Insight(title="Ссылается на чужую заметку", notebook_id=neighbour.id,
                    note_id=note.id, created_by=self.user.id)
        )

        with self.assertRaises(IntegrityError):
            await self.client.delete(f"/api/v1/notebooks/{notebook.id}")

        self.assertTrue(
            await self._exists(Notebook, notebook.id),
            "предусловие проверки: транзакция должна была откатиться",
        )
        self.assertTrue(await self._exists(Document, document.id))
        self.assertTrue(
            os.path.exists(document.path),
            "файл источника удалён, хотя блокнот остался в БД",
        )
        self.rag_delete.assert_not_called()


if __name__ == "__main__":
    unittest.main()
