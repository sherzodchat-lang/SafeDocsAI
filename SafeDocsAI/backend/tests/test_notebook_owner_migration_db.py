"""Владелец блокнота обязателен: бэкфилл и NOT NULL в init_db.

Раньше notebook.owner_id мог быть NULL: колонка владельца появилась позже
первых блокнотов, и такой блокнот считался «legacy — доступен только админу».
Особый случай жил в коде проверки владения (deps.user_owns) и всплывал в
каждом аудите. Решение — убрать его из кода в схему: старые строки получают
владельца, а колонка становится NOT NULL, чтобы состояние не вернулось.

Почему настоящий PostgreSQL. Проверяется не поведение функции, а сама схема:
запрет держит база, а не приложение, и увидеть это можно только в
information_schema живого сервера. Схему, как и в соседних тестах, создаёт код
проекта (app.core.database.init_db) в отдельной базе — см. tests/dbfixtures.py.

Бэкфилл проверяется честно: NOT NULL с колонки временно снимается, строка без
владельца заводится напрямую в базе, после чего init_db прогоняется ещё раз.
Иначе такое состояние на живой схеме не воспроизвести — его же и запрещает
проверяемое ограничение.
"""

import os
import sys
import unittest
from unittest.mock import patch

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

# Каталог с тестами в sys.path: unittest discover кладёт его туда сам, но при
# запуске `python -m unittest tests.test_notebook_owner_migration_db` этого не
# происходит.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dbfixtures import TEST_SCHEMA, DatabaseBackedTestCase  # noqa: E402

import app.core.database as database_module  # noqa: E402


class NotebookOwnerMigrationTestCase(DatabaseBackedTestCase):
    """Общие помощники: повторный прогон init_db и снятие NOT NULL."""

    async def run_init_db(self) -> None:
        """Прогнать миграции проекта по тестовой схеме ещё раз.

        init_db выполняется при каждом старте приложения, поэтому повторный
        прогон — не искусственный сценарий, а ровно то, что делает боевой
        деплой.
        """
        with patch.object(database_module, "engine", self.engine):
            await database_module.init_db()

    async def execute(self, sql: str, **params) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(text(sql), params)

    async def fetch(self, sql: str, **params):
        async with self.engine.begin() as conn:
            return (await conn.execute(text(sql), params)).all()

    async def scalar(self, sql: str, **params):
        async with self.engine.begin() as conn:
            return (await conn.execute(text(sql), params)).scalar_one()

    async def owner_id_is_nullable(self) -> bool:
        value = await self.scalar(
            """
            SELECT is_nullable FROM information_schema.columns
            WHERE table_schema = :schema
              AND table_name = 'notebook'
              AND column_name = 'owner_id'
            """,
            schema=TEST_SCHEMA,
        )
        return value == "YES"

    async def allow_null_owner(self) -> None:
        """Временно вернуть колонку в состояние «до миграции».

        Ограничение восстанавливается на выходе из теста: схема одна на весь
        прогон процесса, и оставленная nullable колонка испортила бы
        соседние проверки.
        """
        await self.execute(
            "ALTER TABLE notebook ALTER COLUMN owner_id DROP NOT NULL"
        )
        self.addAsyncCleanup(self._restore_not_null)

    async def _restore_not_null(self) -> None:
        await self.execute("TRUNCATE TABLE notebook CASCADE")
        await self.execute("ALTER TABLE notebook ALTER COLUMN owner_id SET NOT NULL")

    async def insert_ownerless_notebook(self, name: str = "Legacy notebook") -> int:
        """Блокнот без владельца — напрямую в базу.

        Через модель это уже невозможно: Notebook.owner_id объявлен
        обязательным. Смысл строки именно в том, чтобы повторить состояние
        старой базы, а не в том, чтобы обойти модель.
        """
        return await self.scalar(
            """
            INSERT INTO notebook (name, description, domain_profile, owner_id, created_at)
            VALUES (:name, NULL, 'general', NULL, NOW())
            RETURNING id
            """,
            name=name,
        )


class NotebookOwnerNotNullTests(NotebookOwnerMigrationTestCase):
    """Состояние схемы после штатного init_db."""

    async def test_owner_id_column_is_not_null(self):
        self.assertFalse(
            await self.owner_id_is_nullable(),
            "notebook.owner_id должен быть NOT NULL после init_db",
        )

    async def test_no_notebooks_without_owner_exist(self):
        user = await self.make_user("owner", "user")
        await self.execute(
            """
            INSERT INTO notebook (name, description, domain_profile, owner_id, created_at)
            VALUES ('Обычный', NULL, 'general', :owner, NOW())
            """,
            owner=user.id,
        )
        self.assertEqual(
            0, await self.scalar("SELECT COUNT(*) FROM notebook WHERE owner_id IS NULL")
        )

    async def test_database_rejects_notebook_without_owner(self):
        """Запрет держит база, а не приложение."""
        with self.assertRaises(IntegrityError):
            await self.insert_ownerless_notebook()


class NotebookOwnerBackfillTests(NotebookOwnerMigrationTestCase):
    """Что init_db делает со строками, оставшимися от старой схемы."""

    async def test_backfill_assigns_oldest_admin(self):
        await self.allow_null_owner()
        # Порядок создания задаёт id: первым заводится админ с меньшим id,
        # и именно он должен забрать блокноты — правило детерминированное,
        # а не «какой попадётся».
        first_admin = await self.make_user("root", "admin")
        second_admin = await self.make_user("another-root", "admin")
        self.assertLess(first_admin.id, second_admin.id)
        await self.make_user("bob", "user")

        notebook_id = await self.insert_ownerless_notebook()

        await self.run_init_db()

        owner_id = await self.scalar(
            "SELECT owner_id FROM notebook WHERE id = :id", id=notebook_id
        )
        self.assertEqual(first_admin.id, owner_id)
        self.assertFalse(
            await self.owner_id_is_nullable(),
            "после успешного бэкфилла колонка снова NOT NULL",
        )

    async def test_backfill_is_idempotent(self):
        """Второй прогон не трогает уже проставленного владельца."""
        await self.allow_null_owner()
        admin = await self.make_user("root", "admin")
        owner = await self.make_user("alice", "user")
        notebook_id = await self.scalar(
            """
            INSERT INTO notebook (name, description, domain_profile, owner_id, created_at)
            VALUES ('Чей-то', NULL, 'general', :owner, NOW())
            RETURNING id
            """,
            owner=owner.id,
        )

        await self.run_init_db()
        await self.run_init_db()

        owner_id = await self.scalar(
            "SELECT owner_id FROM notebook WHERE id = :id", id=notebook_id
        )
        self.assertEqual(owner.id, owner_id)
        self.assertNotEqual(admin.id, owner_id)

    async def test_without_admin_not_null_is_postponed(self):
        """Назначить владельца некому — старт не падает, шаг откладывается.

        Падать нельзя: приложение, которое не поднимается без админа, не даёт
        админа и завести. Поведение при этом не меняется — блокнот без
        владельца и так доступен только админу, которого нет.
        """
        await self.allow_null_owner()
        await self.make_user("bob", "user")
        notebook_id = await self.insert_ownerless_notebook()

        with self.assertLogs("app.core.database", level="WARNING") as captured:
            await self.run_init_db()
        self.assertTrue(
            any("notebook.owner_id" in message for message in captured.output),
            captured.output,
        )

        owner_id = await self.scalar(
            "SELECT owner_id FROM notebook WHERE id = :id", id=notebook_id
        )
        self.assertIsNone(owner_id)
        self.assertTrue(
            await self.owner_id_is_nullable(),
            "без бэкфилла ограничение ставить нельзя: старт упал бы на живой базе",
        )

        # Появился админ — следующий же старт доводит миграцию до конца.
        admin = await self.make_user("root", "admin")
        await self.run_init_db()

        owner_id = await self.scalar(
            "SELECT owner_id FROM notebook WHERE id = :id", id=notebook_id
        )
        self.assertEqual(admin.id, owner_id)
        self.assertFalse(await self.owner_id_is_nullable())

    async def test_placeholder_notebook_gets_an_owner(self):
        """Блокнот-заглушка для источников без notebook_id — тоже с владельцем.

        Раньше init_db заводил его с owner_id = NULL, то есть сам создавал то
        состояние, которое теперь запрещено.
        """
        await self.execute("TRUNCATE TABLE notebook CASCADE")
        admin = await self.make_user("root", "admin")

        await self.run_init_db()

        rows = await self.fetch("SELECT owner_id FROM notebook")
        self.assertEqual(1, len(rows))
        self.assertEqual(admin.id, rows[0][0])

    async def test_placeholder_notebook_is_skipped_without_admin(self):
        """Без админа заглушку создавать нечем и незачем."""
        await self.execute("TRUNCATE TABLE notebook CASCADE")

        await self.run_init_db()

        self.assertEqual(0, await self.scalar("SELECT COUNT(*) FROM notebook"))


if __name__ == "__main__":
    unittest.main()
