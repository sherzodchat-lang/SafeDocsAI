"""Владелец документа обязателен: бэкфилл и NOT NULL в init_db.

Тот же переход, что уже сделан для блокнотов (см.
tests/test_notebook_owner_migration_db.py). Раньше document.owner_id мог быть
NULL: колонка владельца появилась позже первых документов, и такой документ
считался «legacy — доступен только админу». Особый случай жил в коде проверки
владения (deps.user_owns) и всплывал в каждом аудите. Решение — убрать его из
кода в схему: старые строки получают владельца, а колонка становится NOT NULL,
чтобы состояние не вернулось.

Бэкфилл документов двухступенчатый, и обе ветки проверяются отдельно:
документ, лежащий в блокноте, наследует владельца блокнота (иначе он ушёл бы
админу, отобрав источник у настоящего хозяина), а документ вне блокнота
достаётся старейшему админу.

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
# запуске `python -m unittest tests.test_document_owner_migration_db` этого не
# происходит.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dbfixtures import TEST_SCHEMA, DatabaseBackedTestCase  # noqa: E402

import app.core.database as database_module  # noqa: E402


class DocumentOwnerMigrationTestCase(DatabaseBackedTestCase):
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
              AND table_name = 'document'
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
            "ALTER TABLE document ALTER COLUMN owner_id DROP NOT NULL"
        )
        self.addAsyncCleanup(self._restore_not_null)

    async def _restore_not_null(self) -> None:
        await self.execute("TRUNCATE TABLE document CASCADE")
        await self.execute("ALTER TABLE document ALTER COLUMN owner_id SET NOT NULL")

    async def make_notebook(self, name: str, owner_id: int | None) -> int:
        return await self.scalar(
            """
            INSERT INTO notebook (name, description, domain_profile, owner_id, created_at)
            VALUES (:name, NULL, 'general', :owner, NOW())
            RETURNING id
            """,
            name=name,
            owner=owner_id,
        )

    async def insert_ownerless_document(
        self, name: str = "legacy.txt", notebook_id: int | None = None
    ) -> int:
        """Документ без владельца — напрямую в базу.

        Через модель это уже невозможно: Document.owner_id объявлен
        обязательным. Смысл строки именно в том, чтобы повторить состояние
        старой базы, а не в том, чтобы обойти модель.
        """
        return await self.scalar(
            """
            INSERT INTO document (name, path, size, language, status,
                                  notebook_id, owner_id, created_at)
            VALUES (:name, :path, 42, 'ru', 'indexed', :notebook, NULL, NOW())
            RETURNING id
            """,
            name=name,
            path=self.make_file(name),
            notebook=notebook_id,
        )

    async def owner_of(self, document_id: int):
        return await self.scalar(
            "SELECT owner_id FROM document WHERE id = :id", id=document_id
        )


class DocumentOwnerNotNullTests(DocumentOwnerMigrationTestCase):
    """Состояние схемы после штатного init_db."""

    async def test_owner_id_column_is_not_null(self):
        self.assertFalse(
            await self.owner_id_is_nullable(),
            "document.owner_id должен быть NOT NULL после init_db",
        )

    async def test_no_documents_without_owner_exist(self):
        user = await self.make_user("owner", "user")
        notebook_id = await self.make_notebook("Свой блокнот", user.id)
        await self.execute(
            """
            INSERT INTO document (name, path, size, language, status,
                                  notebook_id, owner_id, created_at)
            VALUES ('обычный.txt', :path, 42, 'ru', 'indexed', :notebook, :owner, NOW())
            """,
            path=self.make_file("обычный.txt"),
            notebook=notebook_id,
            owner=user.id,
        )
        self.assertEqual(
            0, await self.scalar("SELECT COUNT(*) FROM document WHERE owner_id IS NULL")
        )

    async def test_database_rejects_document_without_owner(self):
        """Запрет держит база, а не приложение."""
        with self.assertRaises(IntegrityError):
            await self.insert_ownerless_document()


class DocumentOwnerBackfillTests(DocumentOwnerMigrationTestCase):
    """Что init_db делает со строками, оставшимися от старой схемы."""

    async def test_document_in_notebook_inherits_notebook_owner(self):
        """Первая ветка бэкфилла: владелец берётся у блокнота, а не у админа.

        Ветка важнее второй: свалить всё на админа означало бы отобрать
        источники у их настоящих хозяев.
        """
        await self.allow_null_owner()
        await self.make_user("root", "admin")
        owner = await self.make_user("alice", "user")
        notebook_id = await self.make_notebook("Блокнот Алисы", owner.id)

        document_id = await self.insert_ownerless_document(
            "alice.txt", notebook_id=notebook_id
        )

        await self.run_init_db()

        self.assertEqual(owner.id, await self.owner_of(document_id))
        self.assertFalse(
            await self.owner_id_is_nullable(),
            "после успешного бэкфилла колонка снова NOT NULL",
        )

    async def test_document_without_notebook_goes_to_oldest_admin(self):
        """Вторая ветка: наследовать не у кого — забирает старейший админ.

        Порядок создания задаёт id: первым заводится админ с меньшим id, и
        именно он должен забрать документы — правило детерминированное, а не
        «какой попадётся».
        """
        await self.allow_null_owner()
        first_admin = await self.make_user("root", "admin")
        second_admin = await self.make_user("another-root", "admin")
        self.assertLess(first_admin.id, second_admin.id)
        await self.make_user("bob", "user")

        document_id = await self.insert_ownerless_document()

        await self.run_init_db()

        self.assertEqual(first_admin.id, await self.owner_of(document_id))
        self.assertFalse(await self.owner_id_is_nullable())

    async def test_document_of_ownerless_notebook_follows_it_to_admin(self):
        """Блокнот тоже был без владельца: обоих забирает один и тот же админ.

        Бэкфилл блокнотов идёт раньше бэкфилла документов ровно ради этого
        случая — иначе документ достался бы админу «вторым путём», а блокнот
        первым, и совпадение владельцев было бы случайным.
        """
        await self.allow_null_owner()
        await self.execute("ALTER TABLE notebook ALTER COLUMN owner_id DROP NOT NULL")
        self.addAsyncCleanup(self._restore_notebook_not_null)
        admin = await self.make_user("root", "admin")
        notebook_id = await self.make_notebook("Ничей блокнот", None)

        document_id = await self.insert_ownerless_document(
            "orphan.txt", notebook_id=notebook_id
        )

        await self.run_init_db()

        notebook_owner = await self.scalar(
            "SELECT owner_id FROM notebook WHERE id = :id", id=notebook_id
        )
        self.assertEqual(admin.id, notebook_owner)
        self.assertEqual(admin.id, await self.owner_of(document_id))

    async def _restore_notebook_not_null(self) -> None:
        await self.execute("TRUNCATE TABLE notebook CASCADE")
        await self.execute("ALTER TABLE notebook ALTER COLUMN owner_id SET NOT NULL")

    async def test_backfill_is_idempotent(self):
        """Второй прогон не трогает уже проставленного владельца."""
        await self.allow_null_owner()
        admin = await self.make_user("root", "admin")
        owner = await self.make_user("alice", "user")
        notebook_id = await self.make_notebook("Блокнот Алисы", owner.id)
        document_id = await self.scalar(
            """
            INSERT INTO document (name, path, size, language, status,
                                  notebook_id, owner_id, created_at)
            VALUES ('чей-то.txt', :path, 42, 'ru', 'indexed', :notebook, :owner, NOW())
            RETURNING id
            """,
            path=self.make_file("чей-то.txt"),
            notebook=notebook_id,
            owner=owner.id,
        )

        await self.run_init_db()
        await self.run_init_db()

        self.assertEqual(owner.id, await self.owner_of(document_id))
        self.assertNotEqual(admin.id, await self.owner_of(document_id))
        self.assertFalse(await self.owner_id_is_nullable())

    async def test_without_admin_not_null_is_postponed(self):
        """Назначить владельца некому — старт не падает, шаг откладывается.

        Падать нельзя: приложение, которое не поднимается без админа, не даёт
        админа и завести. Поведение при этом не меняется — документ без
        владельца и так доступен только админу, которого нет.
        """
        await self.allow_null_owner()
        await self.make_user("bob", "user")
        document_id = await self.insert_ownerless_document()

        with self.assertLogs("app.core.database", level="WARNING") as captured:
            await self.run_init_db()
        self.assertTrue(
            any("document.owner_id" in message for message in captured.output),
            captured.output,
        )

        self.assertIsNone(await self.owner_of(document_id))
        self.assertTrue(
            await self.owner_id_is_nullable(),
            "без бэкфилла ограничение ставить нельзя: старт упал бы на живой базе",
        )

        # Появился админ — следующий же старт доводит миграцию до конца.
        admin = await self.make_user("root", "admin")
        await self.run_init_db()

        self.assertEqual(admin.id, await self.owner_of(document_id))
        self.assertFalse(await self.owner_id_is_nullable())


if __name__ == "__main__":
    unittest.main()
