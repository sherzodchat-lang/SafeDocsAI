"""Аварийный вход: возврат админа мимо API — на настоящем PostgreSQL.

Если админов в системе не осталось, вернуть их запросом невозможно по
устройству API: PUT /settings/users/{id}/role сам требует роль admin,
создания пользователя в API нет, а регистрация жёстко ставит role="user" и по
умолчанию выключена. Остаётся единственный путь — backend/create_admin.py с
доступом к БД, то есть тот же, которым админ заводится при развёртывании.

Раньше скрипт умел только создавать нового пользователя, а существующего
оставлял нетронутым — и на базе, где имена уже заняты (обычный случай:
админов разжаловали, пользователи остались), возвращать доступ было нечем.
Здесь проверяется оба поведения: прежнее по умолчанию и повышение по
ADMIN_PROMOTE=1.

Скрипт запускается с переменными окружения, а читает их в константы модуля на
импорте, поэтому в тестах подменяются именно константы (patch.object). Сессия
подменяется там же: create_admin держит ссылку на get_session у себя в
модуле, и подмена уводит скрипт в тестовую схему, не трогая рабочую базу.
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Каталог backend/ — там лежит сам скрипт.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dbfixtures import DatabaseBackedTestCase  # noqa: E402

import create_admin as create_admin_script  # noqa: E402

from app.core.security import verify_password  # noqa: E402
from app.shared.models import User  # noqa: E402


class BreakGlassTestCase(DatabaseBackedTestCase):
    def _test_session_factory(self):
        """Замена create_admin.get_session: та же форма, тестовая схема."""
        factory = self.session_factory

        async def get_session():
            async with factory() as session:
                yield session

        return get_session

    async def run_script(self, *, username: str, password: str = "s3cret-pass", promote: bool = False) -> str:
        output = io.StringIO()
        with patch.multiple(
            create_admin_script,
            ADMIN_USERNAME=username,
            ADMIN_PASSWORD=password,
            ADMIN_PROMOTE=promote,
            get_session=self._test_session_factory(),
        ):
            with redirect_stdout(output):
                await create_admin_script.create_admin()
        return output.getvalue()

    async def admins(self) -> list[User]:
        return await self.rows_where(User, User.role == "admin")


class NoAdminsLeftTests(BreakGlassTestCase):
    """В базе ноль админов — скрипт возвращает доступ."""

    async def test_promotes_an_existing_user_when_no_admin_is_left(self):
        chief = await self.make_user("chief", "user")
        await self.make_user("colleague", "user")
        self.assertEqual(await self.admins(), [], "предусловие: админов нет")

        output = await self.run_script(username="chief", promote=True)

        self.assertEqual(
            [user.username for user in await self.admins()],
            ["chief"],
            output,
        )
        self.assertEqual((await self.get_row(User, chief.id)).role, "admin")
        self.assertIn("user -> admin", output)

    async def test_promotion_does_not_touch_the_password(self):
        """Доступ возвращается тому, кто им владеет, а не подменяется."""
        chief = await self.make_user("chief", "user")
        stored_hash = (await self.get_row(User, chief.id)).password_hash

        output = await self.run_script(
            username="chief", password="совсем другой пароль", promote=True
        )

        self.assertEqual((await self.get_row(User, chief.id)).password_hash, stored_hash)
        self.assertIn("Password left unchanged", output)

    async def test_promoted_user_regains_the_admin_only_endpoint(self):
        """Главное: после скрипта админский эндпоинт снова отвечает."""
        chief = await self.make_user("chief", "user")

        self.as_user(await self.get_row(User, chief.id))
        forbidden = await self.client.get("/api/v1/settings/users")
        self.assertEqual(forbidden.status_code, 403, forbidden.text)

        await self.run_script(username="chief", promote=True)

        self.as_user(await self.get_row(User, chief.id))
        allowed = await self.client.get("/api/v1/settings/users")
        self.assertEqual(allowed.status_code, 200, allowed.text)

    async def test_creates_a_new_admin_when_the_name_is_free(self):
        """Прежнее поведение: имени в базе нет — заводится новый админ."""
        output = await self.run_script(username="rescue", password="Rescue-123")

        admins = await self.admins()
        self.assertEqual([user.username for user in admins], ["rescue"], output)
        self.assertTrue(verify_password("Rescue-123", admins[0].password_hash))
        self.assertIn("Created new admin user", output)

    async def test_new_admin_can_log_in_with_the_printed_password(self):
        """Пароль из вывода должен работать: это и есть возвращённый доступ."""
        await self.run_script(username="rescue", password="Rescue-123")

        response = await self.client.post(
            "/api/v1/auth/login",
            data={"username": "rescue", "password": "Rescue-123"},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("access_token", response.json())


class ExistingUserIsLeftAloneTests(BreakGlassTestCase):
    """Поведение обычного прогона деплоя не изменилось."""

    async def test_existing_user_is_untouched_without_promote(self):
        ordinary = await self.make_user("chief", "user")
        stored_hash = (await self.get_row(User, ordinary.id)).password_hash

        output = await self.run_script(username="chief", password="new-password")

        stored = await self.get_row(User, ordinary.id)
        self.assertEqual(stored.role, "user")
        self.assertEqual(stored.password_hash, stored_hash)
        self.assertIn("left untouched", output)
        # Подсказка в выводе — единственное место, где видно, что делать
        # дальше: скрипт запускают, когда всё уже сломалось.
        self.assertIn("ADMIN_PROMOTE=1", output)

    async def test_existing_admin_is_untouched_with_promote(self):
        admin = await self.make_user("chief", "admin")
        stored_hash = (await self.get_row(User, admin.id)).password_hash

        output = await self.run_script(
            username="chief", password="new-password", promote=True
        )

        stored = await self.get_row(User, admin.id)
        self.assertEqual(stored.role, "admin")
        self.assertEqual(stored.password_hash, stored_hash)
        self.assertIn("already an admin", output)

    async def test_promote_lifts_a_content_manager_too(self):
        manager = await self.make_user("manager", "content_manager")

        output = await self.run_script(username="manager", promote=True)

        self.assertEqual((await self.get_row(User, manager.id)).role, "admin")
        self.assertIn("content_manager -> admin", output)

    async def test_promotion_does_not_touch_anybody_else(self):
        await self.make_user("chief", "user")
        bystander = await self.make_user("bystander", "user")

        await self.run_script(username="chief", promote=True)

        self.assertEqual((await self.get_row(User, bystander.id)).role, "user")
        self.assertEqual(len(await self.admins()), 1)


if __name__ == "__main__":
    unittest.main()
