"""Система не должна оставаться без администратора — на настоящем PostgreSQL.

PUT /api/v1/settings/users/{id}/role — единственный путь, которым роль admin
снимается, и единственный, которым она выдаётся. Ошибиться здесь дороже, чем
где бы то ни было: обратной дороги через API нет (эндпоинт сам требует роль
admin, создания пользователя в API нет, регистрация жёстко ставит role="user"
и по умолчанию выключена), а миграция владельцев в app/core/database.py без
админа молча пропускает бэкфилл и SET NOT NULL.

Прежняя проверка стояла под условием «понижаю сам себя» и чужую роль не
смотрела вовсе. Отсюда сценарий, которому даже не нужна гонка в БД: два
админа одновременно понижают друг друга, у обоих user.id != current_user.id,
проверка не выполняется ни разу — админов ноль. Ему посвящён
MutualDemotionTests, остальные классы закрывают то, что вокруг: отказы,
поколение токенов, журнал.

Настоящая БД нужна не для удобства: проверка идёт под
pg_advisory_xact_lock, а блокировки в словаре вместо таблицы не существует.
"""

import asyncio
import os
import sys
import unittest

from fastapi import Request
from sqlalchemy import text

# Каталог с тестами в sys.path: unittest discover кладёт его туда сам, но при
# запуске `python -m unittest tests.test_admin_role_guard_db` — нет.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dbfixtures import DatabaseBackedTestCase  # noqa: E402

from app.api import deps  # noqa: E402
from app.api.endpoints import settings as settings_endpoint  # noqa: E402
from app.core.exceptions import SettingsErrors  # noqa: E402
from app.main import app  # noqa: E402
from app.shared.models import User  # noqa: E402


LOGGER_NAME = "app.api.endpoints.settings"

# Сколько транзакций прямо сейчас стоят в очереди за блокировкой смены ролей.
#
# Ключ блокировки — OID таблицы "user" текущей схемы, поэтому соседний прогон
# тестов (своя схема в той же базе) в этот счёт не попадает. classid = 0 и
# objsubid = 1 — признаки односоставного bigint-ключа: так pg_locks раскладывает
# pg_advisory_xact_lock(bigint).
WAITING_ON_ROLE_LOCK = text(
    """
    SELECT count(*) FROM pg_locks
     WHERE locktype = 'advisory'
       AND NOT granted
       AND classid = 0
       AND objsubid = 1
       AND objid = '"user"'::regclass::oid
       AND database = (
           SELECT oid FROM pg_database WHERE datname = current_database()
       )
    """
)


class RoleChangeTestCase(DatabaseBackedTestCase):
    """Общая обвязка: несколько действующих лиц вместо одного.

    dbfixtures отдаёт эндпоинту одного и того же подставленного пользователя,
    а здесь нужны два одновременных запроса от разных админов. Поэтому автор
    запроса выбирается заголовком x-test-actor; без заголовка поведение
    прежнее (self.current_user).
    """

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()

        self.actors: dict[str, User] = {}

        async def current_user_dependency(request: Request) -> User:
            actor = request.headers.get("x-test-actor")
            if actor is not None:
                return self.actors[actor]
            return self.current_user

        app.dependency_overrides[deps.get_current_user] = current_user_dependency

    async def make_actor(self, username: str, role: str) -> User:
        user = await self.make_user(username, role)
        self.actors[username] = user
        return user

    async def set_role(self, actor: User, target: User, role: str):
        return await self.client.put(
            f"/api/v1/settings/users/{target.id}/role",
            json={"role": role},
            headers={"x-test-actor": actor.username},
        )

    async def admin_count(self) -> int:
        return len(await self.rows_where(User, User.role == "admin"))

    async def role_of(self, user: User) -> str:
        return (await self.get_row(User, user.id)).role

    def assertRefused(self, response, status_code: int, error_code: str) -> None:
        self.assertEqual(response.status_code, status_code, response.text)
        self.assertEqual(response.json().get("error_code"), error_code, response.text)


# --- Последний админ ----------------------------------------------------


class LastAdminTests(RoleChangeTestCase):
    async def test_last_admin_cannot_demote_himself(self):
        admin = await self.make_actor("chief", "admin")

        response = await self.set_role(admin, admin, "user")

        self.assertRefused(response, 400, SettingsErrors.LAST_ADMIN)
        self.assertEqual(await self.role_of(admin), "admin")
        self.assertEqual(await self.admin_count(), 1)

    async def test_last_admin_cannot_be_demoted_to_content_manager_either(self):
        """Отказ смотрит на роль admin, а не на конкретную новую роль."""
        admin = await self.make_actor("chief", "admin")

        response = await self.set_role(admin, admin, "content_manager")

        self.assertRefused(response, 400, SettingsErrors.LAST_ADMIN)
        self.assertEqual(await self.role_of(admin), "admin")

    async def test_last_admin_can_be_confirmed_as_admin(self):
        """Запрет — на снятие роли, а не на любой запрос к последнему админу."""
        admin = await self.make_actor("chief", "admin")

        response = await self.set_role(admin, admin, "admin")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(await self.role_of(admin), "admin")

    async def test_admin_may_demote_another_admin_while_one_remains(self):
        """Проверка не должна превратиться в запрет любых понижений."""
        first = await self.make_actor("first", "admin")
        second = await self.make_actor("second", "admin")

        response = await self.set_role(first, second, "user")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["role"], "user")
        self.assertEqual(await self.role_of(second), "user")
        self.assertEqual(await self.admin_count(), 1)

    async def test_admin_may_demote_himself_while_another_admin_remains(self):
        first = await self.make_actor("first", "admin")
        await self.make_actor("second", "admin")

        response = await self.set_role(first, first, "user")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(await self.role_of(first), "user")
        self.assertEqual(await self.admin_count(), 1)

    async def test_promotion_is_never_refused(self):
        admin = await self.make_actor("chief", "admin")
        ordinary = await self.make_user("newcomer", "user")

        response = await self.set_role(admin, ordinary, "admin")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(await self.admin_count(), 2)

    async def test_demoting_a_non_admin_is_not_touched_by_the_guard(self):
        admin = await self.make_actor("chief", "admin")
        manager = await self.make_user("manager", "content_manager")

        response = await self.set_role(admin, manager, "user")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(await self.role_of(manager), "user")
        self.assertEqual(await self.admin_count(), 1)

    async def test_unknown_user_gives_404_with_a_machine_code(self):
        admin = await self.make_actor("chief", "admin")

        response = await self.client.put(
            "/api/v1/settings/users/999999/role",
            json={"role": "user"},
            headers={"x-test-actor": admin.username},
        )

        self.assertRefused(response, 404, SettingsErrors.USER_NOT_FOUND)

    async def test_id_out_of_int32_range_is_rejected_by_validation(self):
        """Не 500: id вне диапазона PostgreSQL integer ронял бы asyncpg."""
        admin = await self.make_actor("chief", "admin")

        response = await self.client.put(
            f"/api/v1/settings/users/{deps.MAX_ID + 1}/role",
            json={"role": "user"},
            headers={"x-test-actor": admin.username},
        )

        self.assertEqual(response.status_code, 422, response.text)

    async def test_non_admin_cannot_change_roles(self):
        manager = await self.make_actor("manager", "content_manager")
        admin = await self.make_actor("chief", "admin")

        response = await self.set_role(manager, admin, "user")

        self.assertEqual(response.status_code, 403, response.text)
        self.assertEqual(await self.role_of(admin), "admin")


# --- Взаимное понижение -------------------------------------------------


class MutualDemotionTests(RoleChangeTestCase):
    """Ключевой сценарий дефекта: два админа снимают друг друга.

    Гонки в БД он не требует — прежняя проверка просто не выполнялась ни в
    одном из двух запросов, потому что ни один админ не понижал сам себя.
    Инвариант, который здесь закрепляется, один и он абсолютный: админов
    после любой пары запросов остаётся не меньше одного.
    """

    async def _wait_until_someone_waits_for_the_lock(
        self, session, timeout: float = 10.0
    ) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            waiting = (await session.exec(WAITING_ON_ROLE_LOCK)).scalar_one()
            if waiting:
                return
            if loop.time() > deadline:
                self.fail("запрос так и не встал на блокировку смены ролей")
            await asyncio.sleep(0.05)

    async def test_two_admins_demoting_each_other_leave_one(self):
        first = await self.make_actor("first", "admin")
        second = await self.make_actor("second", "admin")

        left, right = await asyncio.gather(
            self.set_role(first, second, "user"),
            self.set_role(second, first, "user"),
        )

        statuses = sorted([left.status_code, right.status_code])
        self.assertNotIn(500, statuses, f"{left.text} / {right.text}")
        self.assertEqual(
            statuses.count(200),
            1,
            f"пройти должно ровно одно понижение, а не {statuses}",
        )
        # Проигравший получает штатный отказ. Какой именно — зависит от того,
        # успел ли он посчитать админов до чужого commit: 409, если состояние
        # изменилось у него под руками, 400, если он с самого начала видел
        # единственного админа. Требовать конкретный код от гонки нельзя,
        # поэтому он проверяется отдельно и детерминированно (тест ниже).
        loser = [r for r in (left, right) if r.status_code != 200][0]
        self.assertIn(loser.status_code, (400, 409), loser.text)
        self.assertIn(
            loser.json().get("error_code"),
            (
                SettingsErrors.LAST_ADMIN,
                SettingsErrors.ROLE_CHANGE_CONFLICT,
            ),
            loser.text,
        )

        self.assertEqual(
            await self.admin_count(), 1, "система осталась без администратора"
        )

    async def test_rival_demotion_during_the_wait_gives_409(self):
        """Детерминированный конфликт: 409 у того, кто ждал блокировки.

        Гонку не подстраиваем случайным расписанием — берём ту же
        блокировку, что берёт эндпоинт, и держим её. Запрос успевает
        посчитать админов (их двое: чужое понижение ещё не закоммичено) и
        встаёт в очередь. Соперник снимает второго админа и коммитит, чем
        одновременно отпускает блокировку. Запрос просыпается, видит
        единственного админа там, где только что было двое, — это и есть
        разница между «админ и был последним» (400) и «пока я ждал, его
        сняли» (409).
        """
        first = await self.make_actor("first", "admin")
        second = await self.make_actor("second", "admin")

        async with self.session_factory() as rival:
            await rival.exec(settings_endpoint._LOCK_ROLE_CHANGES)

            request = asyncio.create_task(self.set_role(first, second, "user"))
            await self._wait_until_someone_waits_for_the_lock(rival)

            await rival.exec(
                text('UPDATE "user" SET role = :role WHERE id = :id'),
                params={"role": "user", "id": first.id},
            )
            await rival.commit()

            response = await asyncio.wait_for(request, timeout=15)

        self.assertRefused(
            response, 409, SettingsErrors.ROLE_CHANGE_CONFLICT
        )
        self.assertEqual(await self.role_of(second), "admin")
        self.assertEqual(await self.admin_count(), 1)

    async def test_lock_is_released_after_a_refusal(self):
        """Отказ не оставляет блокировку висеть: следующий запрос проходит."""
        admin = await self.make_actor("chief", "admin")

        self.assertRefused(
            await self.set_role(admin, admin, "user"),
            400,
            SettingsErrors.LAST_ADMIN,
        )

        ordinary = await self.make_user("newcomer", "user")
        response = await asyncio.wait_for(
            self.set_role(admin, ordinary, "content_manager"), timeout=10
        )
        self.assertEqual(response.status_code, 200, response.text)


# --- Поколение токенов --------------------------------------------------


class TokenVersionTests(RoleChangeTestCase):
    """Понижение обесценивает выданные токены.

    Комментарий к модели User (app/shared/models/entities.py) обещает
    инкремент token_version в том числе при понижении роли, но делали его
    только смена пароля и /auth/logout-all. Без него у разжалованного
    оставались рабочие refresh-токены ещё на неделю.
    """

    async def token_version_of(self, user: User) -> int:
        return (await self.get_row(User, user.id)).token_version or 0

    async def test_demotion_to_user_raises_token_version(self):
        admin = await self.make_actor("chief", "admin")
        victim = await self.make_user("colleague", "admin")
        before = await self.token_version_of(victim)

        response = await self.set_role(admin, victim, "user")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(await self.token_version_of(victim), before + 1)

    async def test_demotion_by_one_step_also_raises_token_version(self):
        admin = await self.make_actor("chief", "admin")
        victim = await self.make_user("colleague", "admin")

        await self.set_role(admin, victim, "content_manager")

        self.assertEqual(await self.token_version_of(victim), 1)

    async def test_content_manager_demoted_to_user_raises_token_version(self):
        admin = await self.make_actor("chief", "admin")
        manager = await self.make_user("manager", "content_manager")

        await self.set_role(admin, manager, "user")

        self.assertEqual(await self.token_version_of(manager), 1)

    async def test_promotion_does_not_raise_token_version(self):
        """Повышение никого не выкидывает из сессии: гасить нечего."""
        admin = await self.make_actor("chief", "admin")
        ordinary = await self.make_user("newcomer", "user")

        await self.set_role(admin, ordinary, "admin")

        self.assertEqual(await self.token_version_of(ordinary), 0)

    async def test_unchanged_role_does_not_raise_token_version(self):
        admin = await self.make_actor("chief", "admin")
        manager = await self.make_user("manager", "content_manager")

        await self.set_role(admin, manager, "content_manager")

        self.assertEqual(await self.token_version_of(manager), 0)

    async def test_refused_demotion_leaves_token_version_alone(self):
        admin = await self.make_actor("chief", "admin")

        await self.set_role(admin, admin, "user")

        self.assertEqual(await self.token_version_of(admin), 0)


# --- Журнал -------------------------------------------------------------


class RoleChangeAuditLogTests(RoleChangeTestCase):
    """Кто, кому и с чего на что. Раньше следа не оставалось нигде."""

    async def test_successful_change_is_logged_with_both_parties(self):
        admin = await self.make_actor("chief", "admin")
        ordinary = await self.make_user("newcomer", "user")

        with self.assertLogs(LOGGER_NAME, level="INFO") as captured:
            response = await self.set_role(admin, ordinary, "content_manager")

        self.assertEqual(response.status_code, 200, response.text)
        record = "\n".join(captured.output)
        self.assertIn(f"actor_id={admin.id}", record)
        self.assertIn("chief", record)
        self.assertIn(f"target_id={ordinary.id}", record)
        self.assertIn("newcomer", record)
        self.assertIn("user -> content_manager", record)

    async def test_demotion_of_an_admin_is_logged(self):
        admin = await self.make_actor("chief", "admin")
        victim = await self.make_user("colleague", "admin")

        with self.assertLogs(LOGGER_NAME, level="INFO") as captured:
            await self.set_role(admin, victim, "user")

        self.assertIn("admin -> user", "\n".join(captured.output))

    async def test_refusal_is_logged_as_a_warning(self):
        admin = await self.make_actor("chief", "admin")

        with self.assertLogs(LOGGER_NAME, level="WARNING") as captured:
            await self.set_role(admin, admin, "user")

        record = "\n".join(captured.output)
        self.assertIn("WARNING", record)
        self.assertIn("last admin", record)
        self.assertIn(f"target_id={admin.id}", record)


if __name__ == "__main__":
    unittest.main()
