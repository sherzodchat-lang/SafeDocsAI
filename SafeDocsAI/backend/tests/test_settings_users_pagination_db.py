"""GET /api/v1/settings/users: страница вместо всей таблицы — на настоящем PostgreSQL.

Что закрепляем.

  * **Список больше не отдаётся целиком.** Эндпоинт делал
    `select(User).order_by(...)` без offset/limit вообще: экран админский, но
    растёт он вместе с числом регистраций, и потолка у ответа не было
    никакого.
  * **Общее число наконец видно.** Тело — голый массив, поля для счётчика в
    нём нет и быть не может (менять форму ответа нельзя, клиент читает
    массив), поэтому число уходит заголовком X-Total-Count — ровно как у
    GET /sources/, /notebooks/ и /notes/.
  * **Форма ответа не изменилась.** Клиент, не знающий о пагинации и не
    передающий параметров, обязан получить то же, что и раньше: массив и все
    записи. Отсюда умолчание limit, равное потолку (DEFAULT_PAGE_SIZE ==
    MAX_PAGE_SIZE) — то же решение, что принято в /notebooks/ и /notes/.
  * **Порядок устойчив.** created_at DESC, id DESC. Без второго ключа
    пользователи с одинаковым created_at (регистрации пачкой, заведение
    тестовых учёток скриптом) встают между запросами в разном порядке: одна
    запись показывается на двух соседних страницах, другая — ни на одной.

Настоящая БД нужна не для удобства: проверяется SQL — offset/limit, COUNT(*) и
устойчивость сортировки, — а в словаре вместо таблицы ничего этого нет. Раздел
закрыт get_current_active_superuser, поэтому роль берётся у настоящей строки
user.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta

# Каталог с тестами в sys.path: unittest discover кладёт его туда сам, но при
# запуске `python -m unittest tests.test_settings_users_pagination_db` — нет.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dbfixtures import DatabaseBackedTestCase  # noqa: E402

from app.api.endpoints.documents import MAX_PAGE_SIZE  # noqa: E402
from app.api.endpoints.settings import DEFAULT_PAGE_SIZE  # noqa: E402
from app.main import app  # noqa: E402
from app.shared.models import User  # noqa: E402


USERS = "/api/v1/settings/users"
TOTAL_COUNT = "X-Total-Count"

# Одинаковый момент создания у всех — то состояние, в котором сортировка без
# второго ключа и разъезжается. utcnow() пишет naive-UTC (см. shared/models).
SAME_MOMENT = datetime(2026, 7, 30, 9, 15, 0)


class UsersPageTestCase(DatabaseBackedTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.admin = await self.make_user("root", "admin")
        self.as_user(self.admin)

    async def make_users(self, count: int, *, same_moment: bool = False) -> list[User]:
        """Завести count пользователей помимо админа из asyncSetUp."""
        rows = [
            User(
                username=f"u{index:03d}",
                password_hash="not-a-real-hash",
                role="user",
                created_at=(
                    SAME_MOMENT
                    if same_moment
                    else SAME_MOMENT + timedelta(seconds=index)
                ),
            )
            for index in range(count)
        ]
        await self.seed(*rows)
        return rows

    async def page(self, **params) -> tuple[list[dict], int]:
        response = await self.client.get(USERS, params=params)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        # Форма ответа не изменилась: голый массив, а не объект с items.
        self.assertIsInstance(body, list, body)
        self.assertIn(TOTAL_COUNT, response.headers, dict(response.headers))
        return body, int(response.headers[TOTAL_COUNT])


# --- Совместимость: форма ответа и умолчания -----------------------------


class TheContractDidNotChangeTests(UsersPageTestCase):
    async def test_a_client_that_sends_no_parameters_loses_nothing(self):
        """Главная проверка совместимости.

        SettingsPage.jsx параметров не передаёт. Умолчание limit меньше
        потолка молча урезало бы уже отдаваемый список: админ увидел бы часть
        пользователей и не узнал бы, что видит не всех.
        """
        await self.make_users(20)

        body, total = await self.page()

        self.assertEqual(len(body), 21)  # 20 + админ
        self.assertEqual(total, 21)

    async def test_the_default_page_size_equals_the_ceiling(self):
        """Умолчание выбрано так же, как в /notebooks/ и /notes/."""
        self.assertEqual(DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE)

    async def test_an_item_still_carries_the_same_four_fields(self):
        body, _ = await self.page()

        self.assertEqual(
            sorted(body[0]), ["created_at", "id", "role", "username"]
        )

    async def test_the_freshest_user_is_still_first(self):
        await self.make_users(3)

        body, _ = await self.page()

        # Админ заведён в asyncSetUp с created_at=utcnow(), то есть позже всех
        # остальных: порядок created_at DESC ставит его первым.
        self.assertEqual(body[0]["username"], "root")
        self.assertEqual(
            [item["username"] for item in body[1:]], ["u002", "u001", "u000"]
        )

    async def test_a_page_past_the_end_is_empty_but_still_counts(self):
        """Пустая страница — не «пользователей нет»: счётчик по-прежнему
        говорит, сколько их всего, и клиенту есть куда вернуться."""
        body, total = await self.page(skip=50)

        self.assertEqual(body, [])
        self.assertEqual(total, 1)


# --- Заголовок с общим числом -------------------------------------------


class TotalCountHeaderTests(UsersPageTestCase):
    async def test_the_total_ignores_skip_and_limit(self):
        """Интерфейсу нужно общее число, а не длина страницы, — иначе
        нарисовать пагинацию нечем."""
        await self.make_users(9)

        body, total = await self.page(skip=2, limit=3)

        self.assertEqual(len(body), 3)
        self.assertEqual(total, 10)

    async def test_the_header_is_exposed_to_the_browser(self):
        """Без expose_headers браузер не отдаёт заголовок скрипту страницы, и
        счётчик остаётся недоступен именно в том окружении, ради которого он
        заведён."""
        exposed = [
            middleware.kwargs.get("expose_headers", [])
            for middleware in app.user_middleware
            if "CORS" in str(middleware.cls.__name__)
        ]

        self.assertTrue(exposed, "CORSMiddleware не найден")
        self.assertIn(TOTAL_COUNT, exposed[0])


# --- Страницы ------------------------------------------------------------


class PagingTests(UsersPageTestCase):
    async def test_skip_and_limit_cut_neighbouring_pages(self):
        await self.make_users(5)

        first, total = await self.page(limit=2)
        second, _ = await self.page(skip=2, limit=2)

        self.assertEqual(total, 6)
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 2)
        self.assertFalse(
            {item["id"] for item in first} & {item["id"] for item in second}
        )

    async def test_walking_the_pages_shows_every_user_exactly_once(self):
        """Проверка устойчивости порядка.

        Все созданы в один момент. Без второго ключа сортировки (id DESC)
        PostgreSQL вправе вернуть их в любом порядке, и он вправе быть разным
        от запроса к запросу: обход страницами тогда и дублирует записи, и
        теряет их.
        """
        await self.make_users(12, same_moment=True)

        seen: list[int] = []
        for skip in range(0, 15, 3):
            body, total = await self.page(skip=skip, limit=3)
            seen.extend(item["id"] for item in body)
            self.assertEqual(total, 13)

        self.assertEqual(len(seen), 13)
        self.assertEqual(len(set(seen)), 13, "запись попала на две страницы")

    async def test_the_order_of_a_page_is_repeatable(self):
        await self.make_users(8, same_moment=True)

        once, _ = await self.page(skip=3, limit=3)
        twice, _ = await self.page(skip=3, limit=3)

        self.assertEqual(
            [item["id"] for item in once], [item["id"] for item in twice]
        )


# --- Границы параметров --------------------------------------------------


class ParameterBoundsTests(UsersPageTestCase):
    async def test_the_whole_table_cannot_be_asked_for_in_one_request(self):
        """Потолок и есть смысл правки: ?limit=100000000 больше не выгружает
        таблицу user целиком."""
        response = await self.client.get(USERS, params={"limit": MAX_PAGE_SIZE + 1})

        self.assertEqual(response.status_code, 422, response.text)

    async def test_an_empty_page_cannot_be_asked_for(self):
        response = await self.client.get(USERS, params={"limit": 0})

        self.assertEqual(response.status_code, 422, response.text)

    async def test_a_negative_offset_is_refused(self):
        response = await self.client.get(USERS, params={"skip": -1})

        self.assertEqual(response.status_code, 422, response.text)

    async def test_the_ceiling_itself_is_allowed(self):
        _, total = await self.page(limit=MAX_PAGE_SIZE)

        self.assertEqual(total, 1)


# --- Документация --------------------------------------------------------


class OpenApiTests(UsersPageTestCase):
    async def test_the_header_is_declared_in_the_schema(self):
        """Заголовок, не объявленный в responses, для генератора клиента не
        существует — а тело массивом счётчика не содержит."""
        schema = app.openapi()
        operation = schema["paths"]["/api/v1/settings/users"]["get"]

        self.assertIn(TOTAL_COUNT, operation["responses"]["200"].get("headers", {}))

    async def test_skip_and_limit_are_declared_in_the_schema(self):
        schema = app.openapi()
        operation = schema["paths"]["/api/v1/settings/users"]["get"]

        declared = {parameter["name"] for parameter in operation.get("parameters", [])}
        self.assertLessEqual({"skip", "limit"}, declared)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
