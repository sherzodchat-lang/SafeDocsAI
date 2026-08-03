"""Автор записи журнала: nullable в схеме, обязателен для новых записей.

Решение, которое закрепляет этот файл. log.user_id остаётся nullable
осознанно, в отличие от notebook.owner_id и document.owner_id, где NULL был
багом. Блокнот и документ — ресурсы, а владение ресурсом обязательно. Журнал
хранит запись о событии, и «у события нет автора» бывает правдой:
legacy-строки, системное действие, возможный будущий фоновый писатель.
Переписать такие строки на админа значило бы заставить журнал врать, что
админ делал то, чего не делал; удалить — потерять историю, ради которой
журнал и ведётся. Поведение уже правильное: ничью запись видит только админ
(app/api/deps.py, user_owns).

Обратная сторона того же решения: сегодня безавторского писателя нет ни
одного. Журнал пишут только chat_request, chat_request_stream и
handle_ask_request — обработчики HTTP-запроса аутентифицированного
пользователя; фоновый воркер индексации в журнал не пишет вовсе. Поэтому
новая строка без user_id — не системное событие, а потерянный автор, и её
отсекает require_log_author (app/modules/chat/service.py) до вставки.

Две половины решения тянут в разные стороны, и тест держит обе:

  * NULL в колонке по-прежнему возможен — строка с user_id IS NULL пишется в
    настоящую схему. Тест упадёт, если кто-нибудь «дочинит» журнал по образцу
    блокнота и повесит на колонку NOT NULL: старые строки трогать нельзя;
  * ничья запись доступна только админу — и в списке, и в проставлении
    оценки;
  * новая запись без автора не создаётся: оба входа в журнал требуют автора
    раньше, чем начинают работу.

Почему настоящий PostgreSQL. Проверяется именно схема — что колонка приняла
NULL, — а словарь вместо БД принял бы что угодно и зеленел бы даже после
ALTER TABLE ... SET NOT NULL. Схему создаёт код проекта (init_db), поэтому
ограничения здесь ровно те же, что в рабочей базе.

Ни ChromaDB, ни Ollama не нужны: проверки записи отбивают запрос раньше, чем
сервисы создаются, а остальное — чистые запросы к БД через API.
"""

import os
import sys
import unittest
from datetime import datetime

# Каталог с тестами в sys.path: unittest discover кладёт его туда сам, но при
# запуске `python -m unittest tests.test_log_author_db` этого не происходит.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dbfixtures import DatabaseBackedTestCase  # noqa: E402

from app.core.exceptions import ApiError  # noqa: E402
from app.modules.ask.schemas import AskRequest  # noqa: E402
from app.modules.ask.service import handle_ask_request  # noqa: E402
from app.modules.chat.schemas import ChatRequest  # noqa: E402
from app.modules.chat.service import chat_request, require_log_author  # noqa: E402
from app.shared.models import Log, User  # noqa: E402


LOGS = "/api/v1/logs"


class OrphanLogRowTests(DatabaseBackedTestCase):
    """Ничья запись журнала: существует в схеме и видна только админу."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()

        self.user = await self.make_user("author", "user")
        self.admin = await self.make_user("root", "admin")

        # Строка ровно того вида, что осталась от прежних версий: событие
        # есть, автора у него нет.
        self.orphan = await self.seed(
            Log(
                question="Ничей вопрос",
                answer="Ничей ответ",
                time_ms=1,
                user_id=None,
                created_at=datetime(2026, 1, 1),
            )
        )
        self.owned = await self.seed(
            Log(
                question="Свой вопрос",
                answer="Свой ответ",
                time_ms=1,
                user_id=self.user.id,
                created_at=datetime(2026, 1, 2),
            )
        )

    async def test_log_without_author_is_storable(self):
        """NOT NULL на колонке запретил бы и старые строки — его тут быть не должно."""
        stored = await self.get_row(Log, self.orphan.id)
        self.assertIsNotNone(stored)
        self.assertIsNone(stored.user_id)

    async def test_admin_sees_orphan_log_in_listing(self):
        self.as_user(self.admin)
        response = await self.client.get(f"{LOGS}/")
        self.assertEqual(response.status_code, 200)
        ids = {item["id"] for item in response.json()}
        self.assertIn(self.orphan.id, ids)

    async def test_listing_is_closed_to_regular_user(self):
        self.as_user(self.user)
        response = await self.client.get(f"{LOGS}/")
        self.assertEqual(response.status_code, 403)

    async def test_regular_user_cannot_rate_orphan_log(self):
        self.as_user(self.user)
        response = await self.client.post(
            f"{LOGS}/{self.orphan.id}/rating", json={"rating": "up"}
        )
        # 404, а не 403: ответ не должен подтверждать существование записи.
        self.assertEqual(response.status_code, 404)
        stored = await self.get_row(Log, self.orphan.id)
        self.assertIsNone(stored.rating)

    async def test_orphan_log_does_not_leak_content_to_regular_user(self):
        self.as_user(self.user)
        response = await self.client.post(
            f"{LOGS}/{self.orphan.id}/rating", json={"rating": "up"}
        )
        self.assertNotIn("Ничей вопрос", response.text)
        self.assertNotIn("Ничей ответ", response.text)

    async def test_admin_rates_orphan_log(self):
        self.as_user(self.admin)
        response = await self.client.post(
            f"{LOGS}/{self.orphan.id}/rating", json={"rating": "up"}
        )
        self.assertEqual(response.status_code, 200)
        stored = await self.get_row(Log, self.orphan.id)
        self.assertEqual(stored.rating, "up")

    async def test_own_log_stays_available_to_its_author(self):
        """Ничьи записи закрыты, но своя — по-прежнему своя."""
        self.as_user(self.user)
        response = await self.client.post(
            f"{LOGS}/{self.owned.id}/rating", json={"rating": "down"}
        )
        self.assertEqual(response.status_code, 200)
        stored = await self.get_row(Log, self.owned.id)
        self.assertEqual(stored.rating, "down")


class RequireLogAuthorTests(unittest.TestCase):
    """Проверка автора отдельно от эндпоинтов: БД для неё не нужна."""

    def test_returns_id_of_saved_user(self):
        user = User(id=7, username="author", password_hash="x", role="user")
        self.assertEqual(require_log_author(user), 7)

    def test_user_without_id_is_rejected(self):
        user = User(username="ghost", password_hash="x", role="user")
        with self.assertRaises(ApiError) as ctx:
            require_log_author(user)
        self.assertEqual(ctx.exception.status_code, 401)


class LogWritersRequireAuthorTests(unittest.IsolatedAsyncioTestCase):
    """Оба входа в журнал требуют автора раньше, чем начинают работу.

    session=None здесь не заглушка, а часть проверки: если бы автор
    выяснялся позже, тест упал бы AttributeError на первом же обращении к
    сессии, а не ожидаемым отказом.
    """

    def setUp(self):
        self.ghost = User(username="ghost", password_hash="x", role="user")

    async def test_chat_refuses_to_write_authorless_log(self):
        with self.assertRaises(ApiError) as ctx:
            await chat_request(
                chat_request=ChatRequest(question="Привет"),
                current_user=self.ghost,
                session=None,
            )
        self.assertEqual(ctx.exception.status_code, 401)

    async def test_ask_refuses_to_write_authorless_log(self):
        with self.assertRaises(ApiError) as ctx:
            await handle_ask_request(
                ask_request=AskRequest(question="Привет"),
                current_user=self.ghost,
                session=None,
            )
        self.assertEqual(ctx.exception.status_code, 401)


if __name__ == "__main__":
    unittest.main()
