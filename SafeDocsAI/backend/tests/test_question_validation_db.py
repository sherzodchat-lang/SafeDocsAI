"""Проверка вопроса на входе: пустой и переросший вопрос не доходят до GPU.

Что закрепляем (см. deps.QuestionStr в app/api/deps.py и схемы -In в
app/api/endpoints/chat.py и ask.py):

  * вопрос обязателен на всех точках входа: POST /chat/, POST /chat/stream,
    POST /ask/ и POST /chat/retrieve. Пустая строка и строка из одних
    пробелов отклоняются одинаково — «   » после подрезки не значение;
  * есть верхняя граница длины (deps.QUESTION_MAX_LENGTH). Она стоит не ради
    БД, а ради окна модели: вставленный в поле вопроса документ вытесняет из
    промпта найденные фрагменты и историю диалога;
  * отказ — 422 с машинным кодом (ChatErrors.QUESTION_REQUIRED и
    QUESTION_TOO_LONG), потому что интерфейс переведён на три языка и строит
    сообщение по коду, а не по английскому detail;
  * отказ приходит ДО обработчика. Отсюда две проверки, ради которых тест и
    написан: обработчик модуля не вызывается вовсе (то есть ни поиска, ни
    генерации не было), а у /chat/stream ответ остаётся обычным JSON-ответом
    и не превращается в событие error внутри уже начатого SSE;
  * значение на самой границе проходит, и обработчик получает его
    подрезанным — предел считается по подрезанному вопросу, иначе хвост
    пробелов отбирал бы у пользователя длину.

Почему настоящий PostgreSQL, хотя проверка живёт на слое схемы. Отказ обязан
быть бесследным: ни записи в журнал, ни расхода лимита запросов. Журнал —
таблица с внешними ключами на пользователя и блокнот, и «строк не появилось»
имеет смысл проверять только в настоящей базе. Сам ответ модуля чата
замокан: тест про границу входа, а не про RAG, и поднимать ради него ChromaDB
с Ollama незачем.
"""

import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Каталог с тестами в sys.path: unittest discover кладёт его туда сам, но при
# запуске `python -m unittest tests.test_question_validation_db` этого не
# происходит.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dbfixtures import DatabaseBackedTestCase  # noqa: E402

from app.api import deps  # noqa: E402
from app.core.exceptions import ChatErrors, RequestErrors  # noqa: E402
from app.core.rate_limit import chat_limiter  # noqa: E402
from app.modules.ask.schemas import AskResponse  # noqa: E402
from app.modules.chat.schemas import (  # noqa: E402
    ChatResponse,
    RetrievalResponse,
)
from app.shared.models import Log  # noqa: E402


CHAT = "/api/v1/chat/"
STREAM = "/api/v1/chat/stream"
RETRIEVE = "/api/v1/chat/retrieve"
ASK = "/api/v1/ask/"

# Все точки входа, принимающие вопрос. Значение — имя обработчика модуля в
# модуле эндпоинта: именно его подменяем, чтобы проверить, дошёл ли запрос
# до работы.
ENTRY_POINTS = (
    (CHAT, "app.api.endpoints.chat.handle_chat_request"),
    (STREAM, "app.api.endpoints.chat.handle_chat_request_stream"),
    (RETRIEVE, "app.api.endpoints.chat.handle_retrieve_chunks"),
    (ASK, "app.api.endpoints.ask.handle_ask_request"),
)

BLANK_QUESTIONS = ("", "   ", "\t\n  \r\n")


class QuestionValidationTestCase(DatabaseBackedTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()

        self.user = await self.make_user("asker", "user")
        self.as_user(self.user)

        # Лимитер общий на процесс: без сброса тесты влияли бы друг на друга
        # и на соседние наборы.
        chat_limiter.clients.clear()
        self.addCleanup(chat_limiter.clients.clear)

    # --- помощники ---

    def patch_handler(self, target: str):
        """Подменить обработчик модуля и вернуть мок для проверок вызова."""
        if target.endswith("handle_chat_request_stream"):
            # Обычный MagicMock, а не AsyncMock: обработчик стрима не
            # корутина, его результат уходит в StreamingResponse как
            # асинхронный генератор и не ожидается.
            async def fake_stream(**kwargs):
                yield 'event: done\ndata: {"answer": "ok"}\n\n'

            mock = MagicMock(side_effect=fake_stream)
        elif target.endswith("handle_chat_request"):
            mock = AsyncMock(
                return_value=ChatResponse(answer="ok", sources=[], log_id=1)
            )
        elif target.endswith("handle_retrieve_chunks"):
            mock = AsyncMock(
                return_value=RetrievalResponse(
                    question="ok",
                    search_query="ok",
                    retrieval_top_k=20,
                    top_k=5,
                    chunks=[],
                )
            )
        else:
            mock = AsyncMock(
                return_value=AskResponse(answer="ok", citations=[], log_id=1)
            )

        patcher = patch(target, mock)
        patcher.start()
        self.addCleanup(patcher.stop)
        return mock

    async def post_question(self, url: str, question):
        return await self.client.post(url, json={"question": question})

    def assert_rejected(self, response, expected_code: str, handler) -> None:
        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(response.json().get("error_code"), expected_code)
        # Главное утверждение всего набора: до поиска и генерации не дошло.
        handler.assert_not_called()

    # --- пустой вопрос ---

    async def test_blank_question_is_rejected_on_every_entry_point(self):
        for url, target in ENTRY_POINTS:
            for question in BLANK_QUESTIONS:
                with self.subTest(url=url, question=repr(question)):
                    handler = self.patch_handler(target)
                    response = await self.post_question(url, question)
                    self.assert_rejected(
                        response, ChatErrors.QUESTION_REQUIRED, handler
                    )

    # --- вопрос сверх предела ---

    async def test_question_over_limit_is_rejected_on_every_entry_point(self):
        oversized = "я" * (deps.QUESTION_MAX_LENGTH + 1)
        for url, target in ENTRY_POINTS:
            with self.subTest(url=url):
                handler = self.patch_handler(target)
                response = await self.post_question(url, oversized)
                self.assert_rejected(response, ChatErrors.QUESTION_TOO_LONG, handler)

    async def test_question_far_over_limit_is_rejected(self):
        """Вставленный целиком документ — тот самый случай, ради которого предел."""
        pasted_document = "Статья 1. Общие положения. " * 4000
        self.assertGreater(len(pasted_document), 100_000)
        for url, target in ENTRY_POINTS:
            with self.subTest(url=url):
                handler = self.patch_handler(target)
                response = await self.post_question(url, pasted_document)
                self.assert_rejected(response, ChatErrors.QUESTION_TOO_LONG, handler)

    # --- граница ---

    async def test_question_at_limit_is_accepted_on_every_entry_point(self):
        exact = "я" * deps.QUESTION_MAX_LENGTH
        for url, target in ENTRY_POINTS:
            with self.subTest(url=url):
                handler = self.patch_handler(target)
                response = await self.post_question(url, exact)
                self.assertEqual(response.status_code, 200, response.text)
                handler.assert_called_once()
                received = handler.call_args.kwargs
                request_model = (
                    received.get("chat_request")
                    or received.get("retrieval_request")
                    or received.get("ask_request")
                )
                self.assertEqual(request_model.question, exact)

    async def test_padding_does_not_eat_into_the_limit(self):
        """Предел считается по подрезанному вопросу, а не по присланной строке."""
        padded = "  " + "я" * deps.QUESTION_MAX_LENGTH + "\n\t "
        for url, target in ENTRY_POINTS:
            with self.subTest(url=url):
                handler = self.patch_handler(target)
                response = await self.post_question(url, padded)
                self.assertEqual(response.status_code, 200, response.text)

    async def test_accepted_question_reaches_handler_trimmed(self):
        for url, target in ENTRY_POINTS:
            with self.subTest(url=url):
                handler = self.patch_handler(target)
                response = await self.post_question(url, "  Какова ставка НДС?  ")
                self.assertEqual(response.status_code, 200, response.text)
                received = handler.call_args.kwargs
                request_model = (
                    received.get("chat_request")
                    or received.get("retrieval_request")
                    or received.get("ask_request")
                )
                self.assertEqual(request_model.question, "Какова ставка НДС?")

    # --- особый случай: стрим ---

    async def test_stream_rejects_before_the_stream_starts(self):
        """Отказ у /chat/stream — обычный 422 JSON, а не событие error в SSE.

        Тело запроса разбирается до того, как обработчик соберёт
        StreamingResponse, поэтому ответ ещё можно отдать целиком. Проверяем
        это по типу содержимого и по отсутствию разметки SSE в теле: если
        валидация когда-нибудь переедет внутрь генератора, клиент получит
        200 и text/event-stream, а сообщение об ошибке — внутриполосно.
        """
        handler = self.patch_handler("app.api.endpoints.chat.handle_chat_request_stream")
        response = await self.post_question(STREAM, "   ")

        self.assertEqual(response.status_code, 422, response.text)
        self.assertTrue(
            response.headers.get("content-type", "").startswith("application/json"),
            response.headers.get("content-type"),
        )
        self.assertNotIn("data:", response.text)
        self.assertEqual(response.json().get("error_code"), ChatErrors.QUESTION_REQUIRED)
        handler.assert_not_called()

    # --- отказ не оставляет следов ---

    async def test_rejected_question_is_not_logged(self):
        for url, target in ENTRY_POINTS:
            with self.subTest(url=url):
                self.patch_handler(target)
                await self.post_question(url, "   ")
        self.assertEqual(await self.all_rows(Log), [])

    async def test_rejected_question_does_not_spend_the_rate_limit(self):
        """Отказ дешевле лимита: он срабатывает до check_rate_limit.

        Иначе перебор пустых запросов выбивал бы пользователю минуту тишины,
        ничего при этом не посчитав.
        """
        self.patch_handler("app.api.endpoints.chat.handle_chat_request")
        self.assertEqual(chat_limiter.clients, {})

        for _ in range(5):
            response = await self.post_question(CHAT, "")
            self.assertEqual(response.status_code, 422, response.text)

        self.assertEqual(chat_limiter.clients, {})

    # --- чужие ошибки остаются чужими ---

    async def test_non_string_question_keeps_the_pydantic_error(self):
        """«question: 5» — ошибка типа, и отвечать на неё должен Pydantic.

        Свой код про пустоту здесь был бы неправдой, поэтому валидатор
        пропускает не-строку дальше по цепочке.

        Раньше проверялось «кода нет вовсе»: 422 от Pydantic уходил голым
        телом FastAPI. Теперь код есть у любого отказа валидации
        (RequestErrors.VALIDATION_FAILED, обработчик в app/main.py), и смысл
        проверки в том, что он ОБЩИЙ, а не chat.question_required — вместе с
        массивом detail, по которому клиент называет виноватое поле.
        """
        for value in (5, None, {"text": "вопрос"}):
            with self.subTest(value=value):
                handler = self.patch_handler("app.api.endpoints.chat.handle_chat_request")
                response = await self.client.post(CHAT, json={"question": value})
                self.assertEqual(response.status_code, 422, response.text)
                body = response.json()
                self.assertEqual(
                    body.get("error_code"), RequestErrors.VALIDATION_FAILED
                )
                self.assertNotEqual(
                    body.get("error_code"), ChatErrors.QUESTION_REQUIRED
                )
                self.assertIsInstance(body["detail"], list)
                self.assertEqual(body["detail"][0]["loc"][-1], "question")
                handler.assert_not_called()

    async def test_missing_question_field_is_rejected(self):
        handler = self.patch_handler("app.api.endpoints.chat.handle_chat_request")
        response = await self.client.post(CHAT, json={"notebook_id": None})
        self.assertEqual(response.status_code, 422, response.text)
        body = response.json()
        # Отсутствующее поле — тоже отказ Pydantic: общий код и имя поля в detail.
        self.assertEqual(body.get("error_code"), RequestErrors.VALIDATION_FAILED)
        self.assertEqual(body["detail"][0]["loc"][-1], "question")
        handler.assert_not_called()


if __name__ == "__main__":
    unittest.main()
