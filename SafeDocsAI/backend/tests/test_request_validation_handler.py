"""Регрессия на обработчик RequestValidationError (app/main.py).

До него 422 был последним классом ответов без машинного кода: тело собирал
сам FastAPI, и `GET /api/v1/settings/users?limit=501` уходил клиенту сырым
английским «Input should be less than or equal to 500» — в продукте, где язык
по умолчанию таджикский.

Главный риск правки не в том, что кода не будет, а в двух соседях.

  * Массив detail обязан остаться массивом. На нём держится разбор на клиенте
    (frontend/src/lib/apiError.js): имя непринятого поля берётся из loc, а
    422 от схем с extra="forbid" опознаётся по type == extra_forbidden и
    получает собственное сообщение про баг интерфейса. Свернуть detail в
    строку значило бы обменять «сервер не принял поле limit» на «проверьте
    введённые данные».
  * Обработчик не должен отобрать ответ у тех, кого уже разбирают правильно:
    ApiError, ExternalServiceError, обычный HTTPException и непойманное
    исключение. Отдельный случай — ApiError, брошенный ИЗ валидатора Pydantic
    (deps.QuestionStr в app/api/deps.py): он летит рядом с валидацией и обязан
    сохранить свой код.

Ни Postgres, ни ChromaDB, ни Ollama здесь не нужны: временные маршруты
подцепляются к настоящему приложению и отвергают запрос на разборе тела, до
всяких зависимостей. Поэтому tests/dbfixtures.py не используется.
"""

import unittest

from typing import Annotated

from fastapi import HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict
from pydantic.functional_validators import BeforeValidator

from app.core.exceptions import (
    ApiError,
    ChatErrors,
    ExternalServiceError,
    InternalErrors,
    RequestErrors,
    SourceErrors,
)
from app.main import app


VALIDATION_CODE = RequestErrors.VALIDATION_FAILED


class _Payload(BaseModel):
    number: int


class _StrictPayload(BaseModel):
    """Схема с extra="forbid" — как RuntimeSettingsUpdate в разделе настроек."""

    model_config = ConfigDict(extra="forbid")

    title: str


def _reject_blank(value):
    """Копия правила deps.QuestionStr: отказ с машинным кодом ИЗ валидатора.

    Ради него тест и заведён: ApiError, брошенный внутри валидатора Pydantic,
    не является ошибкой валидации в смысле pydantic-core и до
    RequestValidationError не сворачивается. Если это когда-нибудь изменится,
    коды chat.question_required и chat.question_too_long молча превратятся в
    общий код валидации.
    """
    if not isinstance(value, str):
        return value
    trimmed = value.strip()
    if not trimmed:
        raise ApiError(422, ChatErrors.QUESTION_REQUIRED, "Question must not be empty")
    return trimmed


class _QuestionPayload(BaseModel):
    question: Annotated[str, BeforeValidator(_reject_blank)]


async def _echo(payload: _Payload):
    return {"number": payload.number}


async def _echo_strict(payload: _StrictPayload):
    return {"title": payload.title}


async def _limited(limit: int = Query(default=50, ge=1, le=500)):
    """Слепок постраничного limit из раздела настроек: тот же предел 500."""
    return {"limit": limit}


async def _ask(payload: _QuestionPayload):
    return {"question": payload.question}


async def _api_error():
    raise ApiError(
        status_code=404,
        error_code=SourceErrors.NOT_FOUND,
        detail="Source not found",
    )


async def _external_error():
    raise ExternalServiceError("Ollama is unreachable", service="ollama")


async def _forbidden():
    raise HTTPException(status_code=403, detail="Not enough privileges")


async def _boom():
    raise RuntimeError("unexpected")


async def _stream_echo(payload: _QuestionPayload):
    """Точный слепок chat_stream (app/api/endpoints/chat.py).

    Тело проверяется схемой, а StreamingResponse собирается уже в теле
    обработчика — то есть отказ валидации случается ДО того, как появляется
    поток, и уходит обычным ответом.
    """

    async def _events():
        yield 'event: done\ndata: {"answer": "ok"}\n\n'

    return StreamingResponse(_events(), media_type="text/event-stream")


# Маршруты только на время этого файла: приложение общее для всех тестов,
# и оставлять в нём отладочные эндпоинты нельзя.
_TEST_ROUTES = (
    ("/__tests__/validation/echo", _echo, ["POST"]),
    ("/__tests__/validation/strict", _echo_strict, ["POST"]),
    ("/__tests__/validation/limited", _limited, ["GET"]),
    ("/__tests__/validation/ask", _ask, ["POST"]),
    ("/__tests__/validation/api-error", _api_error, ["GET"]),
    ("/__tests__/validation/external-error", _external_error, ["GET"]),
    ("/__tests__/validation/forbidden", _forbidden, ["GET"]),
    ("/__tests__/validation/boom", _boom, ["GET"]),
    ("/__tests__/validation/stream", _stream_echo, ["POST"]),
)


def setUpModule():
    for path, endpoint, methods in _TEST_ROUTES:
        app.add_api_route(path, endpoint, methods=methods)


def tearDownModule():
    paths = {path for path, _, _ in _TEST_ROUTES}
    app.router.routes = [
        route
        for route in app.router.routes
        if getattr(route, "path", None) not in paths
    ]
    app.openapi_schema = None


class ValidationResponseShapeTests(unittest.TestCase):
    """Что теперь лежит в теле 422."""

    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_invalid_body_carries_the_validation_code(self):
        response = self.client.post(
            "/__tests__/validation/echo", json={"number": "not-a-number"}
        )

        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertEqual(body["error_code"], VALIDATION_CODE)
        self.assertEqual(body["error_code"], "request.validation_failed")

    def test_detail_stays_the_same_array_of_pydantic_errors(self):
        """Форма detail не изменилась — на ней держится разбор на клиенте."""
        response = self.client.post(
            "/__tests__/validation/echo", json={"number": "not-a-number"}
        )

        detail = response.json()["detail"]
        self.assertIsInstance(detail, list)
        self.assertEqual(len(detail), 1)
        item = detail[0]
        # Ключи, которые читает frontend/src/lib/apiError.js.
        self.assertIn("type", item)
        self.assertIn("msg", item)
        self.assertIsInstance(item["loc"], list)
        self.assertEqual(item["loc"][-1], "number")

    def test_query_parameter_out_of_range_names_the_field_and_the_bound(self):
        """Тот самый живой случай: ?limit=501 при пределе 500."""
        response = self.client.get("/__tests__/validation/limited?limit=501")

        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertEqual(body["error_code"], VALIDATION_CODE)
        item = body["detail"][0]
        self.assertEqual(item["loc"][-1], "limit")
        # Граница остаётся в ответе: клиент строит из неё сообщение с
        # конкретикой, а не общее «проверьте введённые данные».
        self.assertIn("500", item["msg"])

    def test_missing_required_field_names_it(self):
        response = self.client.post("/__tests__/validation/echo", json={})

        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertEqual(body["error_code"], VALIDATION_CODE)
        self.assertEqual(body["detail"][0]["loc"][-1], "number")

    def test_malformed_json_is_a_validation_failure_too(self):
        """Нечитаемый JSON приходит тем же RequestValidationError.

        Ответ обязан остаться JSON'ом с кодом, а не выродиться в пятисотку.
        """
        response = self.client.post(
            "/__tests__/validation/echo",
            content=b"{not json at all",
            headers={"Content-Type": "application/json"},
        )

        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertEqual(body["error_code"], VALIDATION_CODE)
        self.assertIsInstance(body["detail"], list)
        # ctx у json_invalid содержит объект, который json.dumps не берёт, —
        # поэтому в обработчике стоит jsonable_encoder.
        self.assertEqual(body["detail"][0]["type"], "json_invalid")

    def test_valid_request_is_untouched(self):
        response = self.client.post("/__tests__/validation/echo", json={"number": 7})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"number": 7})
        self.assertNotIn("error_code", response.text)


class ExtraForbiddenStaysRecognisableTests(unittest.TestCase):
    """Договор с клиентом по 422 от схем с extra="forbid".

    Клиент опознаёт такой ответ по элементам detail (type == extra_forbidden,
    имя ключа последним элементом loc) и показывает своё сообщение про баг
    интерфейса, а не про ошибку ввода. Добавление error_code не должно этому
    мешать: разбор идёт по detail, а не по коду.
    """

    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_unknown_key_keeps_its_type_and_name_in_detail(self):
        response = self.client.post(
            "/__tests__/validation/strict",
            json={"title": "ok", "typo_field": 1},
        )

        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertEqual(body["error_code"], VALIDATION_CODE)

        offenders = [
            item["loc"][-1]
            for item in body["detail"]
            if item.get("type") == "extra_forbidden"
        ]
        self.assertEqual(offenders, ["typo_field"])

    def test_several_unknown_keys_all_reach_the_client(self):
        response = self.client.post(
            "/__tests__/validation/strict",
            json={"title": "ok", "alpha": 1, "beta": 2},
        )

        self.assertEqual(response.status_code, 422)
        offenders = sorted(
            item["loc"][-1]
            for item in response.json()["detail"]
            if item.get("type") == "extra_forbidden"
        )
        self.assertEqual(offenders, ["alpha", "beta"])


class HandlerDoesNotShadowOthersTests(unittest.TestCase):
    """Обработчик валидации не должен перехватывать чужие ответы.

    Выбор обработчика — не порядок регистрации, а обход __mro__ пойманного
    исключения (starlette/_exception_handler.py). RequestValidationError
    наследует ValidationException, а не HTTPException, поэтому пересечься с
    ApiError, HTTPException и ExternalServiceError ему нечем; обработчик на
    Exception к тому же вынесен во внешний слой (ServerErrorMiddleware) и
    сюда не достаёт. Тесты фиксируют это снаружи.
    """

    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_unknown_path_is_still_404(self):
        response = self.client.get("/api/v1/definitely-no-such-endpoint")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Not Found")
        self.assertNotIn(VALIDATION_CODE, response.text)

    def test_method_not_allowed_is_still_405(self):
        response = self.client.delete("/health")

        self.assertEqual(response.status_code, 405)
        self.assertNotIn(VALIDATION_CODE, response.text)

    def test_plain_http_exception_403_keeps_its_body(self):
        response = self.client.get("/__tests__/validation/forbidden")

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], "Not enough privileges")
        self.assertNotIn(VALIDATION_CODE, response.text)

    def test_api_error_keeps_its_own_error_code(self):
        response = self.client.get("/__tests__/validation/api-error")

        self.assertEqual(response.status_code, 404)
        body = response.json()
        self.assertEqual(body["error_code"], SourceErrors.NOT_FOUND)
        self.assertEqual(body["detail"], "Source not found")

    def test_api_error_from_inside_a_validator_keeps_its_own_code(self):
        """Отказ по вопросу приходит с 422 — но со СВОИМ кодом, не общим.

        Так устроен deps.QuestionStr: пустой вопрос отвергается ApiError'ом
        из валидатора именно для того, чтобы у ответа был точный код и
        точный перевод.
        """
        response = self.client.post("/__tests__/validation/ask", json={"question": "  "})

        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertEqual(body["error_code"], ChatErrors.QUESTION_REQUIRED)
        self.assertNotEqual(body["error_code"], VALIDATION_CODE)
        # И detail остаётся строкой, а не массивом: это ответ ApiError.
        self.assertIsInstance(body["detail"], str)

    def test_type_error_next_to_that_validator_is_a_plain_validation_failure(self):
        """«question: 5» валидатор пропускает — отвечает Pydantic, и теперь с кодом."""
        response = self.client.post("/__tests__/validation/ask", json={"question": 5})

        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertEqual(body["error_code"], VALIDATION_CODE)
        self.assertIsInstance(body["detail"], list)

    def test_external_service_error_keeps_its_own_body(self):
        with self.assertLogs("app.main", level="WARNING"):
            response = self.client.get("/__tests__/validation/external-error")

        self.assertEqual(response.status_code, 502)
        body = response.json()
        self.assertEqual(body["service"], "ollama")
        self.assertEqual(body["detail"], "Ollama is unreachable")
        self.assertNotIn(VALIDATION_CODE, response.text)

    def test_unhandled_exception_is_still_the_internal_error(self):
        with self.assertLogs("app.main", level="ERROR"):
            response = self.client.get("/__tests__/validation/boom")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error_code"], InternalErrors.INTERNAL_ERROR)
        self.assertNotIn(VALIDATION_CODE, response.text)


class StreamingEndpointTests(unittest.TestCase):
    """Отказ валидации у SSE-маршрута — обычный ответ, а не событие в потоке.

    Тело запроса разбирается до того, как обработчик соберёт
    StreamingResponse (app/api/endpoints/chat.py, chat_stream: сначала
    параметр-схема, потом return StreamingResponse), поэтому ответ ещё можно
    отдать целиком. Если валидация когда-нибудь переедет внутрь генератора,
    клиент получит 200 и text/event-stream, а ошибку — внутриполосно, и эти
    тесты покраснеют.

    Маршрут здесь синтетический, но собран так же, как chat_stream: настоящий
    /api/v1/chat/stream сначала проверяет сессию и до RequestValidationError
    без базы не доходит. Тот же путь на живом эндпоинте закреплён в
    tests/test_question_validation_db.py.
    """

    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_invalid_body_answers_with_json_not_sse(self):
        response = self.client.post("/__tests__/validation/stream", json={})

        self.assertEqual(response.status_code, 422)
        self.assertTrue(
            response.headers.get("content-type", "").startswith("application/json"),
            response.headers.get("content-type"),
        )
        # Ни разметки SSE, ни следа потока.
        self.assertNotIn("data:", response.text)
        self.assertNotIn("event:", response.text)

        body = response.json()
        self.assertEqual(body["error_code"], VALIDATION_CODE)
        self.assertIsInstance(body["detail"], list)
        self.assertEqual(body["detail"][0]["loc"][-1], "question")

    def test_valid_body_still_streams(self):
        """Контрольный: маршрут и правда SSE, значит предыдущий тест не пустой."""
        response = self.client.post(
            "/__tests__/validation/stream", json={"question": "Какова ставка НДС?"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertIn("data:", response.text)


if __name__ == "__main__":
    unittest.main()
