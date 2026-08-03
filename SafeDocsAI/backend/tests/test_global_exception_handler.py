"""Регрессия на обработчик непойманных исключений (app/main.py).

До него непойманное исключение уходило клиенту голым 500 «Internal Server
Error»: ни JSON, ни error_code, показать пользователю нечего. Так живьём
выглядели два разных дефекта — KeyError в настройках и OSError на длинном
имени файла.

Главный риск правки — не то, что обработчик не сработает, а то, что он
сработает слишком широко и превратит осмысленные 404/422 в пятисотки.
Поэтому половина файла проверяет, что ответы, которые разбираются раньше
(HTTPException, RequestValidationError, ApiError, ExternalServiceError),
остались ровно такими же.

Ни Postgres, ни ChromaDB, ни Ollama здесь не нужны: временные маршруты
подцепляются к настоящему приложению и падают сами, до всяких зависимостей.
Поэтому tests/dbfixtures.py не используется.
"""

import unittest
from unittest.mock import patch

from fastapi import Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.core.config import settings
from app.core.exceptions import (
    ApiError,
    ExternalServiceError,
    InternalErrors,
    SourceErrors,
)
from app.main import REQUEST_ID_HEADER, app


# Текст, который обязан остаться в логе и не появиться в ответе: так в
# реальных исключениях выглядят пути на сервере и строки подключения.
SECRET_TEXT = "postgresql://andozai_user:hunter2@10.0.0.7/andozai_db"


class _Payload(BaseModel):
    number: int


# Идентификатор, который получил сам запрос: по нему проверяется, что в ответ
# попал он, а не свежесгенерированный фолбэк из обработчика.
_seen_request_ids: list[str] = []


async def _boom(request: Request):
    _seen_request_ids.append(getattr(request.state, "request_id", None))
    raise RuntimeError(SECRET_TEXT)


async def _api_error():
    raise ApiError(
        status_code=404,
        error_code=SourceErrors.NOT_FOUND,
        detail="Source not found",
    )


async def _external_error():
    raise ExternalServiceError(
        "Ollama is unreachable",
        service="ollama",
        cause=RuntimeError(SECRET_TEXT),
    )


async def _echo(payload: _Payload):
    return {"number": payload.number}


async def _stream_boom():
    """Исключение внутри генератора, то есть уже после отправки заголовков."""

    async def _events():
        yield 'event: token\ndata: {"token": "hi"}\n\n'
        raise RuntimeError(SECRET_TEXT)

    return StreamingResponse(_events(), media_type="text/event-stream")


# Маршруты только на время этого файла: приложение общее для всех тестов,
# и оставлять в нём падающий эндпоинт нельзя.
_TEST_ROUTES = (
    ("/__tests__/boom", _boom, ["GET"]),
    ("/__tests__/api-error", _api_error, ["GET"]),
    ("/__tests__/external-error", _external_error, ["GET"]),
    ("/__tests__/echo", _echo, ["POST"]),
    ("/__tests__/stream-boom", _stream_boom, ["GET"]),
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


class UnhandledExceptionHandlerTests(unittest.TestCase):
    """Что уходит клиенту и что уходит в лог на непойманном исключении."""

    def setUp(self):
        # raise_server_exceptions=False: ServerErrorMiddleware отдаёт ответ, а
        # затем всегда пробрасывает исключение дальше, чтобы сервер мог его
        # залогировать. С настройкой по умолчанию TestClient поднял бы его в
        # тесте вместо того, чтобы вернуть ответ, и проверять было бы нечего.
        self.client = TestClient(app, raise_server_exceptions=False)

    def _call_boom(self, environment="production"):
        _seen_request_ids.clear()
        with patch.object(settings, "ENVIRONMENT", environment):
            with self.assertLogs("app.main", level="ERROR") as captured:
                response = self.client.get("/__tests__/boom")
        return response, "\n".join(captured.output)

    def test_returns_generic_error_code_and_request_id(self):
        response, _ = self._call_boom()

        self.assertEqual(response.status_code, 500)
        body = response.json()
        self.assertEqual(body["error_code"], InternalErrors.INTERNAL_ERROR)
        self.assertEqual(body["error_code"], "internal.error")
        self.assertTrue(body["detail"])
        self.assertTrue(body["request_id"])

    def test_body_carries_no_traceback_and_no_exception_text(self):
        response, _ = self._call_boom()

        text = response.text
        self.assertNotIn("Traceback", text)
        self.assertNotIn(SECRET_TEXT, text)
        # В production не отдаётся даже имя класса.
        self.assertNotIn("RuntimeError", text)
        self.assertNotIn("exception", response.json())
        # Ни имени файла, ни номера строки.
        self.assertNotIn("main.py", text)
        self.assertNotIn("test_global_exception_handler", text)

    def test_traceback_goes_to_log_under_the_same_request_id(self):
        response, log_text = self._call_boom()

        # По идентификатору из ответа пользователя находится строка лога.
        self.assertIn(response.json()["request_id"], log_text)
        self.assertIn("Traceback", log_text)
        self.assertIn("RuntimeError", log_text)
        self.assertIn(SECRET_TEXT, log_text)
        # Метод и путь — как в обработчике ExternalServiceError рядом.
        self.assertIn("/__tests__/boom", log_text)
        self.assertIn("GET", log_text)

    def test_body_reports_the_id_assigned_to_this_request(self):
        """Не свежий uuid из обработчика, а тот, что запрос получил на входе.

        Иначе идентификатор из ответа не совпал бы с тем, что уходит в
        заголовок и в остальные записи по этому же запросу.
        """
        response, _ = self._call_boom()

        self.assertEqual(len(_seen_request_ids), 1)
        self.assertTrue(_seen_request_ids[0])
        self.assertEqual(response.json()["request_id"], _seen_request_ids[0])

    def test_request_id_is_unique_per_request(self):
        first, _ = self._call_boom()
        second, _ = self._call_boom()

        self.assertNotEqual(
            first.json()["request_id"], second.json()["request_id"]
        )

    def test_development_mode_adds_exception_class_only(self):
        response, _ = self._call_boom(environment="development")

        body = response.json()
        # Имя класса — чтобы обработчик не маскировал свежий баг под
        # безликую пятисотку.
        self.assertEqual(body["exception"], "RuntimeError")
        # Форма ответа при этом та же, что в production.
        self.assertEqual(response.status_code, 500)
        self.assertEqual(body["error_code"], InternalErrors.INTERNAL_ERROR)
        self.assertTrue(body["request_id"])
        # Но ни текста исключения, ни трейсбека нет и здесь.
        self.assertNotIn(SECRET_TEXT, response.text)
        self.assertNotIn("Traceback", response.text)

    def test_successful_response_carries_request_id_header(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers.get(REQUEST_ID_HEADER))


class HandlersAboveAreNotShadowedTests(unittest.TestCase):
    """Обработчик на Exception не должен перехватывать уже разобранное.

    Starlette отдаёт обработчик на Exception внешнему слою
    (ServerErrorMiddleware), а всё остальное разбирается слоем внутри
    (ExceptionMiddleware) — эти тесты фиксируют, что порядок именно такой.
    """

    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_unknown_path_is_still_404(self):
        response = self.client.get("/api/v1/definitely-no-such-endpoint")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Not Found")
        self.assertNotIn(InternalErrors.INTERNAL_ERROR, response.text)

    def test_method_not_allowed_is_still_405(self):
        response = self.client.delete("/health")

        self.assertEqual(response.status_code, 405)
        self.assertNotIn(InternalErrors.INTERNAL_ERROR, response.text)

    def test_invalid_body_is_still_422(self):
        response = self.client.post(
            "/__tests__/echo", json={"number": "not-a-number"}
        )

        self.assertEqual(response.status_code, 422)
        self.assertIsInstance(response.json()["detail"], list)
        self.assertNotIn(InternalErrors.INTERNAL_ERROR, response.text)

    def test_api_error_keeps_its_own_error_code(self):
        response = self.client.get("/__tests__/api-error")

        self.assertEqual(response.status_code, 404)
        body = response.json()
        self.assertEqual(body["error_code"], SourceErrors.NOT_FOUND)
        self.assertEqual(body["detail"], "Source not found")

    def test_external_service_error_keeps_its_own_body(self):
        with self.assertLogs("app.main", level="WARNING") as captured:
            response = self.client.get("/__tests__/external-error")

        self.assertEqual(response.status_code, 502)
        body = response.json()
        self.assertEqual(body["service"], "ollama")
        self.assertEqual(body["detail"], "Ollama is unreachable")
        self.assertNotIn(InternalErrors.INTERNAL_ERROR, response.text)
        # Причина по-прежнему уходит в лог, а не в ответ.
        self.assertNotIn(SECRET_TEXT, response.text)
        self.assertIn(SECRET_TEXT, "\n".join(captured.output))


class StreamingResponseTests(unittest.TestCase):
    """Стрим глобальным обработчиком не покрыт — и не должен об него ломаться.

    Исключение внутри генератора возникает после того, как заголовки уже ушли
    клиенту: подменять ответ поздно, статус остаётся 200. Starlette это знает
    (response_started в ServerErrorMiddleware) и тело обработчика не
    отправляет, поэтому в SSE-поток ничего постороннего не попадает.

    Чат на обработчик и не рассчитывает: chat_request_stream ловит исключения
    сам и отдаёт их внутриполосным событием error — этот путь правка не
    затрагивает. Новое здесь только одно: трейсбек из оборвавшегося стрима
    теперь попадает в лог под request_id, а не теряется.
    """

    def test_midstream_exception_does_not_touch_the_stream_body(self):
        client = TestClient(app, raise_server_exceptions=False)

        with self.assertLogs("app.main", level="ERROR") as captured:
            response = client.get("/__tests__/stream-boom")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertNotIn(InternalErrors.INTERNAL_ERROR, response.text)
        self.assertNotIn(SECRET_TEXT, response.text)
        self.assertNotIn("Traceback", response.text)

        log_text = "\n".join(captured.output)
        self.assertIn("Traceback", log_text)
        self.assertIn(SECRET_TEXT, log_text)

    def test_stream_headers_carry_request_id(self):
        client = TestClient(app, raise_server_exceptions=False)

        with self.assertLogs("app.main", level="ERROR"):
            response = client.get("/__tests__/stream-boom")

        # Слой запроса написан ASGI-обёрткой, а не BaseHTTPMiddleware, и
        # заголовок доезжает в том числе на SSE.
        self.assertTrue(response.headers.get(REQUEST_ID_HEADER))


if __name__ == "__main__":
    unittest.main()
