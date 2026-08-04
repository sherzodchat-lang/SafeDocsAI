"""Временный отказ провайдера переживает повтор, постоянный — нет.

Ни базы, ни Ollama здесь нет: проверяется политика, а не сеть.

Повод — приёмка. Ollama вытеснила модель из памяти («predicted to exceed
available memory, evicting») и грузила её заново: 9 с, 20.7 с, 51.9 с. Вызов
не дождался ответа, call_with_one_retry страховал только невалидный JSON, и
ошибка провайдера ушла наверх с первой же попытки — заказ на пятнадцать
слайдов умер на 76% после десяти минут работы. Повтор через минуту почти
наверняка прошёл бы: модель к тому моменту уже была прогрета.

Второе, что здесь проверяется, — что отказ называет себя правдиво. Ollama была
ЖИВА, а пользователь прочёл «Ollama недоступна» и пошёл к администратору
поднимать поднятое.
"""

from __future__ import annotations

import unittest

import httpx
import ollama

from app.core.exceptions import (
    ExternalServiceError,
    ExternalServiceErrorKind,
    PresentationErrors,
)
from app.modules.presentations.constants import LLM_CALL_ATTEMPTS
from app.modules.presentations.service import (
    PresentationGenerationError,
    call_with_one_retry,
    error_code_for,
)
from app.modules.rag.model_manager import ModelManager


def wrap(exc: Exception) -> ExternalServiceError:
    return ModelManager._wrap_provider_error("Ollama", exc)


class ProviderErrorKeepsItsCauseTests(unittest.TestCase):
    """Обёртка провайдера сохраняет ПРИЧИНУ, а не только факт отказа.

    Пока все причины сплющивались в 503/502, политику повтора не на чем было
    строить: «соединение отвергнуто», «ответ не пришёл вовремя» и «нет такой
    модели» были одной и той же ошибкой.
    """

    def test_a_read_timeout_is_a_timeout(self) -> None:
        error = wrap(httpx.ReadTimeout("timed out"))
        self.assertEqual(error.kind, ExternalServiceErrorKind.TIMEOUT)
        self.assertIn("did not answer in time", error.message)

    def test_a_refused_connection_is_unavailability(self) -> None:
        error = wrap(httpx.ConnectError("connection refused"))
        self.assertEqual(error.kind, ExternalServiceErrorKind.UNAVAILABLE)
        self.assertIn("is unavailable", error.message)

    def test_a_connect_timeout_is_unavailability_too(self) -> None:
        # Соединение не установилось вовсе — отвечать было некому. Считать это
        # «модель не успела» значило бы советовать пользователю заказать
        # колоду покороче при выключенном сервисе.
        self.assertEqual(
            wrap(httpx.ConnectTimeout("timed out")).kind,
            ExternalServiceErrorKind.UNAVAILABLE,
        )

    def test_a_server_error_is_transient(self) -> None:
        error = wrap(ollama.ResponseError("boom", 500))
        self.assertEqual(error.kind, ExternalServiceErrorKind.SERVER_ERROR)
        self.assertTrue(error.is_transient)

    def test_a_missing_model_is_not_transient(self) -> None:
        # Осмысленный отказ запросу: второй вызов вернёт тот же ответ, потратив
        # ещё столько же времени.
        error = wrap(ollama.ResponseError("model 'нетакой' not found", 404))
        self.assertEqual(error.kind, ExternalServiceErrorKind.REQUEST_REJECTED)
        self.assertFalse(error.is_transient)

    def test_an_unclassified_cause_is_not_retried(self) -> None:
        # Умолчание — «не повторять»: неизвестно, что именно повторяется.
        error = ExternalServiceError("странное", service="Ollama")
        self.assertEqual(error.kind, ExternalServiceErrorKind.UNKNOWN)
        self.assertFalse(error.is_transient)

    def test_the_http_status_of_the_old_behaviour_is_preserved(self) -> None:
        """503/502 не поехали: их читает HTTP-слой чата и настроек."""
        self.assertEqual(wrap(httpx.ReadTimeout("timed out")).status_code, 503)
        self.assertEqual(wrap(httpx.ConnectError("refused")).status_code, 503)
        self.assertEqual(wrap(ollama.ResponseError("boom", 500)).status_code, 502)


class FlakyModel:
    """Отвечает сколько-то раз отказом, потом — заготовленным ответом."""

    def __init__(self, failures: list[Exception], answer: str = "ответ") -> None:
        self._failures = failures
        self._answer = answer
        self.attempts = 0

    async def chat(self, *, model=None, messages=None, num_ctx=None) -> str:
        self.attempts += 1
        if self._failures:
            raise self._failures.pop(0)
        return self._answer


async def call(model) -> object:
    return await call_with_one_retry(
        model_manager=model,
        model="какая-нибудь",
        messages=[{"role": "user", "content": "вопрос"}],
        validate=lambda raw: raw,
        label="план",
        stage="план презентации",
    )


class TransientFailuresGetASecondCallTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_timeout_on_the_first_attempt_does_not_kill_the_order(self) -> None:
        model = FlakyModel([wrap(httpx.ReadTimeout("timed out"))])
        self.assertEqual(await call(model), "ответ")
        self.assertEqual(model.attempts, 2)

    async def test_a_lost_connection_is_retried_too(self) -> None:
        model = FlakyModel([wrap(httpx.ConnectError("refused"))])
        self.assertEqual(await call(model), "ответ")
        self.assertEqual(model.attempts, 2)

    async def test_a_server_error_is_retried_too(self) -> None:
        model = FlakyModel([wrap(ollama.ResponseError("boom", 503))])
        self.assertEqual(await call(model), "ответ")
        self.assertEqual(model.attempts, 2)

    async def test_a_meaningful_refusal_is_not_retried(self) -> None:
        model = FlakyModel([wrap(ollama.ResponseError("model not found", 404))])
        with self.assertRaises(ExternalServiceError):
            await call(model)
        self.assertEqual(model.attempts, 1, "повтор на постоянном отказе")

    async def test_the_retry_budget_is_the_declared_one(self) -> None:
        """Попыток ровно LLM_CALL_ATTEMPTS — из них выведен потолок джобы."""
        model = FlakyModel([wrap(httpx.ReadTimeout("timed out"))] * 5)
        with self.assertRaises(PresentationGenerationError):
            await call(model)
        self.assertEqual(model.attempts, LLM_CALL_ATTEMPTS)

    async def test_the_prompt_of_the_retry_is_the_original_one(self) -> None:
        """Ответа не было — предъявлять модели нечего.

        Повтор после невалидного JSON получает отвергнутый ответ и претензию
        валидатора; повтор после отказа связи получил бы разговор о несуществующем
        ответе.
        """
        seen: list[list[dict[str, str]]] = []

        class RecordingModel(FlakyModel):
            async def chat(self, *, model=None, messages=None, num_ctx=None) -> str:
                seen.append(messages)
                return await super().chat(
                    model=model, messages=messages, num_ctx=num_ctx
                )

        model = RecordingModel([wrap(httpx.ReadTimeout("timed out"))])
        await call(model)
        self.assertEqual(seen[0], seen[1])


class ExhaustedRetriesTellTheTruthTests(unittest.IsolatedAsyncioTestCase):
    """Код отказа называет то, что случилось на самом деле."""

    async def test_a_timeout_that_survived_the_retries_is_not_unavailability(
        self,
    ) -> None:
        model = FlakyModel([wrap(httpx.ReadTimeout("timed out"))] * 2)
        with self.assertRaises(PresentationGenerationError) as raised:
            await call(model)
        self.assertEqual(raised.exception.error_code, PresentationErrors.LLM_TIMEOUT)
        self.assertIn("не ответила вовремя", str(raised.exception))
        # Стадия остаётся в тексте: без неё отказ на первой минуте неотличим от
        # отказа на предпоследней секции.
        self.assertIn("план презентации", str(raised.exception))

    async def test_a_dead_service_still_says_it_is_dead(self) -> None:
        model = FlakyModel([wrap(httpx.ConnectError("refused"))] * 2)
        with self.assertRaises(PresentationGenerationError) as raised:
            await call(model)
        self.assertEqual(
            raised.exception.error_code, PresentationErrors.OLLAMA_UNAVAILABLE
        )

    def test_ollama_errors_from_other_stages_are_classified_too(self) -> None:
        # Ретривал ходит в Ollama за embedding-ами и приносит ту же обёртку —
        # мимо call_with_one_retry.
        self.assertEqual(
            error_code_for(wrap(httpx.ReadTimeout("timed out"))),
            PresentationErrors.LLM_TIMEOUT,
        )
        self.assertEqual(
            error_code_for(wrap(httpx.ConnectError("refused"))),
            PresentationErrors.OLLAMA_UNAVAILABLE,
        )

    def test_the_two_codes_are_distinct(self) -> None:
        # Один код на оба случая вернул бы ровно тот дефект, из-за которого
        # администратора послали поднимать работающий сервис.
        self.assertNotEqual(
            PresentationErrors.LLM_TIMEOUT, PresentationErrors.OLLAMA_UNAVAILABLE
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
