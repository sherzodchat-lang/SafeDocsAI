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

Третье — ЦЕНА повтора. После таймаута он идёт с паузой, после остальных
временных отказов — немедленно. Пауза здесь никогда не выжидается по-настоящему:
asyncio.sleep подменяется, и проверяется его АРГУМЕНТ. Тест, который честно
спит тридцать секунд, проверяет не политику повтора, а терпение того, кто
запустил набор.
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

import httpx
import ollama

from app.core.exceptions import (
    ExternalServiceError,
    ExternalServiceErrorKind,
    PresentationErrors,
)
from app.modules.presentations.constants import (
    LLM_CALL_ATTEMPTS,
    LLM_RETRY_PAUSE_AFTER_TIMEOUT,
)
from app.modules.presentations.service import (
    CallTimings,
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


async def call(model, *, validate=lambda raw: raw, timings=None) -> object:
    return await call_with_one_retry(
        model_manager=model,
        model="какая-нибудь",
        messages=[{"role": "user", "content": "вопрос"}],
        validate=validate,
        label="план",
        stage="план презентации",
        timings=timings,
    )


class RetryPolicyTestCase(unittest.IsolatedAsyncioTestCase):
    """Общая обвязка: пауза перед повтором записывается, а не выжидается.

    Подменяется asyncio.sleep, а не константа паузы: проверять надо, что код
    просит подождать И СКОЛЬКО, а не что он умеет не спать при нуле. Заодно эта
    подмена возвращает набору его прежнюю скорость — без неё каждый повтор
    после таймаута стоил бы полминуты стены.
    """

    def setUp(self) -> None:
        self.slept: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            self.slept.append(seconds)

        patcher = patch.object(asyncio, "sleep", fake_sleep)
        patcher.start()
        self.addCleanup(patcher.stop)


class TransientFailuresGetASecondCallTests(RetryPolicyTestCase):
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


class ThePauseIsOnlyForTimeoutsTests(RetryPolicyTestCase):
    """Повтор ждёт только там, где симптом — медленность.

    Разница не косметическая. После таймаута немедленный повтор возвращается в
    тот же затор: на приёмке Ollama грузила вытесненную модель 9 с, 20.7 с и
    51.9 с, и вторая попытка, начатая сразу, сгорала так же, как первая, —
    повтор формально был, а толку от него не было. После отказа СОЕДИНЕНИЯ и
    после невалидного JSON ждать нечего: там либо отвечать некому, либо ответ
    уже получен и он негодный. Пауза в этих двух случаях только замедлила бы
    детект.
    """

    async def test_a_timeout_waits_before_the_second_call(self) -> None:
        model = FlakyModel([wrap(httpx.ReadTimeout("timed out"))])

        self.assertEqual(await call(model), "ответ")

        self.assertEqual(model.attempts, 2)
        self.assertEqual(self.slept, [LLM_RETRY_PAUSE_AFTER_TIMEOUT])

    async def test_a_refused_connection_is_retried_immediately(self) -> None:
        model = FlakyModel([wrap(httpx.ConnectError("refused"))])

        self.assertEqual(await call(model), "ответ")

        self.assertEqual(model.attempts, 2)
        self.assertEqual(self.slept, [], "повтор после отказа связи ждал впустую")

    async def test_an_invalid_answer_is_retried_immediately(self) -> None:
        """Модель ответила сразу и ответила мусором — ждать нечего."""
        from app.modules.presentations.llm_schemas import LlmResponseError

        rejected: list[str] = []

        def reject_once(raw: str) -> str:
            rejected.append(raw)
            if len(rejected) == 1:
                raise LlmResponseError("не json")
            return raw

        model = FlakyModel([], answer="ответ")

        self.assertEqual(await call(model, validate=reject_once), "ответ")

        self.assertEqual(model.attempts, 2)
        self.assertEqual(self.slept, [])

    async def test_a_server_error_is_retried_immediately_too(self) -> None:
        # 5xx — беда на стороне сервиса, а не затор: он либо уже починился, либо
        # ответит тем же самым, и узнать это лучше сразу.
        model = FlakyModel([wrap(ollama.ResponseError("boom", 503))])

        self.assertEqual(await call(model), "ответ")

        self.assertEqual(self.slept, [])

    async def test_the_last_attempt_does_not_pause_before_giving_up(self) -> None:
        """Пауза перед несуществующей попыткой — чистая задержка отказа.

        Ждать после ПОСЛЕДНЕГО таймаута незачем: повторять уже нечего, а заказ
        всё это время стоит в 'generating' и морочит голову пользователю.
        """
        model = FlakyModel([wrap(httpx.ReadTimeout("timed out"))] * LLM_CALL_ATTEMPTS)

        with self.assertRaises(PresentationGenerationError):
            await call(model)

        self.assertEqual(model.attempts, LLM_CALL_ATTEMPTS)
        self.assertEqual(len(self.slept), LLM_CALL_ATTEMPTS - 1)

    async def test_the_pause_is_not_counted_as_call_time(self) -> None:
        """Пауза — не длительность вызова, и в замеры попадать не должна.

        Попади она туда, p50 и max поехали бы на тридцать секунд, а ровное
        число в графе max перестало бы читаться как «сработал чей-то потолок» —
        то есть сломался бы единственный признак, по которому в логе видно, что
        заказ упёрся в границу, а не был медленным.

        Пауза здесь подменена настоящим ожиданием, но КОРОТКИМ: без ожидания
        разницу «внутри замера или снаружи» не увидеть вовсе, а настоящие
        тридцать секунд проверяли бы терпение того, кто запустил набор.
        Порог взят вчетверо меньше подменной паузы и на два порядка больше
        самого вызова (подменённая модель отвечает мгновенно), так что ни та,
        ни другая сторона не зависит от загрузки машины.
        """
        import time

        pause_seconds = 0.2

        async def slow_fake_sleep(seconds: float) -> None:
            self.slept.append(seconds)
            time.sleep(pause_seconds)

        timings = CallTimings()
        model = FlakyModel([wrap(httpx.ReadTimeout("timed out"))])
        with patch.object(asyncio, "sleep", slow_fake_sleep):
            await call(model, timings=timings)

        self.assertEqual(self.slept, [LLM_RETRY_PAUSE_AFTER_TIMEOUT])
        self.assertEqual(len(timings.durations), 2)
        for seconds in timings.durations:
            self.assertLess(
                seconds,
                pause_seconds / 4,
                "пауза перед повтором приписана вызову модели",
            )


class UnclassifiedFailuresAreLoudAndCountedTests(RetryPolicyTestCase):
    """Причина не разобрана — повтора нет, и это обязано быть видно.

    Такой отказ стоит заказу первой же попытки, а выглядит как рядовая ошибка
    провайдера. Счётчик в статистической строке джобы — то место, где «одна
    странная ошибка» превращается в «классификатор устарел».
    """

    async def test_an_unknown_failure_is_not_retried_and_gets_counted(self) -> None:
        timings = CallTimings()
        model = FlakyModel([ExternalServiceError("странное", service="Ollama")])

        with self.assertRaises(ExternalServiceError):
            await call(model, timings=timings)

        self.assertEqual(model.attempts, 1, "неразобранную причину повторили")
        self.assertEqual(timings.unclassified, 1)
        self.assertIn("неклассифицированных 1", timings.summary())

    async def test_a_classified_failure_does_not_touch_the_counter(self) -> None:
        timings = CallTimings()
        model = FlakyModel([wrap(httpx.ReadTimeout("timed out"))])

        await call(model, timings=timings)

        self.assertEqual(timings.unclassified, 0)
        self.assertNotIn("неклассиф", timings.summary())


class ExhaustedRetriesTellTheTruthTests(RetryPolicyTestCase):
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
