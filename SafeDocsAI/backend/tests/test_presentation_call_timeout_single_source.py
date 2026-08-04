"""Потолок вызова модели объявлен один раз и доезжает до HTTP-клиента.

Тот же приём, что в test_presentation_limits_single_source.py, и по той же
причине: величина, объявленная в одном месте и продублированная в другом,
расходится молча — и обнаруживается позже всего.

История здесь не гипотетическая. LLM_CALL_TIMEOUT = 300 калибровали по
замерам приёмки (худший наблюдённый вызов — план за 69.9 с), а HTTP-клиент
Ollama строился в ModelManager с чатовскими settings.OLLAMA_TIMEOUT_SECONDS =
120. Клиент сдавался вдвое раньше, поэтому откалиброванные 300 с не
достигались НИКОГДА, а wait_for вокруг вызова не срабатывал ни разу. В логе
это выглядело так:

    вызовов модели 14 (план 1, слайды 13, повторных 1) -> p50 35.2с,
    p90 68.2с, max 120.0с, суммарно 619.1с; потолок вызова 300с

«max 120.0с» — ровное число там, где стоят замеры: не длительность, а чужой
потолок. Заказ на пятнадцать слайдов умирал на 76% после десяти минут работы.

Чего этот файл НЕ требует: одинаковых бюджетов у чата и у презентаций. Они
разные намеренно — за чатом ждёт человек (120 с — интерактивный бюджет), а
презентация идёт фоном. Поэтому проверки написаны так, чтобы не выродиться в
тавтологию, если однажды эти два числа случайно совпадут: связь проверяется
подменой одного из них на заведомо чужое значение.

Числа в этом файле не пишутся: все проверки спрашивают их у констант.
"""

from __future__ import annotations

import asyncio
import inspect
import unittest
from unittest.mock import patch

from app.modules.presentations import service as presentation_service
from app.modules.presentations.constants import (
    LLM_CALL_ATTEMPTS,
    LLM_CALL_TIMEOUT,
    LLM_CALL_WATCHDOG_MARGIN,
    LLM_CALL_WATCHDOG_TIMEOUT,
    PRESENTATION_NUM_CTX,
)
from app.modules.presentations.service import (
    build_presentation_model_manager,
    call_with_one_retry,
)
from app.modules.rag.model_manager import ModelManager
from app.shared.settings.config import settings

# Значение, которого нет ни у одного из двух бюджетов: подставляется вместо
# чатовского, чтобы отличить «презентации взяли свою константу» от «оба числа
# сегодня совпали».
FOREIGN_TIMEOUT = 7.0


def http_client_timeouts(manager: ModelManager) -> list[float]:
    """Что реально уехало в HTTP-клиенты — синхронный и асинхронный.

    Проверять свойство ModelManager.timeout недостаточно: оно ответит верно и
    в тот день, когда значение перестанет доходить до самого клиента, — а
    сдаётся раньше срока именно клиент. Отсюда и обращение к потрохам httpx:
    другого способа спросить у клиента его бюджет нет.
    """
    values: list[float] = []
    for client in (manager._ollama_client, manager._ollama_async_client):
        timeout = client._client.timeout
        # httpx хранит четыре бюджета; ollama выставляет все из одного числа.
        # Читаем read: «модель думает» — это именно ожидание ответа.
        values.append(timeout.read)
    return values


class PresentationClientUsesTheCalibratedCeilingTests(unittest.TestCase):
    """Клиент пути презентаций живёт по LLM_CALL_TIMEOUT."""

    def test_the_factory_hands_the_calibrated_ceiling_to_the_client(self) -> None:
        manager = build_presentation_model_manager()
        self.assertEqual(manager.timeout, float(LLM_CALL_TIMEOUT))
        for value in http_client_timeouts(manager):
            self.assertEqual(
                value,
                float(LLM_CALL_TIMEOUT),
                "HTTP-клиент презентаций сдастся не на том рубеже, на котором "
                "калибровали потолок вызова",
            )

    def test_the_chat_path_keeps_its_own_interactive_budget(self) -> None:
        # Умолчание ModelManager — чатовское. Поднять его до LLM_CALL_TIMEOUT
        # ради фоновой джобы значило бы заставить зависший запрос человека
        # висеть впятеро дольше.
        manager = ModelManager()
        self.assertEqual(manager.timeout, float(settings.OLLAMA_TIMEOUT_SECONDS))
        for value in http_client_timeouts(manager):
            self.assertEqual(value, float(settings.OLLAMA_TIMEOUT_SECONDS))

    def test_the_two_budgets_are_independent(self) -> None:
        """Смена чатовского бюджета не трогает презентационный, и наоборот.

        Ради этой проверки файл и заведён. Равенство чисел ничего не
        доказывает: пока презентации берут бюджет у settings, они «совпадают с
        LLM_CALL_TIMEOUT» ровно до первой правки чатовской настройки — а
        именно так и было в тот день, когда заказ умер на 76%.
        """
        self.assertNotEqual(
            FOREIGN_TIMEOUT,
            float(LLM_CALL_TIMEOUT),
            "подменное значение совпало с проверяемым — проверка выродилась",
        )
        with patch.object(settings, "OLLAMA_TIMEOUT_SECONDS", FOREIGN_TIMEOUT):
            presentation_manager = build_presentation_model_manager()
            chat_manager = ModelManager()

        self.assertEqual(
            presentation_manager.timeout,
            float(LLM_CALL_TIMEOUT),
            "путь презентаций снова берёт бюджет у чатовской настройки",
        )
        for value in http_client_timeouts(presentation_manager):
            self.assertEqual(value, float(LLM_CALL_TIMEOUT))

        self.assertEqual(
            chat_manager.timeout,
            FOREIGN_TIMEOUT,
            "чатовский путь перестал слушать OLLAMA_TIMEOUT_SECONDS",
        )

    def test_the_pipeline_builds_its_manager_through_the_factory(self) -> None:
        """Фабрика верна ровно настолько, насколько ею пользуются.

        ModelManager() по месту в пайплайне вернул бы чатовский бюджет, и все
        проверки выше остались бы зелёными.
        """
        source = inspect.getsource(presentation_service.generate_presentation)
        self.assertIn("build_presentation_model_manager()", source)
        self.assertNotIn("ModelManager()", source)


class WatchdogStandsBehindTheClientTests(unittest.TestCase):
    """wait_for срабатывает ПОЗЖЕ клиента — иначе он его заслоняет."""

    def test_the_margin_is_real(self) -> None:
        self.assertGreater(LLM_CALL_WATCHDOG_MARGIN, 0)

    def test_the_watchdog_is_derived_from_the_call_ceiling(self) -> None:
        # Третьего числа в этой паре быть не должно: тому, что выведено,
        # дрейфовать не от чего.
        self.assertEqual(
            LLM_CALL_WATCHDOG_TIMEOUT, LLM_CALL_TIMEOUT + LLM_CALL_WATCHDOG_MARGIN
        )

    def test_the_watchdog_fires_strictly_later_than_the_client(self) -> None:
        """Иначе страховка станет первым эшелоном.

        Клиент, сдаваясь, приносит причину («не дождались ответа», «связь не
        встала»), по которой видно, повторять ли вызов и что сказать
        пользователю. wait_for умеет одно — снять корутину; из отмены не видно
        ни стадии, ни причины. Поменяй их местами — и вернётся ровно тот
        отказ, который на приёмке выглядел как «Ollama недоступна» при живой
        Ollama.
        """
        self.assertGreater(LLM_CALL_WATCHDOG_TIMEOUT, LLM_CALL_TIMEOUT)


class CallSiteUsesTheWatchdogTests(unittest.IsolatedAsyncioTestCase):
    """Проверяется само место вызова, а не только константы."""

    async def test_wait_for_gets_the_watchdog_and_not_the_call_ceiling(self) -> None:
        recorded: list[float] = []
        original_wait_for = asyncio.wait_for

        async def recording_wait_for(awaitable, timeout):
            recorded.append(timeout)
            return await original_wait_for(awaitable, timeout)

        class InstantModel:
            def __init__(self) -> None:
                self.num_ctx: int | None = None

            async def chat(self, *, model=None, messages=None, num_ctx=None) -> str:
                self.num_ctx = num_ctx
                return "ответ"

        model = InstantModel()
        with patch.object(asyncio, "wait_for", recording_wait_for):
            result = await call_with_one_retry(
                model_manager=model,
                model="какая-нибудь",
                messages=[{"role": "user", "content": "вопрос"}],
                validate=lambda raw: raw,
                label="план",
                stage="план",
            )

        self.assertEqual(result, "ответ")
        self.assertEqual(recorded, [LLM_CALL_WATCHDOG_TIMEOUT])
        # Заодно: окно модели у пайплайна своё, и оно тоже приезжает из
        # константы — вызов, ушедший с чужим num_ctx, молча теряет хвост
        # промпта.
        self.assertEqual(model.num_ctx, PRESENTATION_NUM_CTX)

    async def test_every_attempt_gets_its_own_full_watchdog(self) -> None:
        """Повтор — отдельный вызов, а не остаток первого.

        Общий на две попытки бюджет означал бы, что медленная первая попытка
        съедает время второй.
        """
        recorded: list[float] = []
        original_wait_for = asyncio.wait_for

        async def recording_wait_for(awaitable, timeout):
            recorded.append(timeout)
            return await original_wait_for(awaitable, timeout)

        from app.modules.presentations.llm_schemas import LlmResponseError

        class InstantModel:
            async def chat(self, *, model=None, messages=None, num_ctx=None) -> str:
                return "мусор"

        def always_reject(raw: str):
            raise LlmResponseError("не json")

        with patch.object(asyncio, "wait_for", recording_wait_for):
            with self.assertRaises(presentation_service.PresentationGenerationError):
                await call_with_one_retry(
                    model_manager=InstantModel(),
                    model="какая-нибудь",
                    messages=[{"role": "user", "content": "вопрос"}],
                    validate=always_reject,
                    label="план",
                    stage="план",
                )

        self.assertEqual(
            recorded, [LLM_CALL_WATCHDOG_TIMEOUT] * LLM_CALL_ATTEMPTS
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
