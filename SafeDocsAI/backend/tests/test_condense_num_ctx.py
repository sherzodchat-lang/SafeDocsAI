"""Конденсация и генерация должны просить у Ollama одно и то же окно.

num_ctx входит в конфигурацию раннера: другой размер окна на соседнем вызове
заставляет Ollama выгрузить и заново поднять модель. С дефолтными 12288 у
конденсации и chat_model_num_ctx у генерации каждый вопрос перезагружал
20-гигабайтную модель дважды (замер: пять стартов llama-server с чередованием
-c 12288/-c 30720 на два вопроса).
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from app.modules.rag.generation_service import GenerationService


class CondenseNumCtxTests(unittest.TestCase):
    def test_condense_uses_chat_model_num_ctx(self):
        service = GenerationService()
        service.model_manager.chat = AsyncMock(return_value="уточнённый запрос")
        history = [
            {"role": "user", "content": "что такое ндс?"},
            {"role": "assistant", "content": "налог на добавленную стоимость"},
        ]
        with patch(
            "app.services.runtime_settings_service.RuntimeSettingsService.get_settings",
            return_value={"chat_model_num_ctx": 31337},
        ):
            asyncio.run(service.condense_query("а ставка какая?", history))
        kwargs = service.model_manager.chat.await_args.kwargs
        self.assertEqual(kwargs.get("num_ctx"), 31337)

    def test_condense_without_history_makes_no_call(self):
        # Без истории конденсация не нужна — и обращения к модели быть не должно.
        service = GenerationService()
        service.model_manager.chat = AsyncMock(return_value="ответ")
        result = asyncio.run(service.condense_query("вопрос", []))
        self.assertEqual(result, "вопрос")
        service.model_manager.chat.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
