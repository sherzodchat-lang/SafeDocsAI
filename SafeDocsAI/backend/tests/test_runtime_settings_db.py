"""Настройки админ-панели: /api/v1/settings/ на чистой установке.

Что закрепляем.

  * GET /api/v1/settings/ отвечает 200, когда backend/data/runtime_settings.json
    ещё не создан. Раньше get_settings() в этом случае уходил коротким
    `return dict(DEFAULTS)`, а в DEFAULTS нет ключа "model" — обработчик читал
    его квадратными скобками и падал с KeyError. Эндпоинт не работал вообще:
    до первого сохранения настроек сохранять их было неоткуда.
  * PUT /api/v1/settings/ ломался симметрично — ответ собирается из того же
    словаря, и патч без chat_model/model не добавлял туда ключ.
  * "model" — устаревшее имя chat_model (его же читают чат и ask через
    `.get("chat_model") or .get("model")`), поэтому он выводится из
    chat_model, а не живёт в DEFAULTS отдельной строкой: два независимых
    умолчания одного значения разъехались бы, и админ-панель показывала бы
    модель, с которой никто не работает.
  * Остальные поля обработчика читаются из того же словаря, поэтому
    проверяются разом: ответ обязан собраться при полностью отсутствующем
    файле настроек.

Настоящий PostgreSQL нужен здесь ради подмены deps.get_current_user: эндпоинт
закрыт get_current_active_superuser, и роль берётся у настоящей строки user.

Путь к файлу настроек подменяется во временный каталог: рабочий
backend/data/runtime_settings.json тесты не создают и не трогают, иначе
следующий запуск сервиса поднялся бы с настройками из прогона тестов.
Каталог моделей тоже подменён — иначе model_catalog() ходил бы в Ollama.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Каталог с тестами в sys.path: unittest discover кладёт его туда сам, но при
# запуске `python -m unittest tests.test_runtime_settings_db` этого не происходит.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dbfixtures import DatabaseBackedTestCase  # noqa: E402

from app.shared.settings.runtime_settings import RuntimeSettingsService  # noqa: E402


SETTINGS = "/api/v1/settings/"

FAKE_CATALOG = {
    "available_models": ["gemma3n:e4b", "nomic-embed-text"],
    "available_chat_models": ["gemma3n:e4b", "nomic-embed-text"],
    "available_embedding_models": ["gemma3n:e4b", "nomic-embed-text"],
    "ollama_available": True,
    "ollama_error": None,
}


class RuntimeSettingsEndpointTestCase(DatabaseBackedTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()

        self.admin = await self.make_user("root", "admin")
        self.as_user(self.admin)

        # Файл настроек — во временном каталоге и намеренно НЕ создан: это и
        # есть состояние чистой установки, на котором эндпоинт падал.
        self._settings_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._settings_dir.cleanup)
        self.settings_path = Path(self._settings_dir.name) / "runtime_settings.json"

        path_patcher = patch.object(
            RuntimeSettingsService, "_settings_path", return_value=self.settings_path
        )
        path_patcher.start()
        self.addCleanup(path_patcher.stop)

        catalog_patcher = patch.object(
            RuntimeSettingsService, "model_catalog", return_value=FAKE_CATALOG
        )
        catalog_patcher.start()
        self.addCleanup(catalog_patcher.stop)


class GetRuntimeSettingsTests(RuntimeSettingsEndpointTestCase):
    async def test_settings_open_without_a_settings_file(self):
        self.assertFalse(self.settings_path.exists(), "файл настроек не должен быть создан")

        response = await self.client.get(SETTINGS)

        self.assertEqual(response.status_code, 200, response.text)

    async def test_model_falls_back_to_chat_model(self):
        response = await self.client.get(SETTINGS)

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["model"], body["chat_model"])
        self.assertEqual(
            body["chat_model"], RuntimeSettingsService.DEFAULTS["chat_model"]
        )

    async def test_every_field_of_the_response_is_filled_from_defaults(self):
        """Один и тот же дефект редко живёт в одиночку: обработчик читает из
        этого словаря семь ключей квадратными скобками."""
        response = await self.client.get(SETTINGS)

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        defaults = RuntimeSettingsService.DEFAULTS
        for key in (
            "chat_model",
            "embedding_model",
            "enable_condense_query",
            "retrieval_top_k",
            "top_k",
            "default_domain_profile",
        ):
            self.assertEqual(body[key], defaults[key], key)
        self.assertEqual(body["available_chat_models"], FAKE_CATALOG["available_chat_models"])
        self.assertIn("tax", body["available_domain_profiles"])

    async def test_reading_settings_does_not_create_the_file(self):
        """GET — чтение: файл появляется только при сохранении."""
        await self.client.get(SETTINGS)

        self.assertFalse(self.settings_path.exists())

    async def test_broken_settings_file_falls_back_to_defaults(self):
        self.settings_path.write_text("{ это не json", encoding="utf-8")

        response = await self.client.get(SETTINGS)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["model"], response.json()["chat_model"])

    async def test_non_admin_still_gets_403(self):
        self.as_user(await self.make_user("plain", "user"))

        response = await self.client.get(SETTINGS)

        self.assertEqual(response.status_code, 403, response.text)


class UpdateRuntimeSettingsTests(RuntimeSettingsEndpointTestCase):
    async def test_saving_without_model_in_the_payload_returns_the_full_response(self):
        """Симметричный случай: ответ PUT собирается из того же словаря, и
        патч, не упоминающий модель, ключа "model" туда не добавлял."""
        response = await self.client.put(SETTINGS, json={"top_k": 7})

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["top_k"], 7)
        self.assertEqual(body["model"], body["chat_model"])

    async def test_saved_settings_survive_a_reread(self):
        put = await self.client.put(SETTINGS, json={"top_k": 7})
        self.assertEqual(put.status_code, 200, put.text)

        response = await self.client.get(SETTINGS)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["top_k"], 7)
        self.assertTrue(self.settings_path.exists())

    async def test_changing_the_chat_model_moves_the_legacy_alias_with_it(self):
        response = await self.client.put(SETTINGS, json={"chat_model": "nomic-embed-text"})

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["chat_model"], "nomic-embed-text")
        self.assertEqual(body["model"], "nomic-embed-text")

    async def test_unknown_chat_model_is_rejected(self):
        response = await self.client.put(SETTINGS, json={"chat_model": "нет-такой-модели"})

        self.assertEqual(response.status_code, 400, response.text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
