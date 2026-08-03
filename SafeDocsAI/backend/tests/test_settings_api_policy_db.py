"""PUT /api/v1/settings/: неизвестные ключи, коды отказов и живая кнопка «Сохранить».

Что закрепляем.

  * **Опечатка в имени поля больше не проходит молча.** RuntimeSettingsUpdate
    не задавала model_config, то есть работало умолчание Pydantic v2
    extra="ignore": PUT {"topk": 7} отвечал 200 OK с полным и корректным телом
    настроек, в котором ничего не изменилось. Клиент считает правку
    применённой и уходит — намерение потеряно без следа.
  * **confirm_reindex объявлен в схеме.** Иначе extra="forbid" отверг бы само
    подтверждение переиндексации — единственный ключ, который SettingsPage.jsx
    добавляет к патчу сверх полей формы.
  * **Кнопка «Сохранить» работает через настоящий эндпоинт.** До правки на
    стенде PUT отвечал 400 на любое сохранение: contextual_embedding_model
    проверялся по каталогу Ollama всегда, а его умолчанием была модель,
    которой в Ollama нет.
  * **Машинный код и статус у каждого отказа.** Интерфейс переведён на три
    языка и показывает пользователю свой перевод по error_code, а не
    английский detail. Недоступный каталог моделей — 503 («повторите позже»),
    а не 400 («вы выбрали не то»).

Настоящий PostgreSQL нужен ради подмены deps.get_current_user: раздел закрыт
get_current_active_superuser, и роль берётся у настоящей строки user. Путь к
файлу настроек и каталог моделей подменены — рабочий
backend/data/runtime_settings.json тесты не трогают, а Ollama не поднимается.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Каталог с тестами в sys.path: unittest discover кладёт его туда сам, но при
# запуске `python -m unittest tests.test_settings_api_policy_db` этого не
# происходит.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dbfixtures import DatabaseBackedTestCase  # noqa: E402

from app.api.endpoints.settings import RuntimeSettingsUpdate  # noqa: E402
from app.core.exceptions import SettingsErrors  # noqa: E402
from app.shared.settings.runtime_settings import (  # noqa: E402
    MAX_NUM_CTX,
    RuntimeSettingsService,
)


SETTINGS = "/api/v1/settings/"

CHAT_ON_STAND = "gemma4:26b"
EMBEDDING_ON_STAND = "qwen3-embedding:8b"
# Прежнее умолчание contextual_embedding_model: в Ollama его нет и не было.
MISSING_MODEL = "gemma3:4b"

FAKE_CATALOG = {
    "available_models": [CHAT_ON_STAND, EMBEDDING_ON_STAND],
    "available_chat_models": [CHAT_ON_STAND],
    "available_embedding_models": [EMBEDDING_ON_STAND],
    "ollama_available": True,
    "ollama_error": None,
}

# Каталог, которого нет: Ollama не ответила. Списки пусты не потому, что
# моделей не установлено.
DOWNED_CATALOG = {
    "available_models": [],
    "available_chat_models": [],
    "available_embedding_models": [],
    "ollama_available": False,
    "ollama_error": "Ollama is unavailable",
}


class SettingsApiTestCase(DatabaseBackedTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()

        self.admin = await self.make_user("root", "admin")
        self.as_user(self.admin)

        self._settings_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._settings_dir.cleanup)
        self.settings_path = Path(self._settings_dir.name) / "runtime_settings.json"

        path_patcher = patch.object(
            RuntimeSettingsService, "_settings_path", return_value=self.settings_path
        )
        path_patcher.start()
        self.addCleanup(path_patcher.stop)

        self.catalog_patcher = patch.object(
            RuntimeSettingsService, "model_catalog", return_value=FAKE_CATALOG
        )
        self.catalog_patcher.start()
        self.addCleanup(self._stop_catalog_patcher)

    def _stop_catalog_patcher(self) -> None:
        try:
            self.catalog_patcher.stop()
        except RuntimeError:  # pragma: no cover - уже остановлен тестом
            pass

    def use_downed_catalog(self) -> None:
        """Ollama не отвечает: каталог собрался, но пустым."""
        self.catalog_patcher.stop()
        patcher = patch.object(
            RuntimeSettingsService, "model_catalog", return_value=DOWNED_CATALOG
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_settings_file(self, **values) -> None:
        self.settings_path.write_text(
            json.dumps(values, ensure_ascii=False), encoding="utf-8"
        )

    def saved(self) -> dict:
        return json.loads(self.settings_path.read_text(encoding="utf-8"))


# --- Задача 1. Сохранение работает --------------------------------------


class SaveButtonWorksTests(SettingsApiTestCase):
    async def test_top_k_saves_while_a_disabled_feature_holds_a_junk_model(self):
        """Главная проверка задачи, через настоящий эндпоинт.

        Ровно состояние стенда: контекстное обогащение выключено, в его поле
        лежит модель, которой в Ollama нет. Здесь приходил 400 на любое
        сохранение.
        """
        self.write_settings_file(
            contextual_embedding_enabled=False,
            contextual_embedding_model=MISSING_MODEL,
        )

        response = await self.client.put(SETTINGS, json={"top_k": 7})

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["top_k"], 7)

    async def test_turning_the_feature_on_with_that_model_is_refused(self):
        self.write_settings_file(
            contextual_embedding_enabled=False,
            contextual_embedding_model=MISSING_MODEL,
        )

        response = await self.client.put(
            SETTINGS, json={"contextual_embedding_enabled": True}
        )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(
            response.json()["error_code"], SettingsErrors.MODEL_NOT_INSTALLED
        )

    async def test_turning_the_feature_on_without_a_model_is_refused(self):
        response = await self.client.put(
            SETTINGS, json={"contextual_embedding_enabled": True}
        )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(
            response.json()["error_code"], SettingsErrors.CONTEXTUAL_MODEL_REQUIRED
        )


# --- Задача 3. Неизвестные ключи в теле ---------------------------------


class UnknownKeysTests(SettingsApiTestCase):
    async def test_a_typo_in_a_field_name_is_no_longer_answered_with_200(self):
        response = await self.client.put(SETTINGS, json={"topk": 7})

        self.assertNotEqual(
            response.status_code, 200, "опечатка не должна выглядеть как успех"
        )
        self.assertEqual(response.status_code, 422, response.text)
        self.assertFalse(
            self.settings_path.exists(), "отклонённое тело не должно сохраняться"
        )

    async def test_the_refusal_names_the_offending_key(self):
        """Клиенту нужно показать, ЧТО именно он прислал не так."""
        response = await self.client.put(SETTINGS, json={"topk": 7})

        detail = response.json()["detail"]
        self.assertEqual(detail[0]["type"], "extra_forbidden")
        self.assertIn("topk", detail[0]["loc"])

    async def test_confirm_reindex_is_declared_and_still_accepted(self):
        """Единственный ключ сверх полей формы, который шлёт SettingsPage.jsx.

        Не объяви его в схеме — extra="forbid" отверг бы само подтверждение, и
        embedding-модель стало бы невозможно сменить вообще.
        """
        response = await self.client.put(
            SETTINGS, json={"top_k": 7, "confirm_reindex": True}
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["top_k"], 7)
        self.assertNotIn("confirm_reindex", self.saved())

    async def test_every_field_the_form_sends_is_declared(self):
        """Список — из frontend/src/pages/SettingsPage.jsx (TEXT_FIELDS,
        BOOLEAN_FIELDS, NUMBER_FIELDS плюс confirm_reindex). С extra="forbid"
        любое расхождение с ним — отказ на живом сохранении."""
        sent_by_the_form = {
            "chat_model",
            "embedding_model",
            "contextual_embedding_model",
            "reranker_model",
            "enable_condense_query",
            "contextual_embedding_enabled",
            "reranker_enabled",
            "top_k",
            "chat_model_num_ctx",
            "contextual_embedding_num_ctx",
            "confirm_reindex",
        }

        self.assertEqual(
            sent_by_the_form - set(RuntimeSettingsUpdate.model_fields), set()
        )

    async def test_a_full_form_patch_is_accepted_as_a_whole(self):
        """То же самое живьём: тело со всеми полями формы обязано проходить."""
        response = await self.client.put(
            SETTINGS,
            json={
                "chat_model": CHAT_ON_STAND,
                "embedding_model": EMBEDDING_ON_STAND,
                "contextual_embedding_model": CHAT_ON_STAND,
                "reranker_model": "gemma4:e4b",
                "enable_condense_query": True,
                "contextual_embedding_enabled": True,
                "reranker_enabled": False,
                "top_k": 5,
                "chat_model_num_ctx": 20000,
                "contextual_embedding_num_ctx": 8192,
                "confirm_reindex": True,
            },
        )

        self.assertEqual(response.status_code, 200, response.text)

    async def test_an_unknown_key_in_the_reset_body_is_refused_too(self):
        """Опечатка в единственном ключе сброса означала бы сброс БЕЗ
        подтверждения — с отказом, непонятным админу."""
        response = await self.client.post(
            "/api/v1/settings/reset", json={"confirm_reindx": True}
        )

        self.assertEqual(response.status_code, 422, response.text)


# --- Задачи 2 и 4. Коды и статусы отказов -------------------------------


class RefusalCodesTests(SettingsApiTestCase):
    async def test_an_unknown_domain_profile_is_refused_with_its_own_code(self):
        """Отвечало 200 и подменяло значение на "tax": админ видел успех, а
        правила ответов ассистента менялись на другие."""
        response = await self.client.put(
            SETTINGS, json={"default_domain_profile": "banking"}
        )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(
            response.json()["error_code"], SettingsErrors.UNSUPPORTED_DOMAIN_PROFILE
        )

    async def test_a_context_window_that_does_not_fit_is_refused(self):
        response = await self.client.put(SETTINGS, json={"chat_model_num_ctx": 262144})

        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(
            response.json()["error_code"], SettingsErrors.VALUE_OUT_OF_RANGE
        )
        self.assertIn(str(MAX_NUM_CTX), response.json()["detail"])

    async def test_the_upper_bound_itself_is_accepted(self):
        response = await self.client.put(
            SETTINGS, json={"chat_model_num_ctx": MAX_NUM_CTX}
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["chat_model_num_ctx"], MAX_NUM_CTX)

    async def test_zero_is_refused_instead_of_quietly_becoming_2048(self):
        response = await self.client.put(SETTINGS, json={"chat_model_num_ctx": 0})

        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(
            response.json()["error_code"], SettingsErrors.VALUE_OUT_OF_RANGE
        )


# --- Задача 5. Ollama лежит ---------------------------------------------


class DownedOllamaTests(SettingsApiTestCase):
    async def test_settings_still_open_when_the_catalog_is_empty(self):
        self.use_downed_catalog()

        response = await self.client.get(SETTINGS)

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertFalse(body["ollama_available"])
        self.assertEqual(body["ollama_error"], "Ollama is unavailable")

    async def test_choosing_a_model_answers_503_and_not_400(self):
        """400 обвинял бы админа в чужой аварии: модель может стоять на месте.

        Ответ «повторите тот же запрос позже» — это 503, и клиенту нужен
        отдельный код, чтобы не предлагать `ollama pull`.
        """
        self.use_downed_catalog()

        response = await self.client.put(SETTINGS, json={"chat_model": CHAT_ON_STAND})

        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(
            response.json()["error_code"], SettingsErrors.MODEL_CATALOG_UNAVAILABLE
        )

    async def test_settings_without_models_are_still_saved(self):
        """Лежащая Ollama не должна мешать править top_k."""
        self.use_downed_catalog()

        response = await self.client.put(SETTINGS, json={"top_k": 7})

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["top_k"], 7)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
