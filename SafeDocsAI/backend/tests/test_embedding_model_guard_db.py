"""Смена embedding-модели — операция с последствиями, а не настройка.

Что закрепляем.

  * Имя коллекции ChromaDB выводится из embedding-модели, поэтому её смена
    мгновенно уводит поиск в коллекцию, которую никто не заполнял: система
    отвечает так, будто документов нет вовсе. Вернуть их может только полная
    переиндексация. Поэтому PUT, меняющий embedding_model, требует
    подтверждения в теле (confirm_reindex), и отказ приходит машинным кодом.
    Guard стоит на сервере: клиентских путей к настройкам несколько, и ни
    один из них — включая случайный — не должен задеть эту настройку заодно
    с соседними.
  * Флаг reindex_required не читал НИКТО: в репозитории было ровно одно его
    вхождение — сама строка записи. В ответе API поля не было, и интерфейс не
    мог показать даже предупреждения.
  * Снять флаг может только полностью успешная переиндексация
    (POST /api/v1/documents/reindex). Частичная ("partial") его не трогает:
    у непереиндексированных документов чанки уже удалены, а новых векторов
    нет — индекс неполон, и флаг остаётся единственным напоминанием.
  * Сброс к умолчаниям (POST /api/v1/settings/reset) — отдельная находка
    аудита: вернуться назад было нельзя вообще. Он админский и подтверждается
    так же, потому что возвращает к умолчанию и embedding_model.

Настоящий PostgreSQL нужен ради подмены deps.get_current_user: раздел закрыт
get_current_active_superuser, и роль берётся у настоящей строки user.

Файл настроек подменён во временный каталог — рабочий
backend/data/runtime_settings.json тесты не трогают. Каталог моделей подменён
тоже: иначе валидация ходила бы в Ollama.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

# Каталог с тестами в sys.path: unittest discover кладёт его туда сам, но при
# запуске `python -m unittest tests.test_embedding_model_guard_db` — нет.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dbfixtures import DatabaseBackedTestCase  # noqa: E402

from app.core.exceptions import SettingsErrors  # noqa: E402
from app.modules.documents import DocumentModuleService  # noqa: E402
from app.shared.settings.config import settings as app_settings  # noqa: E402
from app.shared.settings.runtime_settings import RuntimeSettingsService  # noqa: E402


SETTINGS = "/api/v1/settings/"
RESET = "/api/v1/settings/reset"
REINDEX = "/api/v1/documents/reindex"

CHAT_MODEL = "gemma4:26b"
# Исходная модель приходит из ПЕРЕМЕННОЙ ОКРУЖЕНИЯ: умолчания у embedding-модели
# в коде больше нет, порядок разрешения — файл настроек ->
# OLLAMA_MODEL_EMBEDDING -> отказ. Переменная подменяется в asyncSetUp, иначе
# проверки зависели бы от окружения машины, а на чистой машине (модель не
# задана нигде) менять было бы не с чего — и guard проверять стало бы не на чем.
DEFAULT_EMBEDDING = "qwen3-embedding:8b"
OTHER_EMBEDDING = "bge-m3"

FAKE_CATALOG = {
    "available_models": [CHAT_MODEL, DEFAULT_EMBEDDING, OTHER_EMBEDDING],
    "available_chat_models": [CHAT_MODEL],
    "available_embedding_models": [DEFAULT_EMBEDDING, OTHER_EMBEDDING],
    "ollama_available": True,
    "ollama_error": None,
}


class SettingsApiTestCase(DatabaseBackedTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()

        self.admin = await self.make_user("root", "admin")
        self.as_user(self.admin)

        env_patcher = patch.object(
            app_settings, "OLLAMA_MODEL_EMBEDDING", DEFAULT_EMBEDDING
        )
        env_patcher.start()
        self.addCleanup(env_patcher.stop)

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

    def assertRefused(self, response, status_code: int, error_code: str) -> None:
        self.assertEqual(response.status_code, status_code, response.text)
        self.assertEqual(response.json().get("error_code"), error_code, response.text)

    async def switch_embedding_model(self, model: str = OTHER_EMBEDDING):
        """Подтверждённая смена модели — исходное состояние для проверок ниже."""
        response = await self.client.put(
            SETTINGS, json={"embedding_model": model, "confirm_reindex": True}
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response


# --- Guard на смену модели ----------------------------------------------


class EmbeddingModelGuardTests(SettingsApiTestCase):
    async def test_switching_the_model_without_confirmation_is_refused(self):
        response = await self.client.put(
            SETTINGS, json={"embedding_model": OTHER_EMBEDDING}
        )

        self.assertRefused(
            response, 409, SettingsErrors.REINDEX_CONFIRMATION_REQUIRED
        )
        self.assertFalse(
            self.settings_path.exists(), "отклонённый патч не должен сохраняться"
        )
        self.assertEqual(
            RuntimeSettingsService.get_settings()["embedding_model"],
            DEFAULT_EMBEDDING,
        )

    async def test_a_confirmed_switch_goes_through_and_flags_a_reindex(self):
        response = await self.switch_embedding_model()

        body = response.json()
        self.assertEqual(body["embedding_model"], OTHER_EMBEDDING)
        self.assertTrue(body["reindex_required"])

    async def test_the_guard_does_not_stand_in_the_way_of_other_settings(self):
        """Подтверждение — цена смены модели, а не каждого сохранения."""
        response = await self.client.put(SETTINGS, json={"top_k": 7})

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["top_k"], 7)
        self.assertFalse(response.json()["reindex_required"])

    async def test_saving_the_same_model_again_needs_no_confirmation(self):
        """Повторное сохранение той же модели ничего не меняет.

        Требовать подтверждение и здесь значило бы приучить клиента слать
        confirm_reindex=true всегда — и guard перестал бы что-либо значить.
        """
        response = await self.client.put(
            SETTINGS, json={"embedding_model": DEFAULT_EMBEDDING, "top_k": 7}
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()["reindex_required"])

    async def test_a_bogus_model_name_is_named_as_such_not_as_a_confirmation(self):
        """Порядок проверок: сперва «есть ли такая модель», потом подтверждение.

        Иначе на опечатку в имени админ получал бы «подтвердите
        переиндексацию» и подтверждал бы её, чтобы узнать настоящую причину.
        """
        response = await self.client.put(
            SETTINGS, json={"embedding_model": "нет-такой-модели"}
        )

        self.assertRefused(response, 400, SettingsErrors.MODEL_NOT_INSTALLED)

    async def test_each_kind_of_bad_model_gets_its_own_code(self):
        """Раньше всё это было голым 400 с английским текстом в detail."""
        cases = (
            ({"chat_model": "нет-такой-модели"}, SettingsErrors.MODEL_NOT_INSTALLED),
            ({"chat_model": OTHER_EMBEDDING}, SettingsErrors.MODEL_WRONG_KIND),
            ({"embedding_model": CHAT_MODEL}, SettingsErrors.MODEL_WRONG_KIND),
            ({"chat_model": ""}, SettingsErrors.MODEL_REQUIRED),
            ({"embedding_model": ""}, SettingsErrors.MODEL_REQUIRED),
        )
        for body, code in cases:
            with self.subTest(body=body):
                response = await self.client.put(SETTINGS, json=body)
                self.assertRefused(response, 400, code)


# --- Флаг в ответах API -------------------------------------------------


class ReindexRequiredInResponsesTests(SettingsApiTestCase):
    async def test_get_reports_the_flag(self):
        await self.switch_embedding_model()

        response = await self.client.get(SETTINGS)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["reindex_required"])

    async def test_the_flag_is_false_on_a_clean_install(self):
        response = await self.client.get(SETTINGS)

        self.assertIn("reindex_required", response.json())
        self.assertFalse(response.json()["reindex_required"])

    async def test_an_unrelated_save_does_not_clear_the_flag(self):
        """Долг за сменой модели гасится переиндексацией, а не сохранением."""
        await self.switch_embedding_model()

        response = await self.client.put(SETTINGS, json={"top_k": 7})

        self.assertTrue(response.json()["reindex_required"])


# --- Снятие флага переиндексацией ---------------------------------------


class ReindexClearsTheFlagTests(SettingsApiTestCase):
    def reindex_returns(self, result: dict) -> None:
        patcher = patch.object(
            DocumentModuleService,
            "reindex_all_documents",
            new=AsyncMock(return_value=dict(result)),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_a_fully_successful_reindex_clears_the_flag(self):
        await self.switch_embedding_model()
        self.reindex_returns(
            {"status": "ok", "total_documents": 2, "total_chunks": 10, "errors": []}
        )

        response = await self.client.post(REINDEX)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()["reindex_required"])
        self.assertFalse(RuntimeSettingsService.get_settings()["reindex_required"])

    async def test_an_empty_corpus_counts_as_success(self):
        """Устаревших векторов не осталось, потому что их нет вовсе."""
        await self.switch_embedding_model()
        self.reindex_returns(
            {"status": "ok", "message": "No documents to reindex", "total_chunks": 0}
        )

        response = await self.client.post(REINDEX)

        self.assertFalse(response.json()["reindex_required"])

    async def test_a_partial_reindex_keeps_the_flag(self):
        """Часть документов осталась без векторов — индекс неполон.

        Погасить флаг здесь значило бы стереть единственное напоминание о том,
        что поиск отвечает не по всей базе.
        """
        await self.switch_embedding_model()
        self.reindex_returns(
            {
                "status": "partial",
                "total_documents": 2,
                "total_chunks": 5,
                "errors": ["File missing for document 2: договор.pdf"],
            }
        )

        response = await self.client.post(REINDEX)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["reindex_required"])
        self.assertTrue(RuntimeSettingsService.get_settings()["reindex_required"])

    async def test_the_original_answer_of_the_reindex_is_kept_intact(self):
        """Флаг добавлен к ответу, а не вместо него."""
        self.reindex_returns(
            {"status": "ok", "total_documents": 3, "total_chunks": 42, "errors": []}
        )

        body = (await self.client.post(REINDEX)).json()

        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["total_chunks"], 42)
        self.assertEqual(body["total_documents"], 3)

    async def test_reindex_stays_admin_only(self):
        self.reindex_returns({"status": "ok", "errors": []})
        self.as_user(await self.make_user("plain", "user"))

        response = await self.client.post(REINDEX)

        self.assertEqual(response.status_code, 403, response.text)


# --- Сброс к умолчаниям -------------------------------------------------


class ResetEndpointTests(SettingsApiTestCase):
    async def test_reset_returns_the_settings_to_their_defaults(self):
        await self.client.put(SETTINGS, json={"top_k": 7, "retrieval_top_k": 33})

        response = await self.client.post(RESET, json={})

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        defaults = RuntimeSettingsService.DEFAULTS
        self.assertEqual(body["top_k"], defaults["top_k"])
        self.assertEqual(body["retrieval_top_k"], defaults["retrieval_top_k"])
        # И при следующем чтении тоже — сброс сохранён, а не показан.
        self.assertEqual((await self.client.get(SETTINGS)).json()["top_k"], defaults["top_k"])

    async def test_reset_answers_with_the_same_shape_as_get(self):
        """Клиент обновляет весь экран одним ответом, без второго запроса."""
        response = await self.client.post(RESET, json={})

        self.assertEqual(
            sorted(response.json()), sorted((await self.client.get(SETTINGS)).json())
        )

    async def test_reset_works_without_a_body_at_all(self):
        response = await self.client.post(RESET)

        self.assertEqual(response.status_code, 200, response.text)

    async def test_reset_that_would_switch_the_embedding_model_needs_confirmation(self):
        await self.switch_embedding_model()

        response = await self.client.post(RESET, json={})

        self.assertRefused(
            response, 409, SettingsErrors.REINDEX_CONFIRMATION_REQUIRED
        )
        self.assertEqual(
            RuntimeSettingsService.get_settings()["embedding_model"],
            OTHER_EMBEDDING,
            "отклонённый сброс не должен ничего менять",
        )

    async def test_a_confirmed_reset_switches_the_model_back_and_flags_a_reindex(self):
        await self.switch_embedding_model()

        response = await self.client.post(RESET, json={"confirm_reindex": True})

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["embedding_model"], DEFAULT_EMBEDDING)
        self.assertTrue(body["reindex_required"])

    async def test_reset_is_admin_only(self):
        self.as_user(await self.make_user("plain", "user"))

        response = await self.client.post(RESET, json={})

        self.assertEqual(response.status_code, 403, response.text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
