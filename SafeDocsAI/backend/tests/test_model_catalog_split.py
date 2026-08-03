"""Списки моделей чата и эмбеддингов — разные списки.

Что закрепляем.

  * model_catalog() отдавал available_chat_models и available_embedding_models
    как ОДИН И ТОТ ЖЕ список всего, что установлено в Ollama
    (`available_chat_models = list(available_embedding_models)`), а валидация
    при сохранении сверялась с этим общим списком. То есть не защищала ни от
    чего: на стенде в поле «модель чата» проходил qwen3-embedding:8b — чат
    после этого ломается на каждом запросе, — а в поле эмбеддингов проходил
    gemma4:26b, и поиск молча уезжал в пустую коллекцию ChromaDB (имя
    коллекции выводится из embedding-модели).
  * Признака «embedding-модель» у Ollama в /api/tags нет (см. комментарий к
    MODEL_KIND_* в runtime_settings.py), поэтому вид модели определяется
    эвристикой по имени. Эвристика намеренно неполная, и это главное, что
    здесь проверяется: НЕОПОЗНАННОЕ имя остаётся разрешённым в обоих полях.
    Без этого админ не смог бы настроить систему на модель со своей сборкой
    или из приватного реестра — а «слишком строго» здесь дороже пропуска.

Ollama не поднимается: подменяется ModelManager.list_ollama_models, то есть
ровно та точка, где каталог получает список установленных моделей. Файл
runtime_settings.json подменён во временный каталог — рабочий
backend/data/runtime_settings.json тесты не трогают.

Базы данных этим проверкам не нужно: RuntimeSettingsService в неё не ходит.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Каталог с тестами в sys.path: unittest discover кладёт его туда сам, но при
# запуске `python -m unittest tests.test_model_catalog_split` этого не происходит.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.exceptions import (  # noqa: E402
    ExternalServiceError,
    SettingsError,
    SettingsErrors,
)
from app.shared.settings.runtime_settings import (  # noqa: E402
    MODEL_KIND_CHAT,
    MODEL_KIND_EMBEDDING,
    MODEL_KIND_UNKNOWN,
    RuntimeSettingsService,
)


# Ровно то, что установлено в Ollama на стенде.
STAND_MODELS = ["gemma4:e4b", "gemma4:26b", "qwen3-embedding:8b"]

CHAT_ON_STAND = "gemma4:26b"
EMBEDDING_ON_STAND = "qwen3-embedding:8b"
# Имя, которое эвристика опознать не может: ни маркера эмбеддинга, ни
# известного семейства чат-моделей.
UNKNOWN_NAME = "svoya-sborka:v2"


class ModelCatalogTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.settings_path = Path(self._dir.name) / "runtime_settings.json"

        path_patcher = patch.object(
            RuntimeSettingsService, "_settings_path", return_value=self.settings_path
        )
        path_patcher.start()
        self.addCleanup(path_patcher.stop)

        self.install(STAND_MODELS)

    def install(self, models: list[str]) -> None:
        """Подменить список моделей, установленных в Ollama."""
        patcher = patch(
            "app.modules.rag.model_manager.ModelManager.list_ollama_models",
            return_value=list(models),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def saved(self) -> dict:
        return json.loads(self.settings_path.read_text(encoding="utf-8"))


# --- Классификация имён -------------------------------------------------


class ModelKindTests(ModelCatalogTestCase):
    def test_names_of_real_models_are_classified(self):
        expected = {
            # Стенд
            "gemma4:e4b": MODEL_KIND_CHAT,
            "gemma4:26b": MODEL_KIND_CHAT,
            "qwen3-embedding:8b": MODEL_KIND_EMBEDDING,
            # Ходовые embedding-модели: у части из них слова "embed" в имени
            # нет вообще, поэтому одной подстроки мало.
            "nomic-embed-text": MODEL_KIND_EMBEDDING,
            "mxbai-embed-large": MODEL_KIND_EMBEDDING,
            "granite-embedding:278m": MODEL_KIND_EMBEDDING,
            "bge-m3": MODEL_KIND_EMBEDDING,
            "multilingual-e5-large:latest": MODEL_KIND_EMBEDDING,
            "all-minilm": MODEL_KIND_EMBEDDING,
            "labse": MODEL_KIND_EMBEDDING,
            # Семейство чата стоит вторым словом, но модель — embedding.
            "gte-qwen2-7b-instruct": MODEL_KIND_EMBEDDING,
            # Ходовые чат-модели
            "llama3.1:8b": MODEL_KIND_CHAT,
            "qwen2.5-coder:7b": MODEL_KIND_CHAT,
            "gpt-oss:20b": MODEL_KIND_CHAT,
            "mistral-small3.2": MODEL_KIND_CHAT,
            # Неопознанное
            UNKNOWN_NAME: MODEL_KIND_UNKNOWN,
            "hf.co/user/custom-model": MODEL_KIND_UNKNOWN,
            "": MODEL_KIND_UNKNOWN,
        }
        for name, kind in expected.items():
            self.assertEqual(RuntimeSettingsService.classify_model(name), kind, name)

    def test_size_tag_is_not_mistaken_for_the_e5_family(self):
        """"e5" ищется отдельным словом, а не префиксом.

        Иначе тег размера вида ":e5b" (родня gemma4:e4b) записал бы чат-модель
        в embedding-список — и поиск уехал бы в коллекцию, которую никто
        никогда не заполнял.
        """
        self.assertEqual(
            RuntimeSettingsService.classify_model("gemma4:e5b"), MODEL_KIND_CHAT
        )


# --- Каталог ------------------------------------------------------------


class ModelCatalogSplitTests(ModelCatalogTestCase):
    def test_stand_catalog_offers_each_model_only_where_it_works(self):
        catalog = RuntimeSettingsService.model_catalog()

        self.assertEqual(catalog["available_chat_models"], ["gemma4:e4b", "gemma4:26b"])
        self.assertEqual(catalog["available_embedding_models"], ["qwen3-embedding:8b"])

    def test_the_two_lists_are_no_longer_the_same_list(self):
        """Главный дефект: раньше один список отдавался под двумя именами."""
        catalog = RuntimeSettingsService.model_catalog()

        self.assertNotEqual(
            catalog["available_chat_models"], catalog["available_embedding_models"]
        )

    def test_available_models_still_lists_everything_installed(self):
        """Общий список остаётся полным: его отдаёт API и читает фронт."""
        catalog = RuntimeSettingsService.model_catalog()

        self.assertEqual(catalog["available_models"], STAND_MODELS)
        self.assertEqual(RuntimeSettingsService.available_models(), STAND_MODELS)

    def test_unknown_name_is_offered_in_both_fields(self):
        """Эвристика не должна отнимать у админа выбор.

        Модель, которую по имени опознать нельзя, обязана остаться доступной
        и как модель чата, и как модель эмбеддингов.
        """
        self.install([*STAND_MODELS, UNKNOWN_NAME])

        catalog = RuntimeSettingsService.model_catalog()

        self.assertIn(UNKNOWN_NAME, catalog["available_chat_models"])
        self.assertIn(UNKNOWN_NAME, catalog["available_embedding_models"])

    def test_downed_ollama_reports_the_error_and_empties_both_lists(self):
        patcher = patch(
            "app.modules.rag.model_manager.ModelManager.list_ollama_models",
            side_effect=ExternalServiceError(
                "Ollama is unavailable", service="Ollama", status_code=503
            ),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        catalog = RuntimeSettingsService.model_catalog()

        self.assertFalse(catalog["ollama_available"])
        self.assertEqual(catalog["ollama_error"], "Ollama is unavailable")
        self.assertEqual(catalog["available_chat_models"], [])
        self.assertEqual(catalog["available_embedding_models"], [])


# --- Сохранение настроек ------------------------------------------------


class ModelValidationOnSaveTests(ModelCatalogTestCase):
    def test_each_field_accepts_a_model_of_its_own_kind(self):
        updated = RuntimeSettingsService.update_settings(
            {
                "chat_model": CHAT_ON_STAND,
                "embedding_model": EMBEDDING_ON_STAND,
                # Смена embedding-модели подтверждается явно — здесь это не
                # предмет проверки, см. test_embedding_model_guard.py.
                "confirm_reindex": True,
            }
        )

        self.assertEqual(updated["chat_model"], CHAT_ON_STAND)
        self.assertEqual(updated["model"], CHAT_ON_STAND, "устаревший алиас")
        self.assertEqual(updated["embedding_model"], EMBEDDING_ON_STAND)
        self.assertEqual(self.saved()["embedding_model"], EMBEDDING_ON_STAND)

    def test_embedding_model_is_rejected_as_a_chat_model(self):
        with self.assertRaises(ValueError) as raised:
            RuntimeSettingsService.update_settings({"chat_model": EMBEDDING_ON_STAND})

        self.assertIn("embedding model", str(raised.exception))
        self.assertFalse(
            self.settings_path.exists(), "отклонённый патч не должен сохраняться"
        )

    def test_chat_model_is_rejected_as_an_embedding_model(self):
        with self.assertRaises(ValueError) as raised:
            RuntimeSettingsService.update_settings({"embedding_model": CHAT_ON_STAND})

        self.assertIn("chat model", str(raised.exception))
        self.assertFalse(self.settings_path.exists())

    def test_legacy_model_alias_is_validated_the_same_way(self):
        """Устаревшее имя поля ("model") — тот же вход в те же настройки."""
        with self.assertRaises(ValueError):
            RuntimeSettingsService.update_settings({"model": EMBEDDING_ON_STAND})

    def test_model_that_is_not_installed_is_still_rejected(self):
        for patch_body in (
            {"chat_model": "нет-такой-модели"},
            {"embedding_model": "нет-такой-модели"},
        ):
            with self.subTest(patch_body=patch_body):
                with self.assertRaises(ValueError) as raised:
                    RuntimeSettingsService.update_settings(patch_body)
                self.assertIn("Unsupported", str(raised.exception))

    def test_empty_value_is_rejected_before_the_catalog_is_consulted(self):
        for patch_body in ({"chat_model": ""}, {"embedding_model": ""}):
            with self.subTest(patch_body=patch_body):
                with self.assertRaises(ValueError):
                    RuntimeSettingsService.update_settings(patch_body)

    def test_unknown_name_can_be_saved_into_either_field(self):
        """Обратная сторона эвристики: неопознанное имя проходит везде."""
        self.install([*STAND_MODELS, UNKNOWN_NAME])

        updated = RuntimeSettingsService.update_settings(
            {"embedding_model": UNKNOWN_NAME, "confirm_reindex": True}
        )
        self.assertEqual(updated["embedding_model"], UNKNOWN_NAME)

        updated = RuntimeSettingsService.update_settings({"chat_model": UNKNOWN_NAME})
        self.assertEqual(updated["chat_model"], UNKNOWN_NAME)

    def test_contextual_embedding_model_is_checked_against_chat_models(self):
        """Вопреки имени поля это чат-модель: она пишет описание чанка.

        Вектор считает embedding_model, а сюда попадает обычная LLM, поэтому
        embedding-модель здесь так же бессмысленна, как в поле чата.

        Проверка идёт при ВКЛЮЧЁННОМ обогащении: у выключенной функции значение
        поля ни на что не влияет, и валидация там ломала сохранение целиком —
        см. tests/test_settings_validation_policy.py.
        """
        updated = RuntimeSettingsService.update_settings(
            {
                "contextual_embedding_enabled": True,
                "contextual_embedding_model": CHAT_ON_STAND,
            }
        )
        self.assertEqual(updated["contextual_embedding_model"], CHAT_ON_STAND)

        with self.assertRaises(ValueError):
            RuntimeSettingsService.update_settings(
                {"contextual_embedding_model": EMBEDDING_ON_STAND}
            )

    def test_saving_without_model_fields_does_not_ask_ollama(self):
        """Каталог собирается запросом в Ollama — на патче без моделей его
        трогать незачем, а лежащая Ollama не должна мешать менять top_k."""
        with patch.object(
            RuntimeSettingsService, "model_catalog", side_effect=AssertionError
        ):
            updated = RuntimeSettingsService.update_settings({"top_k": 7})

        self.assertEqual(updated["top_k"], 7)

    def test_switching_the_embedding_model_still_flags_a_reindex(self):
        """Смежная половина задачи (флаг для UI) опирается на этот признак.

        confirm_reindex обязателен с тех пор, как смена embedding-модели стала
        подтверждаемой операцией (см. test_embedding_model_guard.py).
        """
        self.install([*STAND_MODELS, "bge-m3"])
        RuntimeSettingsService.update_settings(
            {"embedding_model": EMBEDDING_ON_STAND, "confirm_reindex": True}
        )

        updated = RuntimeSettingsService.update_settings(
            {"embedding_model": "bge-m3", "confirm_reindex": True}
        )

        self.assertTrue(updated["reindex_required"])

    def test_each_refusal_carries_its_own_machine_code(self):
        """Три разных исхода — три разных кода.

        Слить их в один нельзя: «модели нет в Ollama» лечится `ollama pull`,
        «модель не того вида» — выбором другой из списка, а пустое поле —
        вводом значения. Интерфейс переведён на три языка и показывает
        подсказку по коду, а не по английскому тексту.
        """
        cases = (
            ({"chat_model": ""}, SettingsErrors.MODEL_REQUIRED),
            ({"embedding_model": ""}, SettingsErrors.MODEL_REQUIRED),
            ({"chat_model": "нет-такой-модели"}, SettingsErrors.MODEL_NOT_INSTALLED),
            (
                {"embedding_model": "нет-такой-модели"},
                SettingsErrors.MODEL_NOT_INSTALLED,
            ),
            ({"chat_model": EMBEDDING_ON_STAND}, SettingsErrors.MODEL_WRONG_KIND),
            ({"embedding_model": CHAT_ON_STAND}, SettingsErrors.MODEL_WRONG_KIND),
            (
                # Модель контекстного обогащения проверяется, когда обогащение
                # включают: пока функция выключена, значение поля мертво.
                {
                    "contextual_embedding_enabled": True,
                    "contextual_embedding_model": EMBEDDING_ON_STAND,
                },
                SettingsErrors.MODEL_WRONG_KIND,
            ),
        )
        for patch_body, code in cases:
            with self.subTest(patch_body=patch_body):
                with self.assertRaises(SettingsError) as raised:
                    RuntimeSettingsService.update_settings(patch_body)
                self.assertEqual(raised.exception.error_code, code)
                # Договор с вызывающими вне API (backend/*.py ловят ValueError)
                # не сломан: код добавлен рядом, а не вместо.
                self.assertIsInstance(raised.exception, ValueError)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
