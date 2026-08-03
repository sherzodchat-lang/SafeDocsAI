"""Политика валидации настроек: что чинится при чтении и что отвергается при записи.

Что закрепляем.

  * **Кнопка «Сохранить» ожила.** contextual_embedding_model проверялся по
    каталогу Ollama ВСЕГДА, независимо от того, включено ли контекстное
    обогащение, а умолчанием поля стояла модель, которой нет ни в одном
    развёртывании ("gemma3:4b"). Итог: PUT /api/v1/settings/ отвечал 400 на
    любое сохранение — админ не мог поправить даже top_k, пока не переставит
    модель в визуально пустом селекте выключенной функции. Теперь поле
    проверяется по ИТОГОВОМУ состоянию переключателя: правка при выключенном
    обогащении проходит, включение с негодной моделью — нет.
  * **Неизвестное значение при записи — отказ, а не подмена.**
    _normalize_domain_profile отвечал 200 OK на {"default_domain_profile":
    "banking"} и сохранял "tax", то есть молча менял правила ответов
    ассистента, показывая админу успех. То же по сути делали кламп чисел и
    bool(что угодно).
  * **Чтение осталось снисходительным.** В файле может лежать значение,
    ставшее невалидным (профиль убрали из реестра, модель удалили из Ollama,
    num_ctx остался от прежних границ). Падать на нём нельзя: get_settings
    зовут чат, поиск, индексация и сам экран настроек — отказ погасил бы в том
    числе экран, на котором это чинят.
  * **Границы num_ctx.** Верхний предел 262144 не защищал ни от чего: столько
    KV-кэша на gemma4:26b не влезает в память, модель выталкивается в CPU и
    запрос заканчивается 502 по таймауту. Предел стал 32768 (см. MAX_NUM_CTX),
    и выход за него — отказ, а не тихий кламп.
  * **Отказы вокруг Ollama.** Каталог моделей обязан собираться всегда, пусть и
    пустым: неверный OLLAMA_API_BASE давал 500 на GET /api/v1/settings/, то
    есть гасил экран, где эту опечатку и правят. А пустой каталог при записи
    больше не выдаётся за «модель не установлена»: если Ollama не ответила,
    отказ отдельный, со смыслом «повторите позже».

Базы данных здесь не нужно: RuntimeSettingsService в неё не ходит. Ollama не
поднимается — подменяется list_ollama_models. Файл настроек живёт во временном
каталоге: рабочий backend/data/runtime_settings.json тесты не трогают.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Каталог с тестами в sys.path: unittest discover кладёт его туда сам, но при
# запуске `python -m unittest tests.test_settings_validation_policy` этого не
# происходит.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.exceptions import (  # noqa: E402
    ExternalServiceError,
    SettingsError,
    SettingsErrors,
)
from app.shared.settings.runtime_settings import (  # noqa: E402
    MAX_NUM_CTX,
    MIN_NUM_CTX,
    RuntimeSettingsService,
)


LOGGER_NAME = "app.shared.settings.runtime_settings"

# Ровно то, что установлено в Ollama на стенде.
STAND_MODELS = ["gemma4:e4b", "gemma4:26b", "qwen3-embedding:8b"]
CHAT_ON_STAND = "gemma4:26b"
OTHER_CHAT_ON_STAND = "gemma4:e4b"
EMBEDDING_ON_STAND = "qwen3-embedding:8b"

# Прежнее умолчание contextual_embedding_model. В Ollama его нет и не было — из
# него и росла неисправность.
MISSING_MODEL = "gemma3:4b"


class SettingsPolicyTestCase(unittest.TestCase):
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

        # Жалоба на подмену при чтении пишется один раз за жизнь процесса
        # (get_settings зовут на каждый чанк при индексации). Для тестов,
        # проверяющих журнал, память нужно очистить — иначе результат зависит
        # от порядка прогона.
        RuntimeSettingsService._reported_read_fallbacks.clear()
        self.addCleanup(RuntimeSettingsService._reported_read_fallbacks.clear)

    def install(self, models: list[str]) -> None:
        """Подменить список моделей, установленных в Ollama."""
        patcher = patch(
            "app.modules.rag.model_manager.ModelManager.list_ollama_models",
            return_value=list(models),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_settings_file(self, **values) -> None:
        """Положить в файл настроек то, что туда мог записать кто угодно."""
        self.settings_path.write_text(
            json.dumps(values, ensure_ascii=False), encoding="utf-8"
        )

    def saved(self) -> dict:
        return json.loads(self.settings_path.read_text(encoding="utf-8"))


# --- Задача 1. Контекстное обогащение -----------------------------------


class ContextualEmbeddingModelTests(SettingsPolicyTestCase):
    def test_top_k_is_saved_while_enrichment_is_off_and_its_model_is_junk(self):
        """Главная проверка: кнопка «Сохранить» работает.

        Ровно то состояние, в котором стенд и жил: обогащение выключено, а в
        его поле лежит модель, которой в Ollama нет. Раньше здесь приходил 400
        на любое сохранение.
        """
        self.write_settings_file(
            contextual_embedding_enabled=False,
            contextual_embedding_model=MISSING_MODEL,
            top_k=5,
        )

        updated = RuntimeSettingsService.update_settings({"top_k": 7})

        self.assertEqual(updated["top_k"], 7)
        self.assertEqual(self.saved()["top_k"], 7)
        # Негодное значение осталось лежать как лежало: сохранение чужого поля
        # не повод его трогать, а починит его админ выбором модели.
        self.assertEqual(updated["contextual_embedding_model"], MISSING_MODEL)

    def test_turning_enrichment_on_with_a_junk_model_is_refused(self):
        """Дыра, которую нельзя оставить: модель становится живой в этот момент.

        Переключатель приходит патчем, модель лежит в сохранённых настройках —
        значит смотреть надо на итоговое состояние, а не на состав патча.
        """
        self.write_settings_file(
            contextual_embedding_enabled=False,
            contextual_embedding_model=MISSING_MODEL,
        )

        with self.assertRaises(SettingsError) as raised:
            RuntimeSettingsService.update_settings(
                {"contextual_embedding_enabled": True}
            )

        self.assertEqual(
            raised.exception.error_code, SettingsErrors.MODEL_NOT_INSTALLED
        )
        self.assertFalse(
            self.saved()["contextual_embedding_enabled"],
            "отклонённый патч не должен включать обогащение",
        )

    def test_turning_enrichment_on_without_a_model_is_refused(self):
        """Иначе переключатель стоял бы во «включено», не делая ничего.

        Индексация читает `if _ctx_enabled and _ctx_model`
        (app/modules/documents/service.py): без модели обогащение просто
        пропускается, и снаружи это неотличимо от выключенного.
        """
        with self.assertRaises(SettingsError) as raised:
            RuntimeSettingsService.update_settings(
                {"contextual_embedding_enabled": True}
            )

        self.assertEqual(
            raised.exception.error_code, SettingsErrors.CONTEXTUAL_MODEL_REQUIRED
        )

    def test_enrichment_can_be_turned_on_together_with_its_model(self):
        """Обычный путь: переключатель и модель приходят одним патчем."""
        updated = RuntimeSettingsService.update_settings(
            {
                "contextual_embedding_enabled": True,
                "contextual_embedding_model": CHAT_ON_STAND,
            }
        )

        self.assertTrue(updated["contextual_embedding_enabled"])
        self.assertEqual(updated["contextual_embedding_model"], CHAT_ON_STAND)

    def test_changing_the_model_while_enrichment_is_on_is_still_checked(self):
        RuntimeSettingsService.update_settings(
            {
                "contextual_embedding_enabled": True,
                "contextual_embedding_model": CHAT_ON_STAND,
            }
        )

        with self.assertRaises(SettingsError) as raised:
            RuntimeSettingsService.update_settings(
                {"contextual_embedding_model": EMBEDDING_ON_STAND}
            )

        self.assertEqual(raised.exception.error_code, SettingsErrors.MODEL_WRONG_KIND)
        self.assertEqual(self.saved()["contextual_embedding_model"], CHAT_ON_STAND)

    def test_clearing_the_model_while_enrichment_is_on_is_refused(self):
        """Пустое значение при включённой функции — то же тихое отключение."""
        RuntimeSettingsService.update_settings(
            {
                "contextual_embedding_enabled": True,
                "contextual_embedding_model": CHAT_ON_STAND,
            }
        )

        with self.assertRaises(SettingsError) as raised:
            RuntimeSettingsService.update_settings({"contextual_embedding_model": ""})

        self.assertEqual(
            raised.exception.error_code, SettingsErrors.CONTEXTUAL_MODEL_REQUIRED
        )

    def test_saving_other_fields_while_enrichment_is_on_does_not_ask_ollama(self):
        """Правка top_k не должна зависеть от Ollama и при включённой функции.

        Перепроверять уже сохранённую модель на каждом сохранении значило бы
        вернуть ту же неисправность другим боком: удалили модель из Ollama —
        и настройки снова не сохраняются целиком.
        """
        RuntimeSettingsService.update_settings(
            {
                "contextual_embedding_enabled": True,
                "contextual_embedding_model": CHAT_ON_STAND,
            }
        )

        with patch.object(
            RuntimeSettingsService, "model_catalog", side_effect=AssertionError
        ):
            updated = RuntimeSettingsService.update_settings({"top_k": 9})

        self.assertEqual(updated["top_k"], 9)

    def test_turning_enrichment_off_never_asks_about_the_model(self):
        self.write_settings_file(
            contextual_embedding_enabled=True,
            contextual_embedding_model=MISSING_MODEL,
        )

        updated = RuntimeSettingsService.update_settings(
            {"contextual_embedding_enabled": False}
        )

        self.assertFalse(updated["contextual_embedding_enabled"])

    def test_the_default_model_is_empty_rather_than_a_model_nobody_has(self):
        """Умолчанием стояла "gemma3:4b" — её нет ни в одном развёртывании.

        Пустое значение читается как «модель не выбрана»: индексация его уже
        умеет (см. `if _ctx_enabled and _ctx_model`), а включение с пустым
        значением отвергается отдельно.
        """
        self.assertEqual(
            RuntimeSettingsService.DEFAULTS["contextual_embedding_model"], ""
        )
        self.assertEqual(
            RuntimeSettingsService.get_settings()["contextual_embedding_model"], ""
        )


# --- Задача 2. Неизвестное значение: запись отвергает --------------------


class WriteRefusesUnknownValuesTests(SettingsPolicyTestCase):
    def test_unknown_domain_profile_is_refused_instead_of_being_replaced(self):
        """Худший случай прежней политики: 200 OK и другие правила ответов.

        PUT {"default_domain_profile": "banking"} отвечал успехом и сохранял
        "tax" — админ видел, что всё применилось.
        """
        with self.assertRaises(SettingsError) as raised:
            RuntimeSettingsService.update_settings(
                {"default_domain_profile": "banking"}
            )

        self.assertEqual(
            raised.exception.error_code, SettingsErrors.UNSUPPORTED_DOMAIN_PROFILE
        )
        self.assertFalse(
            self.settings_path.exists(), "отклонённый патч не должен сохраняться"
        )

    def test_a_registered_profile_is_still_accepted(self):
        updated = RuntimeSettingsService.update_settings(
            {"default_domain_profile": "tax"}
        )

        self.assertEqual(updated["default_domain_profile"], "tax")

    def test_numbers_out_of_range_are_refused_instead_of_being_clamped(self):
        cases = (
            ({"top_k": 0}, SettingsErrors.VALUE_OUT_OF_RANGE),
            ({"top_k": 21}, SettingsErrors.VALUE_OUT_OF_RANGE),
            ({"retrieval_top_k": 0}, SettingsErrors.VALUE_OUT_OF_RANGE),
            ({"retrieval_top_k": 51}, SettingsErrors.VALUE_OUT_OF_RANGE),
        )
        for patch_body, code in cases:
            with self.subTest(patch_body=patch_body):
                with self.assertRaises(SettingsError) as raised:
                    RuntimeSettingsService.update_settings(patch_body)
                self.assertEqual(raised.exception.error_code, code)

    def test_values_that_are_not_numbers_are_refused(self):
        for patch_body in ({"top_k": "много"}, {"chat_model_num_ctx": 8192.5}):
            with self.subTest(patch_body=patch_body):
                with self.assertRaises(SettingsError) as raised:
                    RuntimeSettingsService.update_settings(patch_body)
                self.assertEqual(
                    raised.exception.error_code, SettingsErrors.INVALID_NUMBER
                )

    def test_a_switch_is_not_turned_on_by_arbitrary_junk(self):
        """bool("banana") — это True: переключатель включался от чего угодно."""
        with self.assertRaises(SettingsError) as raised:
            RuntimeSettingsService.update_settings({"reranker_enabled": "banana"})

        self.assertEqual(raised.exception.error_code, SettingsErrors.INVALID_BOOLEAN)

    def test_words_that_do_mean_a_boolean_are_still_accepted(self):
        """Договор не сузился: строки-признаки читались и раньше, и в файле
        настроек они могут лежать в этом виде."""
        updated = RuntimeSettingsService.update_settings(
            {"reranker_enabled": "yes", "enable_condense_query": "off"}
        )

        self.assertTrue(updated["reranker_enabled"])
        self.assertFalse(updated["enable_condense_query"])

    def test_confirmation_of_a_reindex_cannot_be_given_by_junk(self):
        """Согласие на переиндексацию — тоже логическое значение."""
        self.install([*STAND_MODELS, "bge-m3"])
        RuntimeSettingsService.update_settings(
            {"embedding_model": EMBEDDING_ON_STAND, "confirm_reindex": True}
        )

        with self.assertRaises(SettingsError) as raised:
            RuntimeSettingsService.update_settings(
                {"embedding_model": "bge-m3", "confirm_reindex": "banana"}
            )

        self.assertEqual(raised.exception.error_code, SettingsErrors.INVALID_BOOLEAN)

    def test_every_refusal_is_a_valueerror_with_a_machine_code(self):
        """Договор с вызывающими вне API (backend/*.py ловят ValueError) цел."""
        with self.assertRaises(ValueError) as raised:
            RuntimeSettingsService.update_settings({"default_domain_profile": "нет"})

        self.assertTrue(getattr(raised.exception, "error_code", None))


# --- Задача 2. Неизвестное значение: чтение чинит ------------------------


class ReadRepairsUnknownValuesTests(SettingsPolicyTestCase):
    def test_a_profile_that_left_the_registry_does_not_break_reading(self):
        """Профиль убрали из реестра уже после сохранения настроек.

        Если бы get_settings падал на этом, погас бы и экран настроек — тот
        самый, на котором профиль выбирают заново.
        """
        self.write_settings_file(default_domain_profile="banking")

        values = RuntimeSettingsService.get_settings()

        self.assertEqual(values["default_domain_profile"], "tax")

    def test_a_repair_on_read_leaves_a_line_in_the_log(self):
        """Тихая подмена и была тем, что нечем объяснить: «настройка не
        работает», и ни одной записи о том, что она подменена."""
        self.write_settings_file(default_domain_profile="banking")

        with self.assertLogs(LOGGER_NAME, level="WARNING") as captured:
            RuntimeSettingsService.get_settings()

        message = "\n".join(captured.output)
        self.assertIn("default_domain_profile", message)
        self.assertIn("banking", message)

    def test_numbers_out_of_range_are_clamped_on_read(self):
        """В файле может лежать значение, записанное при прежних границах."""
        self.write_settings_file(top_k=999, chat_model_num_ctx=262144)

        values = RuntimeSettingsService.get_settings()

        self.assertEqual(values["top_k"], 20)
        self.assertEqual(values["chat_model_num_ctx"], MAX_NUM_CTX)

    def test_garbage_in_a_number_falls_back_to_the_default_on_read(self):
        self.write_settings_file(top_k="много")

        values = RuntimeSettingsService.get_settings()

        self.assertEqual(values["top_k"], RuntimeSettingsService.DEFAULTS["top_k"])

    def test_a_word_meaning_no_no_longer_turns_a_heavy_feature_on(self):
        """Главная проверка: bool непустой строки — это True.

        Читалось `bool(value)`, поэтому "нет", "disabled" и "нет-нет" —
        то есть ровно те слова, которыми функцию пытаются ВЫКЛЮЧИТЬ, —
        включали её. И включали именно тяжёлое: реранкер гоняет модель на
        каждый запрос, контекстное обогащение — на каждый чанк при индексации.
        """
        self.write_settings_file(
            reranker_enabled="нет",
            contextual_embedding_enabled="disabled",
            reindex_required="нет-нет",
        )

        values = RuntimeSettingsService.get_settings()

        self.assertFalse(values["reranker_enabled"])
        self.assertFalse(values["contextual_embedding_enabled"])
        self.assertFalse(values["reindex_required"])

    def test_an_unreadable_flag_reads_as_that_fields_own_default(self):
        """Не глобальное False, а умолчание поля.

        У enable_condense_query умолчание True — функция включена намеренно.
        Безусловное False чинило бы одно тихое включение ценой другого тихого
        выключения; «значение непонятно» и «значения нет» — одно состояние
        знания, и вести себя они обязаны одинаково.
        """
        self.write_settings_file(enable_condense_query="ага")

        values = RuntimeSettingsService.get_settings()

        self.assertIs(values["enable_condense_query"], True)
        self.assertEqual(
            values["enable_condense_query"],
            RuntimeSettingsService.DEFAULTS["enable_condense_query"],
        )

    def test_an_unreadable_flag_is_never_silent(self):
        """Предсказуемо И громко: подмена уходит тем же механизмом, что и
        остальные подмены при чтении (_report_read_fallback)."""
        self.write_settings_file(reranker_enabled="нет")

        with self.assertLogs(LOGGER_NAME, level="WARNING") as captured:
            RuntimeSettingsService.get_settings()

        message = "\n".join(captured.output)
        self.assertIn("reranker_enabled", message)
        self.assertIn("нет", message)

    def test_an_unreadable_flag_does_not_break_reading(self):
        """Это путь ЧТЕНИЯ: падать нельзя ни на чём.

        get_settings зовут чат, поиск, индексация и сам экран настроек —
        отказ погасил бы в том числе экран, на котором флаг исправляют.
        """
        self.write_settings_file(reranker_enabled={"не": "логическое"})

        values = RuntimeSettingsService.get_settings()

        self.assertFalse(values["reranker_enabled"])
        # Остальные настройки прочитались как ни в чём не бывало.
        self.assertEqual(values["top_k"], RuntimeSettingsService.DEFAULTS["top_k"])

    def test_words_that_do_mean_a_boolean_are_still_read_on_read(self):
        """Снисходительность к понятным словам осталась: _TRUE_WORDS и
        _FALSE_WORDS общие у чтения и у записи."""
        self.write_settings_file(reranker_enabled="yes", enable_condense_query="off")

        values = RuntimeSettingsService.get_settings()

        self.assertIs(values["reranker_enabled"], True)
        self.assertIs(values["enable_condense_query"], False)

    def test_a_model_that_left_ollama_is_still_returned_as_it_is(self):
        """Каталог при чтении не спрашивается вовсе.

        Иначе настройки нельзя было бы прочитать, пока Ollama лежит, — а
        читают их чат, поиск и индексация.
        """
        self.write_settings_file(chat_model=MISSING_MODEL)

        with patch.object(
            RuntimeSettingsService, "model_catalog", side_effect=AssertionError
        ):
            values = RuntimeSettingsService.get_settings()

        self.assertEqual(values["chat_model"], MISSING_MODEL)


# --- Задача 4. Границы окна контекста ------------------------------------


class NumCtxBoundsTests(SettingsPolicyTestCase):
    def test_a_window_nobody_has_memory_for_is_refused(self):
        """262144 на gemma4:26b раздувает KV-кэш на порядок: модель уезжает в
        CPU, запрос упирается в таймаут и пользователь получает 502."""
        with self.assertRaises(SettingsError) as raised:
            RuntimeSettingsService.update_settings({"chat_model_num_ctx": 262144})

        self.assertEqual(
            raised.exception.error_code, SettingsErrors.VALUE_OUT_OF_RANGE
        )
        self.assertIn(str(MAX_NUM_CTX), str(raised.exception))

    def test_zero_is_refused_instead_of_quietly_becoming_2048(self):
        with self.assertRaises(SettingsError) as raised:
            RuntimeSettingsService.update_settings({"contextual_embedding_num_ctx": 0})

        self.assertEqual(
            raised.exception.error_code, SettingsErrors.VALUE_OUT_OF_RANGE
        )

    def test_the_bounds_themselves_are_accepted(self):
        updated = RuntimeSettingsService.update_settings(
            {"chat_model_num_ctx": MAX_NUM_CTX, "contextual_embedding_num_ctx": MIN_NUM_CTX}
        )

        self.assertEqual(updated["chat_model_num_ctx"], MAX_NUM_CTX)
        self.assertEqual(updated["contextual_embedding_num_ctx"], MIN_NUM_CTX)

    def test_every_window_used_in_deployment_still_fits(self):
        """Граница не должна мешать тому, что уже развёрнуто.

        Modelfile'ы стенда пиннят 20000 (gemma4:e4b) и 12000 (gemma4:26b),
        умолчания в коде — 20000 и 8192, ModelManager без явного значения
        берёт 12288.
        """
        for window in (8192, 12000, 12288, 20000):
            with self.subTest(window=window):
                updated = RuntimeSettingsService.update_settings(
                    {"chat_model_num_ctx": window}
                )
                self.assertEqual(updated["chat_model_num_ctx"], window)


# --- Задача 5. Отказы вокруг Ollama --------------------------------------


class OllamaFailureTests(SettingsPolicyTestCase):
    def test_a_broken_ollama_client_does_not_break_the_settings_screen(self):
        """Ловился только ExternalServiceError, а в блоке три операции.

        Конструктор ModelManager создаёт ollama.Client по адресу из
        OLLAMA_API_BASE: опечатка в конфиге давала 500 на GET /settings/ — то
        есть гасила экран, на котором эту опечатку и правят.
        """
        with patch(
            "app.modules.rag.model_manager.ModelManager.__init__",
            side_effect=ValueError("invalid host in OLLAMA_API_BASE"),
        ):
            catalog = RuntimeSettingsService.model_catalog()

        self.assertFalse(catalog["ollama_available"])
        self.assertIn("ValueError", catalog["ollama_error"])
        self.assertEqual(catalog["available_models"], [])
        self.assertEqual(catalog["available_chat_models"], [])

    def test_a_failed_import_does_not_break_the_settings_screen_either(self):
        with patch(
            "app.modules.rag.model_manager.ModelManager.__init__",
            side_effect=ImportError("No module named 'ollama'"),
        ):
            catalog = RuntimeSettingsService.model_catalog()

        self.assertFalse(catalog["ollama_available"])
        self.assertIn("ImportError", catalog["ollama_error"])

    def test_the_failure_is_reported_in_the_log(self):
        with patch(
            "app.modules.rag.model_manager.ModelManager.__init__",
            side_effect=ValueError("invalid host"),
        ):
            with self.assertLogs(LOGGER_NAME, level="WARNING") as captured:
                RuntimeSettingsService.model_catalog()

        self.assertIn("invalid host", "\n".join(captured.output))

    def test_a_downed_ollama_is_not_blamed_on_the_model(self):
        """Пустой каталог не означает «модель не установлена».

        Модель может стоять на месте и быть настроенной уже сейчас; чинить
        надо сервис, а не выбор админа. Поэтому отдельный код и 503 — с
        просьбой повторить тот же запрос позже.
        """
        patcher = patch(
            "app.modules.rag.model_manager.ModelManager.list_ollama_models",
            side_effect=ExternalServiceError(
                "Ollama is unavailable", service="Ollama", status_code=503
            ),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        for patch_body in (
            {"chat_model": CHAT_ON_STAND},
            {"embedding_model": EMBEDDING_ON_STAND},
            {
                "contextual_embedding_enabled": True,
                "contextual_embedding_model": CHAT_ON_STAND,
            },
        ):
            with self.subTest(patch_body=patch_body):
                with self.assertRaises(SettingsError) as raised:
                    RuntimeSettingsService.update_settings(patch_body)
                self.assertEqual(
                    raised.exception.error_code,
                    SettingsErrors.MODEL_CATALOG_UNAVAILABLE,
                )
                self.assertIn("Ollama is unavailable", str(raised.exception))

    def test_a_downed_ollama_does_not_block_settings_without_models(self):
        """Обратная сторона: правка top_k при лежащей Ollama обязана проходить."""
        patcher = patch(
            "app.modules.rag.model_manager.ModelManager.list_ollama_models",
            side_effect=ExternalServiceError(
                "Ollama is unavailable", service="Ollama", status_code=503
            ),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        updated = RuntimeSettingsService.update_settings({"top_k": 7})

        self.assertEqual(updated["top_k"], 7)

    def test_the_catalog_is_collected_once_per_save(self):
        """Запрос в Ollama на каждое поле с моделью — три запроса на патч."""
        with patch.object(
            RuntimeSettingsService,
            "model_catalog",
            wraps=RuntimeSettingsService.model_catalog,
        ) as catalog:
            RuntimeSettingsService.update_settings(
                {
                    "chat_model": CHAT_ON_STAND,
                    "embedding_model": EMBEDDING_ON_STAND,
                    "contextual_embedding_enabled": True,
                    "contextual_embedding_model": OTHER_CHAT_ON_STAND,
                    "confirm_reindex": True,
                }
            )

        self.assertEqual(catalog.call_count, 1)


class ModelManagerDiscoveryTests(unittest.TestCase):
    """Тот же дефект уровнем ниже: клиент создавался ВНЕ try."""

    def test_a_client_that_refuses_to_be_created_becomes_a_service_error(self):
        from app.modules.rag import model_manager as model_manager_module

        manager = model_manager_module.ModelManager()

        with patch.object(
            model_manager_module.ollama,
            "Client",
            side_effect=ValueError("invalid host"),
        ):
            with self.assertRaises(ExternalServiceError):
                manager.list_ollama_models()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
