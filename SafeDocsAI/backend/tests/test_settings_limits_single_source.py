"""Границы числовых настроек: один источник на все места, где они объявлены.

Что закрепляем.

  * **Границы больше не выписаны по местам.** Каждая пара чисел стояла трижды:
    в строгой проверке при записи (_require_int_in_range), в снисходительном
    клампе при чтении (_clamp_on_read) и в схеме ответа
    (RuntimeSettingsResponse). Совпадали они только пока их не правили: правка
    одного места из трёх разводит запись с чтением молча — запись отвергает
    ровно то значение, которое чтение подставляет само, или наоборот
    сохраняется величина, которую чтение тут же подрезает. Теперь источник
    один — SETTING_LIMITS, и тесты ниже сверяют между собой все три пути, а не
    каждый с выписанным в тесте числом: тест на конкретные числа краснел бы
    вместе с осознанной правкой границы, а разъезд между путями — пропускал.
  * **Контракт OpenAPI остался документацией.** ge/le в схеме ответа обязаны
    вычисляться на импорте и попадать в /openapi.json: генератор клиента и
    админ читают границы оттуда. Подстановка «по вызову», как у умолчаний
    (_default), здесь невозможна — и проверяется, что её и нет.
  * **Клиентские константы — проверяемое зеркало.** SettingsPage.jsx повторяет
    те же числа (python он читать не умеет, а в теле ответа границ нет — они в
    схеме). Повторение осталось, но молчаливым быть перестало: тест разбирает
    объявления в jsx и сверяет их с SETTING_LIMITS. Разойдись они — форма
    предлагала бы значение, на котором сохранение упирается в отказ, или
    запрещала бы то, что сервер принимает.
  * **Новая числовая настройка без границ не проедет.** Любое целое в DEFAULTS
    обязано иметь запись в SETTING_LIMITS: поле без границ принимает что
    угодно.
  * **Потребитель настроек не подрезает их по-своему.** Последним на top_k и
    retrieval_top_k смотрит поиск (resolve_retrieval_limits в
    app/modules/chat/service.py), и границы он держит свои. Разойдись они с
    SETTING_LIMITS — сохранённое значение перестало бы что-либо значить, а
    жалоба пришла бы на качество ответов, а не на настройки.

Ни базы данных, ни Ollama здесь не нужно: RuntimeSettingsService в них не
ходит, а патч из одного числового поля каталог моделей не запрашивает. Файл
настроек живёт во временном каталоге — рабочий
backend/data/runtime_settings.json тесты не трогают.
"""

import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Каталог с тестами в sys.path: unittest discover кладёт его туда сам, но при
# запуске `python -m unittest tests.test_settings_limits_single_source` этого не
# происходит.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.exceptions import SettingsError, SettingsErrors  # noqa: E402
from app.main import app  # noqa: E402
from app.shared.settings.runtime_settings import (  # noqa: E402
    SETTING_LIMITS,
    RuntimeSettingsService,
)


LOGGER_NAME = "app.shared.settings.runtime_settings"

# Экран настроек: единственное место на клиенте, где границы повторены.
# backend/tests -> backend -> корень репозитория.
SETTINGS_PAGE = (
    Path(__file__).resolve().parents[2] / "frontend" / "src" / "pages" / "SettingsPage.jsx"
)

# Имена клиентских констант, которыми объявлена граница каждого поля. Список
# держится здесь, а не выводится из jsx: он и есть то, что сверяется.
CLIENT_BOUND_CONSTANTS = {
    "retrieval_top_k": ("MIN_RETRIEVAL_TOP_K", "MAX_RETRIEVAL_TOP_K"),
    "top_k": ("MIN_TOP_K", "MAX_TOP_K"),
    "chat_model_num_ctx": ("MIN_NUM_CTX", "MAX_NUM_CTX"),
    "contextual_embedding_num_ctx": ("MIN_NUM_CTX", "MAX_NUM_CTX"),
}

_CONST_RE = re.compile(r"^const ([A-Z][A-Z0-9_]*)\s*=\s*(\d+);", re.MULTILINE)
_RANGES_BLOCK_RE = re.compile(r"^const NUMBER_FIELD_RANGES = \{(.*?)^\};", re.M | re.S)
_RANGES_ENTRY_RE = re.compile(r"(\w+):\s*\{([^}]*)\}", re.S)
_BOUND_RE = re.compile(r"\b(min|max):\s*([A-Za-z_$][\w$]*|\d+)")
_INPUT_BOUND_RE = re.compile(r"\b(?:min|max)=\{([A-Za-z_$][\w$]*)\}")


def _client_source() -> str:
    return SETTINGS_PAGE.read_text(encoding="utf-8")


# --- Полнота: числовая настройка без границ ------------------------------


class LimitsCoverAllNumericSettingsTests(unittest.TestCase):
    def test_every_numeric_setting_has_limits(self):
        """Целое в DEFAULTS без записи в SETTING_LIMITS — поле без границ.

        Ловит не сегодняшний разъезд, а завтрашний: настройку добавляют в
        DEFAULTS, а границы ей не заводят, и запись принимает любое число.
        """
        numeric = {
            field
            for field, value in RuntimeSettingsService.DEFAULTS.items()
            # bool — подкласс int, а у переключателя границ не бывает.
            if isinstance(value, int) and not isinstance(value, bool)
        }

        self.assertEqual(numeric, set(SETTING_LIMITS))

    def test_limits_are_ordered_and_hold_their_own_default(self):
        """Умолчание обязано лежать внутри своих границ.

        Иначе система приезжает в состояние, которое сама же не даёт сохранить:
        сброс настроек ставит значение, а следующее сохранение его отвергает.
        """
        for field, limits in SETTING_LIMITS.items():
            with self.subTest(field=field):
                self.assertLess(limits.min, limits.max)
                self.assertLessEqual(limits.min, RuntimeSettingsService.DEFAULTS[field])
                self.assertLessEqual(RuntimeSettingsService.DEFAULTS[field], limits.max)


# --- Запись и чтение: одни и те же границы -------------------------------


class WriteAndReadShareLimitsTests(unittest.TestCase):
    """Строгая проверка при записи и кламп при чтении — по одной границе.

    Числа в тесте не выписаны: он берёт их из SETTING_LIMITS и требует, чтобы
    оба пути вели себя по ним. Верни кто-нибудь литерал в один из путей —
    красным станет этот тест, а не пользовательский сценарий через полгода.
    """

    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.settings_path = Path(self._dir.name) / "runtime_settings.json"

        path_patcher = patch.object(
            RuntimeSettingsService, "_settings_path", return_value=self.settings_path
        )
        path_patcher.start()
        self.addCleanup(path_patcher.stop)

        # Жалоба на подмену при чтении пишется один раз за жизнь процесса:
        # без очистки результат зависел бы от порядка прогона.
        RuntimeSettingsService._reported_read_fallbacks.clear()
        self.addCleanup(RuntimeSettingsService._reported_read_fallbacks.clear)

    def write_settings_file(self, **values) -> None:
        self.settings_path.write_text(
            json.dumps(values, ensure_ascii=False), encoding="utf-8"
        )

    def test_write_accepts_both_ends_of_the_range(self):
        for field, limits in SETTING_LIMITS.items():
            for value in (limits.min, limits.max):
                with self.subTest(field=field, value=value):
                    updated = RuntimeSettingsService.update_settings({field: value})

                    self.assertEqual(updated[field], value)
                    self.assertEqual(RuntimeSettingsService.get_settings()[field], value)

    def test_write_refuses_a_step_outside_the_range(self):
        for field, limits in SETTING_LIMITS.items():
            for value in (limits.min - 1, limits.max + 1):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(SettingsError) as raised:
                        RuntimeSettingsService.update_settings({field: value})

                    self.assertEqual(
                        raised.exception.error_code, SettingsErrors.VALUE_OUT_OF_RANGE
                    )
                    # Границы названы в отказе: админу нужно знать, куда
                    # попадать, а не только что он промахнулся.
                    self.assertIn(str(limits.min), str(raised.exception))
                    self.assertIn(str(limits.max), str(raised.exception))

    def test_read_clamps_to_the_same_range(self):
        """Значение из файла подрезается ровно в ту границу, что держит запись.

        Разъедься они — чтение чинило бы значение в то, что запись отвергает
        (или наоборот). Здесь это одно и то же число по построению.
        """
        for field, limits in SETTING_LIMITS.items():
            for value, expected in (
                (limits.min - 1, limits.min),
                (limits.max + 1, limits.max),
            ):
                with self.subTest(field=field, value=value):
                    RuntimeSettingsService._reported_read_fallbacks.clear()
                    self.write_settings_file(**{field: value})

                    # Подмена при чтении обязана оставлять след в журнале —
                    # заодно это глушит вывод в прогоне тестов.
                    with self.assertLogs(LOGGER_NAME, level="WARNING") as logs:
                        values = RuntimeSettingsService.get_settings()

                    self.assertEqual(values[field], expected)
                    self.assertIn(field, "\n".join(logs.output))

    def test_what_read_repairs_is_exactly_what_write_accepts(self):
        """Итог чтения всегда сохраняем через запись.

        Это и есть та связка, ради которой источник границ сделан общим:
        значение, которое чтение выдало за рабочее, не должно отвергаться при
        первом же сохранении с экрана настроек.
        """
        for field, limits in SETTING_LIMITS.items():
            for value in (limits.min - 1, limits.max + 1, "junk", None):
                with self.subTest(field=field, value=value):
                    RuntimeSettingsService._reported_read_fallbacks.clear()
                    self.write_settings_file(**{field: value})
                    # О подмене остаётся строка в журнале — кроме null: «ключа
                    # нет» это норма, а не испорченное значение. Заодно вывод
                    # прогона остаётся чистым.
                    expect_log = (
                        self.assertNoLogs if value is None else self.assertLogs
                    )
                    with expect_log(LOGGER_NAME, level="WARNING"):
                        repaired = RuntimeSettingsService.get_settings()[field]

                    self.assertEqual(
                        RuntimeSettingsService.update_settings({field: repaired})[field],
                        repaired,
                    )


# --- Схема ответа: те же границы и по-прежнему в OpenAPI -----------------


class SchemaDeclaresTheSameLimitsTests(unittest.TestCase):
    def properties(self) -> dict:
        schema = app.openapi()["components"]["schemas"]["RuntimeSettingsResponse"]
        return schema["properties"]

    def test_declared_bounds_match_the_single_source(self):
        """Границы в схеме — те же, что проверяет сервер.

        Проверяются оба направления: поле с границами в схеме обязано быть
        числовой настройкой (иначе клиенту документируют то, чего сервер не
        держит), а объявленные числа обязаны совпасть с SETTING_LIMITS.
        """
        properties = self.properties()
        for field, schema in properties.items():
            if "minimum" not in schema and "maximum" not in schema:
                continue
            with self.subTest(field=field):
                self.assertIn(field, SETTING_LIMITS)
                self.assertEqual(schema.get("minimum"), SETTING_LIMITS[field].min)
                self.assertEqual(schema.get("maximum"), SETTING_LIMITS[field].max)

    def test_bounds_stay_in_the_generated_contract(self):
        """ge/le обязаны вычисляться на импорте и доезжать до /openapi.json.

        Границы полей пула кандидатов и числа фрагментов — часть контракта:
        по ним генератор клиента строит проверку, и потерять их, заменив
        Field(...) на голую аннотацию или на подстановку «по вызову», нельзя.
        Молчаливой такая потеря была бы полной: сервер продолжал бы отвечать
        теми же значениями.
        """
        properties = self.properties()
        for field in ("retrieval_top_k", "top_k"):
            with self.subTest(field=field):
                self.assertIn("minimum", properties[field])
                self.assertIn("maximum", properties[field])


# --- Потребитель настроек: тот же диапазон -------------------------------


class RetrievalLimitsAgreeWithSettingsTests(unittest.TestCase):
    """Поиск подрезает top_k и retrieval_top_k в те же границы.

    resolve_retrieval_limits (app/modules/chat/service.py) — последний, кто
    смотрит на эти два числа перед запросом в ChromaDB, и свои границы он
    выписывает сам (safe_int(..., min_value=1, max_value=20)). Разойдись они с
    SETTING_LIMITS — сохранённая настройка перестала бы что-либо значить:
    админ ставит и видит одно, поиск работает по другому, и жалобы приходят на
    качество ответов, а не на настройки.
    """

    def test_saved_values_survive_the_search_path(self):
        from app.modules.chat.service import resolve_retrieval_limits

        top_k_limits = SETTING_LIMITS["top_k"]
        retrieval_limits = SETTING_LIMITS["retrieval_top_k"]

        # Оба конца сохраняемого диапазона доезжают до поиска нетронутыми.
        for top_k in (top_k_limits.min, top_k_limits.max):
            for retrieval_top_k in (top_k, retrieval_limits.max):
                with self.subTest(top_k=top_k, retrieval_top_k=retrieval_top_k):
                    self.assertEqual(
                        resolve_retrieval_limits(
                            {"top_k": top_k, "retrieval_top_k": retrieval_top_k}
                        ),
                        (retrieval_top_k, top_k),
                    )

    def test_search_clamps_to_the_same_limits(self):
        from app.modules.chat.service import resolve_retrieval_limits

        top_k_limits = SETTING_LIMITS["top_k"]
        retrieval_limits = SETTING_LIMITS["retrieval_top_k"]
        settings_beyond = {
            "top_k": top_k_limits.max + 1,
            "retrieval_top_k": retrieval_limits.max + 1,
        }

        self.assertEqual(
            resolve_retrieval_limits(settings_beyond),
            (retrieval_limits.max, top_k_limits.max),
        )
        self.assertEqual(
            resolve_retrieval_limits({"top_k": top_k_limits.min - 1})[1],
            top_k_limits.min,
        )

    def test_search_falls_back_to_the_same_defaults(self):
        """Настроек нет — берётся то же умолчание, что отдаёт экран настроек."""
        from app.modules.chat.service import resolve_retrieval_limits

        self.assertEqual(
            resolve_retrieval_limits({}),
            (
                RuntimeSettingsService.DEFAULTS["retrieval_top_k"],
                RuntimeSettingsService.DEFAULTS["top_k"],
            ),
        )


# --- Клиент: зеркало серверных границ ------------------------------------


class ClientMirrorsServerLimitsTests(unittest.TestCase):
    """Числа в SettingsPage.jsx сверяются с SETTING_LIMITS.

    Автоматически клиенту границы не достаются: python он не читает, а в ТЕЛЕ
    ответа границ нет — они в схеме, и добавить их в тело значило бы менять
    контракт ради того, что и так известно на сборке. Поэтому зеркало осталось
    ручным, но перестало быть молчаливым: правка одной стороны без другой
    роняет этот тест.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not SETTINGS_PAGE.exists():
            # Бэкенд разворачивают и без дерева фронтенда. Сверять тогда не с
            # чем, и это не отказ.
            raise unittest.SkipTest(f"{SETTINGS_PAGE} is not available")

    def constants(self) -> dict[str, int]:
        return {
            name: int(value) for name, value in _CONST_RE.findall(_client_source())
        }

    def test_client_constants_match_the_server(self):
        constants = self.constants()
        for field, (min_name, max_name) in CLIENT_BOUND_CONSTANTS.items():
            with self.subTest(field=field):
                self.assertIn(min_name, constants)
                self.assertIn(max_name, constants)
                self.assertEqual(constants[min_name], SETTING_LIMITS[field].min)
                self.assertEqual(constants[max_name], SETTING_LIMITS[field].max)

    def test_client_checks_the_same_fields_by_the_same_numbers(self):
        """Таблица NUMBER_FIELD_RANGES — то, чем форма проверяет ввод.

        Сверяется и состав (поле, которое сервер ограничивает, а клиент нет,
        отдаёт отказ вместо подсказки), и сами числа — через имена констант,
        то есть заодно проверяется, что по месту они не выписаны литералами.
        """
        constants = self.constants()
        block = _RANGES_BLOCK_RE.search(_client_source())
        self.assertIsNotNone(block, "NUMBER_FIELD_RANGES не найден в SettingsPage.jsx")

        client_ranges: dict[str, dict[str, int]] = {}
        for field, body in _RANGES_ENTRY_RE.findall(block.group(1)):
            bounds = {}
            for bound, token in _BOUND_RE.findall(body):
                self.assertIn(
                    token,
                    constants,
                    f"{field}.{bound} должен ссылаться на константу границы, "
                    f"а не на {token}",
                )
                bounds[bound] = constants[token]
            client_ranges[field] = bounds

        self.assertEqual(set(client_ranges), set(SETTING_LIMITS))
        for field, limits in SETTING_LIMITS.items():
            with self.subTest(field=field):
                self.assertEqual(client_ranges[field].get("min"), limits.min)
                self.assertEqual(client_ranges[field].get("max"), limits.max)

    def test_form_inputs_use_the_mirrored_constants(self):
        """min/max самих полей ввода взяты из тех же констант.

        Иначе форма подсказывала бы стрелками одно, а проверяла другое.
        """
        used = set(_INPUT_BOUND_RE.findall(_client_source()))
        expected = {name for names in CLIENT_BOUND_CONSTANTS.values() for name in names}

        self.assertLessEqual(expected, used)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
