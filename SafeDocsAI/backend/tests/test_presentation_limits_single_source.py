"""Границы заказа презентации объявлены один раз и зеркалятся клиентом.

Тот же приём, что в test_settings_limits_single_source.py, и по той же
причине: фронтенд не читает python, поэтому его константы — копия. Копия без
сторожа расходится молча, и расхождение обнаруживается позже всего: форма
запрещает то, что сервер принимает (или наоборот), а тесты обеих сторон
зелёные.

История здесь не гипотетическая. Этап 1 завёл SLIDE_COUNT_MIN=3 (схемный
минимум) и SLIDE_COUNT_MAX=20 (бюджетный потолок), этап 2 привёл их к
продуктовым 5..15, а клиент к этому моменту уже отзеркалил 3..20 — то есть
форма позволяла заказать колоду, которую сервер отвергнет. Поймал это
человек при сверке отчётов, а не прогон.

Числа в этом файле не пишутся: все проверки спрашивают их у констант.
Тест на конкретные 5 и 15 краснел бы при честной правке границы и пропускал
бы то единственное, ради чего заведён, — разъезд сторон.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from app.modules.presentations import constants as presentation_constants
from app.modules.presentations.llm_schemas import MIN_SLIDE_COUNT

# backend/tests -> backend -> корень репозитория.
CLIENT_MODULE = (
    Path(__file__).resolve().parents[2]
    / "frontend"
    / "src"
    / "lib"
    / "presentations.js"
)

# Имя серверной константы -> имя клиентской. Список держится здесь, а не
# выводится из исходников: он и есть то, что сверяется.
MIRRORED_CONSTANTS = {
    "SLIDE_COUNT_MIN": "SLIDE_COUNT_MIN",
    "SLIDE_COUNT_MAX": "SLIDE_COUNT_MAX",
    "DESCRIPTION_MAX": "DESCRIPTION_MAX",
}

_CONST_RE = re.compile(r"^const ([A-Z][A-Z0-9_]*)\s*=\s*(\d+);", re.MULTILINE)


def _client_constants() -> dict[str, int]:
    source = CLIENT_MODULE.read_text(encoding="utf-8")
    return {name: int(value) for name, value in _CONST_RE.findall(source)}


class ClientMirrorsServerLimitsTests(unittest.TestCase):
    """Клиентские копии равны серверным оригиналам."""

    def test_client_module_exists(self) -> None:
        # Если модуль переименуют, остальные проверки молча выродятся в пустые.
        self.assertTrue(
            CLIENT_MODULE.is_file(),
            f"не найден клиентский модуль границ: {CLIENT_MODULE}",
        )

    def test_every_mirrored_constant_is_declared_on_the_client(self) -> None:
        client = _client_constants()
        for server_name, client_name in MIRRORED_CONSTANTS.items():
            with self.subTest(constant=server_name):
                self.assertIn(
                    client_name,
                    client,
                    f"{client_name} не объявлена в {CLIENT_MODULE.name} — "
                    "либо переименована, либо записана числом по месту",
                )

    def test_mirrored_values_match(self) -> None:
        client = _client_constants()
        for server_name, client_name in MIRRORED_CONSTANTS.items():
            with self.subTest(constant=server_name):
                self.assertEqual(
                    client[client_name],
                    getattr(presentation_constants, server_name),
                    f"{client_name} на клиенте разошлась с {server_name} на "
                    "сервере: форма и валидатор пропускают разное",
                )

    def test_client_default_is_inside_the_mirrored_range(self) -> None:
        # Умолчание — решение интерфейса (у сервера своего нет), но оно обязано
        # лежать внутри границ: иначе форма открывается с незаказуемым числом.
        client = _client_constants()
        self.assertIn("DEFAULT_SLIDE_COUNT", client)
        self.assertGreaterEqual(client["DEFAULT_SLIDE_COUNT"], client["SLIDE_COUNT_MIN"])
        self.assertLessEqual(client["DEFAULT_SLIDE_COUNT"], client["SLIDE_COUNT_MAX"])


class ServerLimitsAreCoherentTests(unittest.TestCase):
    """Серверные границы согласованы между собой."""

    def test_product_minimum_is_not_below_the_schema_minimum(self) -> None:
        # Схема собирает титул + содержательные + «Источники»; заказать меньше
        # того, что вообще собирается, нельзя.
        self.assertGreaterEqual(presentation_constants.SLIDE_COUNT_MIN, MIN_SLIDE_COUNT)

    def test_range_is_not_empty(self) -> None:
        self.assertLess(
            presentation_constants.SLIDE_COUNT_MIN,
            presentation_constants.SLIDE_COUNT_MAX,
        )

    def test_default_is_inside_the_range(self) -> None:
        self.assertGreaterEqual(
            presentation_constants.SLIDE_COUNT_DEFAULT,
            presentation_constants.SLIDE_COUNT_MIN,
        )
        self.assertLessEqual(
            presentation_constants.SLIDE_COUNT_DEFAULT,
            presentation_constants.SLIDE_COUNT_MAX,
        )


class ErrorCodesAreMirroredTests(unittest.TestCase):
    """Каждый код раздела имеет перевод на клиенте.

    Код без записи в таблице переводов деградирует в общий фолбэк — то есть
    отказ теряет ровно ту конкретику, ради которой заводился отдельный код.
    """

    def test_every_presentation_code_has_a_translation_key(self) -> None:
        from app.core.exceptions import PresentationErrors

        api_error_module = (
            CLIENT_MODULE.parent / "apiError.js"
        ).read_text(encoding="utf-8")

        codes = [
            value
            for name, value in vars(PresentationErrors).items()
            if not name.startswith("_") and isinstance(value, str)
        ]
        self.assertTrue(codes, "PresentationErrors пуст — проверка выродилась")

        for code in codes:
            with self.subTest(code=code):
                self.assertIn(
                    f"'{code}'",
                    api_error_module,
                    f"код {code} не сопоставлен переводу в apiError.js: "
                    "пользователь увидит общий текст вместо причины",
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
