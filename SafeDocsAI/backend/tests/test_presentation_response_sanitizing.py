"""Разбор ответа модели: неэкранированные управляющие символы.

На живом стенде презентация #4 (блокнот 17, ru, 10 слайдов) ушла в ошибку с
`response is not valid JSON: Invalid control character at (line 22, column 14)`:
модель поставила внутрь строкового значения живой перевод строки, и обе
попытки — исходная и повторная — упали на одном и том же символе.

Здесь проверяется шаг извлечения, который лечит именно этот случай, и, что не
менее важно, его безвредность: валидный ответ обязан пройти через чистку
НЕИЗМЕННЫМ, а мусор, который чисткой не лечится, — по-прежнему получить
честный отказ. Машинерия повтора не участвует: она отработала правильно, и
здесь её нет.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.modules.presentations.llm_schemas import (  # noqa: E402
    LlmResponseError,
    escape_control_characters,
    parse_model_json,
    validate_slide,
)

# Живые управляющие символы держим кодами: в исходнике теста они были бы
# невидимы, а тест про них ровно и написан.
LF = chr(0x0A)
TAB = chr(0x09)
ESC = chr(0x1B)
NUL = chr(0x00)
FORM_FEED = chr(0x0C)
CR = chr(0x0D)


class ControlCharacterRepairTests(unittest.TestCase):
    """Тот самый отказ: перевод строки внутри строкового значения."""

    def test_raw_newline_inside_a_string_no_longer_breaks_the_parse(self):
        raw = '{"heading": "Ставки НДС' + LF + 'и льготы", "slide": 4}'
        # Без чистки это ровно та ошибка, что легла в журнал стенда.
        with self.assertRaises(json.JSONDecodeError):
            json.loads(raw)
        payload = parse_model_json(raw)
        self.assertEqual(payload["heading"], "Ставки НДС" + LF + "и льготы")
        self.assertEqual(payload["slide"], 4)

    def test_newline_survives_as_a_newline(self):
        # Смысл сохранён: перенос строки в буллете остался переносом, а не
        # исчез и не склеил два слова.
        raw = '{"bullets": ["первая строка' + LF + 'вторая строка"]}'
        payload = parse_model_json(raw)
        self.assertEqual(payload["bullets"], ["первая строка" + LF + "вторая строка"])

    def test_tab_survives_as_a_tab(self):
        raw = '{"bullets": ["колонка' + TAB + 'значение"]}'
        payload = parse_model_json(raw)
        self.assertEqual(payload["bullets"], ["колонка" + TAB + "значение"])

    def test_junk_control_characters_are_dropped(self):
        # ESC, NUL и подача страницы смысла не несут; довезти их до рендерера
        # значит показать пользователю "_x001B_" прямо на слайде.
        raw = '{"heading": "Ставки' + ESC + NUL + FORM_FEED + ' НДС"}'
        payload = parse_model_json(raw)
        self.assertEqual(payload["heading"], "Ставки НДС")

    def test_windows_line_break_becomes_a_single_newline(self):
        raw = '{"heading": "первая' + CR + LF + 'вторая"}'
        payload = parse_model_json(raw)
        self.assertEqual(payload["heading"], "первая" + LF + "вторая")

    def test_multiline_plan_like_the_one_from_the_stand(self):
        raw = (
            "{\n"
            '  "title": "Налоги",\n'
            '  "sections": [\n'
            "    {\n"
            '      "heading": "Ставки' + LF + 'и льготы",\n'
            '      "search_query": "ставки НДС"\n'
            "    }\n"
            "  ]\n"
            "}"
        )
        payload = parse_model_json(raw)
        self.assertEqual(payload["sections"][0]["heading"], "Ставки" + LF + "и льготы")

    def test_repaired_response_reaches_the_schema(self):
        raw = (
            '{"layout": "bullets", "heading": "Ставки НДС", '
            '"bullets": ["первая' + LF + 'вторая", '
            '"второй факт"], "citations": [{"source_id": 7, "chunk_id": 45}]}'
        )
        slide = validate_slide(raw, allowed_citations={"45": 7})
        self.assertEqual(slide.bullets[0], "первая" + LF + "вторая")

    def test_preamble_path_is_repaired_too(self):
        # Второй проход (от первой { до последней }) разбирает уже вычищенный
        # текст — иначе лечение зависело бы от того, добавила ли модель
        # вступление.
        raw = 'Вот JSON:\n{"heading": "Ставки' + LF + 'НДС"}\nГотово.'
        payload = parse_model_json(raw)
        self.assertEqual(payload["heading"], "Ставки" + LF + "НДС")


class LegitimateResponsesAreUntouchedTests(unittest.TestCase):
    """Чистка обязана быть невидимой для всего, что и так валидно."""

    def test_valid_json_passes_through_byte_for_byte(self):
        raw = json.dumps(
            {
                "title": "Налоги",
                "sections": [{"heading": "Ставки", "search_query": "НДС"}],
            },
            ensure_ascii=False,
            indent=2,
        )
        self.assertEqual(escape_control_characters(raw), raw)
        self.assertEqual(parse_model_json(raw), json.loads(raw))

    def test_already_escaped_sequences_stay_as_they_are(self):
        # Внутри значения стоят два символа, обратный слэш и n, — это уже
        # корректный JSON, и трогать его нельзя.
        raw = '{"heading": "первая\\nвторая\\tтретья"}'
        self.assertEqual(escape_control_characters(raw), raw)
        payload = parse_model_json(raw)
        self.assertEqual(payload["heading"], "первая" + LF + "вторая" + TAB + "третья")

    def test_escaped_quotes_do_not_confuse_string_tracking(self):
        raw = '{"a": "он сказал \\"да\\"", "b": "путь C:\\\\", "c": "x' + LF + 'y"}'
        payload = parse_model_json(raw)
        self.assertEqual(payload["a"], 'он сказал "да"')
        self.assertEqual(payload["b"], "путь C:\\")
        self.assertEqual(payload["c"], "x" + LF + "y")

    def test_emoji_and_characters_outside_the_bmp_are_intact(self):
        raw = '{"heading": "Итоги 📈 роста 𝄞 и 漢字"}'
        self.assertEqual(escape_control_characters(raw), raw)
        payload = parse_model_json(raw)
        self.assertEqual(payload["heading"], "Итоги 📈 роста 𝄞 и 漢字")

    def test_structural_whitespace_outside_strings_is_left_alone(self):
        raw = "{" + LF + TAB + '"a": 1,' + LF + TAB + '"b": 2' + LF + "}"
        self.assertEqual(escape_control_characters(raw), raw)
        self.assertEqual(parse_model_json(raw), {"a": 1, "b": 2})

    def test_code_fences_are_still_stripped(self):
        raw = '```json\n{"heading": "Ставки НДС"}\n```'
        self.assertEqual(parse_model_json(raw), {"heading": "Ставки НДС"})

    def test_code_fences_and_a_control_character_together(self):
        raw = '```json\n{"heading": "Ставки' + LF + 'НДС"}\n```\nПояснение.'
        payload = parse_model_json(raw)
        self.assertEqual(payload["heading"], "Ставки" + LF + "НДС")


class UnfixableGarbageIsStillRejectedTests(unittest.TestCase):
    """Чистка лечит один класс поломок и не притворяется, что лечит все."""

    def test_plain_prose_is_rejected(self):
        with self.assertRaises(LlmResponseError) as ctx:
            parse_model_json("Извините, я не могу составить план.")
        self.assertIn("not valid JSON", str(ctx.exception))

    def test_truncated_object_is_rejected(self):
        raw = '{"heading": "Ставки' + LF + 'НДС", "bullets": ['
        with self.assertRaises(LlmResponseError) as ctx:
            parse_model_json(raw)
        self.assertIn("not valid JSON", str(ctx.exception))

    def test_trailing_comma_is_rejected(self):
        with self.assertRaises(LlmResponseError):
            parse_model_json('{"a": 1,}')

    def test_empty_response_is_rejected(self):
        with self.assertRaises(LlmResponseError) as ctx:
            parse_model_json("   ")
        self.assertIn("empty response", str(ctx.exception))

    def test_json_array_is_still_not_an_object(self):
        with self.assertRaises(LlmResponseError) as ctx:
            parse_model_json('["Ставки' + LF + 'НДС"]')
        self.assertIn("must be a JSON object", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
