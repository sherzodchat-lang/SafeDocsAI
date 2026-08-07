"""Сведение редакционных разделов к рубрикам.

Главная проверка файла — про гомоглиф. ozodi пишет свою рубрику «Ҷомeа» с
ЛАТИНСКОЙ буквой e (U+0065) вместо кириллической е (U+0435). На экране это
неотличимо, для словаря — две разные строки, и рубрика из двух с лишним сотен
документов развалилась бы надвое без единого видимого признака. Такую ошибку
нельзя заметить чтением, поэтому она и проверяется машиной.

Вторая по важности — что «неизвестный раздел» и «раздел, решено не считать
темой» остаются РАЗНЫМИ состояниями. Свести их к одному значило бы позволить
новому разделу сайта тихо просочиться в корзину, вместо того чтобы потребовать
решения человека.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.modules.topics.pipeline.rubrics import (  # noqa: E402
    RAW_TO_RUBRIC,
    RUBRIC_BY_CODE,
    RUBRICS,
    UNLABELLED,
    normalize_section,
    rubric_names,
)


class HomoglyphTests(unittest.TestCase):
    def test_the_latin_e_inside_a_tajik_word_does_not_split_a_rubric(self):
        cyrillic = "Ҷомеа"
        with_latin_e = "Ҷомeа"
        self.assertNotEqual(cyrillic, with_latin_e)  # строки разные...
        self.assertEqual(  # ...а рубрика одна
            normalize_section(cyrillic), normalize_section(with_latin_e)
        )
        self.assertEqual(normalize_section(with_latin_e), "R02")

    def test_a_genuinely_latin_section_name_is_not_mangled(self):
        """Замена гомоглифов применяется только там, где кириллица уже есть.
        Иначе честно латинское название раздела превратилось бы в мешанину, и в
        отчёте о непокрытых разделах выглядело бы испорченным нами."""
        self.assertIsNone(normalize_section("Tajikistan style"))

    def test_case_and_spacing_do_not_matter(self):
        for written in ("Маориф", "маориф", "  МАОРИФ  ", "Ма ориф".replace(" ", "")):
            with self.subTest(section=written):
                self.assertEqual(normalize_section(written), "R05")

    def test_multiple_spaces_inside_a_name_collapse(self):
        self.assertEqual(normalize_section("Сиёсати   хориҷӣ"), "R08")


class ThreeOutcomesTests(unittest.TestCase):
    def test_a_known_section_gives_its_rubric(self):
        self.assertEqual(normalize_section("Қонунҳои ҶТ"), "R09")
        self.assertEqual(normalize_section("Паёмҳо"), "R10")

    def test_a_section_deliberately_ignored_is_marked_as_such(self):
        for section in ("Хабарҳо", "Мақолаҳо", ""):
            with self.subTest(section=section):
                self.assertEqual(normalize_section(section), UNLABELLED)

    def test_an_unknown_section_is_none_so_a_human_has_to_decide(self):
        """None, а не UNLABELLED: новый раздел сайта обязан потребовать решения,
        а не тихо уехать в корзину. Скрипт сборки называет такие поимённо."""
        self.assertIsNone(normalize_section("Совершенно новый раздел"))


class TableConsistencyTests(unittest.TestCase):
    def test_every_mapped_code_actually_exists(self):
        unknown = sorted({code for code in RAW_TO_RUBRIC.values()} - set(RUBRIC_BY_CODE))
        self.assertEqual(unknown, [], f"таблица ссылается на несуществующие рубрики: {unknown}")

    def test_every_rubric_has_at_least_one_section_pointing_at_it(self):
        """Рубрика без единого раздела — мёртвая строка в списке тем: кластера
        с ней не возникнет никогда, а пользователь увидит её в фильтре."""
        used = set(RAW_TO_RUBRIC.values())
        orphans = sorted(rubric.code for rubric in RUBRICS if rubric.code not in used)
        self.assertEqual(orphans, [])

    def test_codes_are_unique(self):
        codes = [rubric.code for rubric in RUBRICS]
        self.assertEqual(len(codes), len(set(codes)))

    def test_both_languages_are_filled_and_differ_from_the_code(self):
        for rubric in RUBRICS:
            with self.subTest(code=rubric.code):
                self.assertTrue(rubric.tg.strip())
                self.assertTrue(rubric.ru.strip())
                self.assertNotEqual(rubric.tg, rubric.code)
                self.assertNotEqual(rubric.ru, rubric.code)

    def test_names_are_not_duplicated_between_rubrics(self):
        """Две рубрики с одинаковым именем — ровно та беда, из-за которой всё и
        затевалось: в прежней модели «Образование» стояло у трёх тем подряд."""
        for language, names in (("tg", [r.tg for r in RUBRICS]), ("ru", [r.ru for r in RUBRICS])):
            with self.subTest(language=language):
                self.assertEqual(len(names), len(set(names)))

    def test_keys_are_already_canonical(self):
        """Ключ таблицы, записанный с заглавной буквы или с гомоглифом, не
        нашёлся бы никогда — и рубрика молча осталась бы пустой."""
        for key in RAW_TO_RUBRIC:
            with self.subTest(key=key):
                self.assertEqual(key, key.lower().strip())
                self.assertEqual(normalize_section(key), RAW_TO_RUBRIC[key])


class NamesTests(unittest.TestCase):
    def test_names_of_every_code(self):
        for rubric in RUBRICS:
            self.assertEqual(rubric_names(rubric.code), (rubric.tg, rubric.ru))

    def test_the_unlabelled_bucket_has_its_own_names(self):
        tg, ru = rubric_names(UNLABELLED)
        self.assertTrue(tg and ru)

    def test_an_unknown_code_is_refused_rather_than_named_somehow(self):
        with self.assertRaises(KeyError):
            rubric_names("R99")


class ProductionGenresTests(unittest.TestCase):
    """Две рубрики, ради которых всё затевалось: боевые документы системы —
    налоговый кодекс и паёмы президента, и раньше подходящих тем не было."""

    def test_laws_decrees_and_resolutions_land_in_one_rubric(self):
        for section in ("Қонунҳои ҶТ", "Фармонҳои Президент", "Қарорҳои Ҳукумат", "Ҳуқуқ"):
            with self.subTest(section=section):
                self.assertEqual(normalize_section(section), "R09")

    def test_addresses_speeches_and_the_presidential_chronicle_land_in_one_rubric(self):
        """jumhuriyat режет публичную деятельность президента дробно, khovar
        держит одним разделом «Президент». Развести их значило бы получить два
        кластера, различающихся только тем, какой сайт их напечатал."""
        for section in ("Паёмҳо", "Суханрониҳо", "Мулоқотҳо", "Сафарҳо", "Президент"):
            with self.subTest(section=section):
                self.assertEqual(normalize_section(section), "R10")


if __name__ == "__main__":
    unittest.main()
