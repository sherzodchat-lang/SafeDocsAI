"""Подписи кластеров: находит ли c-TF-IDF то, чем кластеры отличаются.

Тексты синтетические и короткие: правильный ответ должен быть виден из самого
теста, иначе «слово выбрано верно» не проверяется, а принимается на веру.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.modules.topics.labeling import (  # noqa: E402
    STOPWORDS,
    c_tf_idf,
    compose_labels,
    tokenize,
    tokenize_all,
)


class TokenizeTests(unittest.TestCase):
    def test_tajik_letters_stay_inside_words(self):
        """Ғ, Ӣ, Қ, Ӯ, Ҳ, Ҷ лежат в кириллическом блоке вразнобой, и диапазон
        «а-я» их не покрывает. Без явного перечисления «Ҷумҳурӣ» распалось бы
        на обрывки, и характерным словом кластера законов стало бы «ум»."""
        self.assertEqual(
            tokenize("Ҷумҳурии Тоҷикистон қонунро қабул кард"),
            ["ҷумҳурии", "тоҷикистон", "қонунро", "қабул"],
        )

    def test_stopwords_and_digits_are_dropped(self):
        self.assertEqual(tokenize("дар соли 2024 ва барои мактаб"), ["соли", "мактаб"])

    def test_short_words_are_dropped(self):
        self.assertEqual(tokenize("аб вгд ежзи"), ["вгд", "ежзи"])

    def test_case_is_folded(self):
        self.assertEqual(tokenize("МАКТАБ Мактаб мактаб"), ["мактаб"] * 3)

    def test_stopword_list_covers_both_languages(self):
        self.assertIn("мебошад", STOPWORDS)
        self.assertIn("который", STOPWORDS)


class CharacteristicTermsTests(unittest.TestCase):
    def test_a_word_unique_to_one_group_beats_a_word_shared_by_all(self):
        groups = {
            "спорт": [tokenize("тим тим тим футбол футбол варзиш")] * 3,
            "школа": [tokenize("мактаб мактаб мактаб талаба талаба варзиш")] * 3,
        }
        terms = c_tf_idf(groups, top_n=3)
        sport = [term.term for term in terms["спорт"]]
        self.assertIn("тим", sport)
        # «варзиш» есть в обеих группах, поэтому характерным быть не должно.
        self.assertNotIn("варзиш", sport[:1])

    def test_a_word_present_everywhere_is_pushed_down_without_a_blacklist(self):
        """«Тоҷикистон» встречается в каждом документе корпуса. Списка
        запрещённых слов у нас нет намеренно — давить такое обязана сама мера."""
        common = "тоҷикистон " * 5
        groups = {
            "а": [tokenize(common + "барқ барқ барқ нерӯгоҳ нерӯгоҳ")],
            "б": [tokenize(common + "мактаб мактаб мактаб донишҷӯ донишҷӯ")],
            "в": [tokenize(common + "варзиш варзиш варзиш тим тим")],
        }
        terms = c_tf_idf(groups, top_n=2)
        for key in groups:
            with self.subTest(group=key):
                self.assertNotIn("тоҷикистон", [term.term for term in terms[key]])

    def test_rare_word_does_not_win_on_a_huge_idf(self):
        """У слова из единственной статьи idf огромен. Без min_count оно
        вынесло бы наверх опечатку или фамилию из одного документа."""
        groups = {
            "а": [tokenize("иқтисод иқтисод иқтисод сармоя сармоя опечатканиквстречалась")],
            "б": [tokenize("мактаб мактаб мактаб")],
        }
        terms = c_tf_idf(groups, top_n=3, min_count=2)
        self.assertNotIn("опечатканиквстречалась", [term.term for term in terms["а"]])

    def test_a_word_common_to_the_whole_corpus_does_not_distinguish_twins(self):
        """Замер на настоящем корпусе: в подпись попало «Законодательство и
        право — ҷумҳурии, тоҷикистон». «Таджикистан» стоит в каждом документе,
        но внутри семьи из двух кластеров он поделён 70 на 30 и оттого выглядит
        различающим. Лечится тем, что idf берётся по ВСЕМ кластерам, а tf — по
        семье: это два разных вопроса, и мерить их по одному набору нельзя.
        """
        family = {
            0: [tokenize("тоҷикистон " * 7 + "барқ барқ барқ")] * 3,
            1: [tokenize("тоҷикистон " * 3 + "пахта пахта пахта")] * 3,
        }
        everything = dict(family)
        everything[2] = [tokenize("тоҷикистон " * 8 + "мактаб мактаб")] * 3
        everything[3] = [tokenize("тоҷикистон " * 8 + "варзиш варзиш")] * 3

        chosen = [term.term for term in c_tf_idf(family, top_n=2, idf_groups=everything)[0]]
        self.assertNotIn("тоҷикистон", chosen)
        self.assertEqual(chosen, ["барқ"])

    def test_a_word_absent_from_the_idf_set_is_skipped_not_infinite(self):
        """Слово, которого нет в наборе для idf, дало бы деление на ноль, то
        есть слово-победитель, взявшееся из ниоткуда."""
        family = {0: [tokenize("барқ барқ нерӯгоҳ нерӯгоҳ")] * 2}
        elsewhere = {9: [tokenize("мактаб мактаб талаба талаба")] * 2}
        self.assertEqual(c_tf_idf(family, top_n=3, idf_groups=elsewhere)[0], [])

    def test_ties_are_broken_by_the_word_so_two_builds_agree(self):
        groups = {"а": [tokenize("аба аба бвб бвб вгв вгв")]}
        first = [term.term for term in c_tf_idf(groups, top_n=3)["а"]]
        second = [term.term for term in c_tf_idf(dict(groups), top_n=3)["а"]]
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first))

    def test_empty_group_gets_an_empty_list_not_a_crash(self):
        terms = c_tf_idf({"пусто": [], "есть": [tokenize("мактаб мактаб талаба талаба")]})
        self.assertEqual(terms["пусто"], [])
        self.assertTrue(terms["есть"])

    def test_everything_empty_is_not_a_crash_either(self):
        self.assertEqual(c_tf_idf({"а": [], "б": [[]]}), {"а": [], "б": []})

    def test_top_n_must_be_positive(self):
        with self.assertRaises(ValueError):
            c_tf_idf({"а": [tokenize("мактаб мактаб")]}, top_n=0)


class ComposeLabelsTests(unittest.TestCase):
    NAMES = {"R01": "Иқтисод", "R05": "Маориф"}

    def test_a_lone_cluster_is_named_by_its_rubric_alone(self):
        labels = compose_labels(
            {0: "R05"}, {0: [tokenize("мактаб мактаб талаба талаба")]}, self.NAMES
        )
        self.assertEqual(labels[0], "Маориф")

    def test_twins_get_told_apart_and_never_share_a_name(self):
        """Ровно та беда, ради которой файл написан: в прежней модели
        «Образование» стояло у трёх кластеров подряд."""
        tokens = {
            0: [tokenize("барқ барқ барқ нерӯгоҳ нерӯгоҳ иқтисод")] * 4,
            1: [tokenize("кишоварзӣ кишоварзӣ кишоварзӣ пахта пахта иқтисод")] * 4,
        }
        labels = compose_labels({0: "R01", 1: "R01"}, tokens, self.NAMES, terms_in_label=1)
        self.assertNotEqual(labels[0], labels[1])
        self.assertTrue(labels[0].startswith("Иқтисод — "))
        self.assertIn("барқ", labels[0])
        self.assertIn("кишоварзӣ", labels[1])

    def test_qualifiers_are_computed_inside_the_family_not_across_the_corpus(self):
        """Уточнение обязано различать близнецов МЕЖДУ СОБОЙ. Слово, общее для
        обоих кластеров рубрики, уточнением быть не может, даже если по всему
        корпусу оно для этой рубрики характерно."""
        shared = "иқтисод " * 6
        tokens = {
            0: [tokenize(shared + "барқ барқ барқ")] * 3,
            1: [tokenize(shared + "пахта пахта пахта")] * 3,
            2: [tokenize("мактаб мактаб талаба талаба")] * 3,
        }
        labels = compose_labels(
            {0: "R01", 1: "R01", 2: "R05"}, tokens, self.NAMES, terms_in_label=2
        )
        self.assertNotIn("иқтисод", labels[0].split("—", 1)[1])
        self.assertNotIn("иқтисод", labels[1].split("—", 1)[1])
        self.assertIn("барқ", labels[0])
        self.assertIn("пахта", labels[1])
        self.assertEqual(labels[2], "Маориф")

    def test_a_word_is_not_handed_to_two_twins_at_once(self):
        tokens = {
            0: [tokenize("сомонӣ сомонӣ сомонӣ бонк бонк")] * 3,
            1: [tokenize("сомонӣ сомонӣ сомонӣ қарз қарз")] * 3,
        }
        labels = compose_labels({0: "R01", 1: "R01"}, tokens, self.NAMES, terms_in_label=1)
        self.assertNotEqual(labels[0], labels[1])

    def test_the_label_takes_no_more_words_than_asked(self):
        """Без ограничения в подпись уходили все уцелевшие после отбора слова, и
        вместо «Иқтисод — барқ, нерӯгоҳ» получалось перечисление в полстроки."""
        tokens = {
            0: [tokenize("барқ барқ нерӯгоҳ нерӯгоҳ обанбор обанбор нерӯ нерӯ")] * 3,
            1: [tokenize("пахта пахта гандум гандум кишт кишт замин замин")] * 3,
        }
        labels = compose_labels({0: "R01", 1: "R01"}, tokens, self.NAMES, terms_in_label=2)
        self.assertEqual(labels[0].split("—", 1)[1].count(","), 1)

    def test_twins_split_evenly_get_corpus_words_rather_than_numbers(self):
        """У близнецов, поделивших словарь ровно, отличительных слов МЕЖДУ СОБОЙ
        нет. Но сказать о кластере что-то осмысленное всё равно можно — по его
        словам на фоне всего корпуса. «Экономика #4» не говорит ничего."""
        # Четыре кластера, а не три: потолок распространённости — половина от их
        # числа, и на трёх он вырождается в «слово не должно встречаться больше
        # чем в одном», то есть запрещает вообще всё общее.
        shared = "сомонӣ бонк қарз " * 4
        tokens = {
            0: [tokenize(shared)] * 3,
            1: [tokenize(shared)] * 3,
            2: [tokenize("мактаб мактаб талаба талаба")] * 3,
            3: [tokenize("варзиш варзиш тим тим")] * 3,
        }
        labels = compose_labels(
            {0: "R01", 1: "R01", 2: "R05", 3: "R04"}, tokens, {**self.NAMES, "R04": "Варзиш"}
        )
        self.assertNotEqual(labels[0], labels[1])
        self.assertNotIn("#", labels[0])
        self.assertIn("сомонӣ", labels[0] + labels[1])

    def test_the_same_word_in_two_forms_does_not_take_the_whole_label(self):
        """Замер на настоящем корпусе: «Послания и выступления Президента —
        конфронси, конфронс». Таджикский маркирует изафет суффиксом, приведения
        к начальной форме у нас нет, и оба варианта одного слова уходили в
        подпись — половина уточнения не говорила ничего."""
        tokens = {
            0: [tokenize("конфронс конфронси конфронс конфронси симпозиум симпозиум")] * 3,
            1: [tokenize("кишоварзӣ кишоварзӣ пахта пахта")] * 3,
            2: [tokenize("мактаб мактаб талаба талаба")] * 3,
            3: [tokenize("варзиш варзиш тим тим")] * 3,
        }
        labels = compose_labels(
            {0: "R01", 1: "R01", 2: "R05", 3: "R04"},
            tokens,
            {**self.NAMES, "R04": "Варзиш"},
            terms_in_label=2,
        )
        qualifier = labels[0].split("—", 1)[1]
        self.assertNotIn("конфронси", qualifier.replace("конфронс,", ""))
        self.assertIn("симпозиум", qualifier)

    def test_a_cluster_that_barely_matches_its_rubric_is_not_named_by_it(self):
        """Кластер, где преобладающая рубрика набирает 19%, рубрикой не
        является: назвать его «Послания Президента» значит соврать четырём
        пятым его содержимого."""
        tokens = {
            0: [tokenize("мебел мебел курсӣ курсӣ")] * 3,
            1: [tokenize("мактаб мактаб талаба талаба")] * 3,
            2: [tokenize("варзиш варзиш тим тим")] * 3,
            3: [tokenize("пахта пахта кишт кишт")] * 3,
        }
        labels = compose_labels(
            {0: "R01", 1: "R05", 2: "R05", 3: "R05"},
            tokens,
            self.NAMES,
            share_of_cluster={0: 0.19, 1: 0.8, 2: 0.8, 3: 0.8},
            mixed_name="Разное",
        )
        self.assertTrue(labels[0].startswith("Разное"), labels[0])
        self.assertIn("мебел", labels[0])
        # Сильные кластеры своё имя сохраняют.
        self.assertTrue(labels[1].startswith("Маориф"), labels[1])

    def test_twins_without_any_word_at_all_fall_back_to_numbers(self):
        """Два одинаковых имени хуже некрасивого: по ним нельзя перейти к
        документам, а именно за этим в список тем и заходят."""
        tokens = {0: [], 1: []}
        labels = compose_labels({0: "R01", 1: "R01"}, tokens, self.NAMES)
        self.assertNotEqual(labels[0], labels[1])
        self.assertIn("#", labels[0])

    def test_unknown_rubric_code_falls_back_to_the_code_itself(self):
        labels = compose_labels({0: "R99"}, {0: [tokenize("мактаб мактаб")]}, self.NAMES)
        self.assertEqual(labels[0], "R99")

    def test_every_cluster_gets_exactly_one_label(self):
        rubrics = {0: "R01", 1: "R01", 2: "R05", 3: "R05", 4: "R05"}
        tokens = {
            index: [tokenize(f"слово{index} слово{index} умумӣ умумӣ")] * 3 for index in rubrics
        }
        labels = compose_labels(rubrics, tokens, self.NAMES)
        self.assertEqual(sorted(labels), sorted(rubrics))
        self.assertEqual(len(set(labels.values())), len(rubrics))


class TokenizeAllTests(unittest.TestCase):
    def test_matches_tokenize_row_by_row(self):
        texts = ["мактаб талаба", "барқ нерӯгоҳ"]
        self.assertEqual(tokenize_all(texts), [tokenize(text) for text in texts])


if __name__ == "__main__":
    unittest.main()
