"""Ссылка «статья N» и её таджикская форма «моддаи N» — одна сущность.

Живой замер: вопрос «Что говорит статья 91 Трудового кодекса?» не находил
«Моддаи 91» — буст и лексика сравнивали строки буквально, а заодно
подстрочное `in` засчитывало «статью 91» тексту про статью 910.
"""

import unittest

from app.modules.rag.text_utils import (
    article_reference_variants,
    boost_article_chunks,
    contains_article_reference,
)


class ArticleReferenceVariantsTests(unittest.TestCase):
    def test_russian_reference_gains_tajik_twin(self):
        self.assertIn("моддаи 91", article_reference_variants("статья 91"))

    def test_tajik_reference_gains_russian_twin(self):
        self.assertIn("статья 91", article_reference_variants("моддаи 91"))

    def test_law_reference_has_no_twin(self):
        # У «закона»/«пункта» одно-однозначной пары нет — вариант один.
        self.assertEqual(article_reference_variants("закон 14"), ["закон 14"])

    def test_empty_reference(self):
        self.assertEqual(article_reference_variants(""), [])


class ContainsArticleReferenceTests(unittest.TestCase):
    def test_russian_reference_matches_tajik_text(self):
        text = "моддаи 91. намудҳои рухсатии меҳнатӣ"
        self.assertTrue(contains_article_reference(text, "статья 91"))

    def test_tajik_reference_matches_russian_text(self):
        self.assertTrue(
            contains_article_reference("согласно статья 91 кодекса", "моддаи 91")
        )

    def test_number_prefix_is_not_a_match(self):
        # «статья 910» — другая статья; до правки `in` засчитывал совпадение.
        self.assertFalse(
            contains_article_reference("статья 910 регулирует иное", "статья 91")
        )

    def test_whitespace_between_word_and_number(self):
        # В тексте из PDF между словом и номером бывает перенос строки.
        self.assertTrue(
            contains_article_reference("моддаи\n91. рухсатӣ", "статья 91")
        )


class BoostArticleChunksCrossFormTests(unittest.TestCase):
    def _results(self, texts):
        return {
            "documents": [list(texts)],
            "ids": [[str(i) for i in range(len(texts))]],
            "metadatas": [[{} for _ in texts]],
            "distances": [[0.8 for _ in texts]],
        }

    def test_tajik_article_is_boosted_for_russian_reference(self):
        results = self._results(
            [
                "текст без статьи вообще",
                "Моддаи 91. Намудҳои рухсатии меҳнатӣ",
            ]
        )
        boosted = boost_article_chunks(results, "статья 91")
        self.assertEqual(boosted["documents"][0][0], "Моддаи 91. Намудҳои рухсатии меҳнатӣ")
        # Бустнутому чанку режется расстояние — признак, что ветка сработала.
        self.assertAlmostEqual(boosted["distances"][0][0], 0.4)

    def test_other_article_number_is_not_boosted(self):
        results = self._results(
            [
                "текст без статьи вообще",
                "статья 910 не имеет отношения к делу",
            ]
        )
        boosted = boost_article_chunks(results, "статья 91")
        self.assertEqual(boosted["documents"][0][0], "текст без статьи вообще")
        self.assertAlmostEqual(boosted["distances"][0][0], 0.8)


if __name__ == "__main__":
    unittest.main()
