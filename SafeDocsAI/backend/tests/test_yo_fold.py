"""Свёртка «ѐ» (U+0450) в «ё»: артефакт текстового слоя государственных PDF.

Замер по базе: 1858 из 7729 чанков (24%) содержат «ѐ». Токенизатор эту букву
не знал, и «меъѐр» разваливался на «меъ» и «р» — слово исчезало из BM25 при
любой словоформе в запросе.
"""

import unittest

from app.modules.rag.text_utils import fold_yo_variants, normalize_query, tokenize
from app.services.hybrid_chunker import HybridChunker, TextBlock


class YoFoldTextUtilsTests(unittest.TestCase):
    def test_fold_both_cases(self):
        self.assertEqual(fold_yo_variants("меъѐр Ѐлка"), "меъёр Ёлка")

    def test_tokenize_treats_variants_identically(self):
        self.assertEqual(tokenize("меъѐри андоз"), tokenize("меъёри андоз"))

    def test_tokenize_does_not_split_word_on_variant_letter(self):
        # До правки «меъѐр» давал обломки «меъ» и «р» вместо одной основы.
        # Первым токенизатор кладёт саму основу, дальше — её 3-граммы
        # (страховка от опечаток), поэтому проверяем первый элемент.
        tokens = tokenize("меъѐр")
        self.assertEqual(tokens[0], "меъёр")

    def test_normalize_query_folds_variant(self):
        self.assertEqual(normalize_query("Меъѐри андоз"), "меъёри андоз")


class YoFoldChunkerTests(unittest.TestCase):
    def test_chunk_text_is_folded_at_ingestion(self):
        # Сторона индексации: у уже лежащих в базе чанков это исправит только
        # переиндексация, но новые документы обязаны ложиться чистыми.
        chunker = HybridChunker(
            target_tokens=90, max_tokens=180, min_tokens=50, overlap_tokens=0
        )
        blocks = [
            TextBlock(
                text="Меъѐри андоз аз даромад бист фоиз муқаррар карда мешавад.",
                page=1,
                order=0,
                source="txt",
            )
        ]
        chunks = chunker.chunk(blocks)
        self.assertTrue(chunks)
        joined = " ".join(c.text for c in chunks)
        self.assertNotIn("ѐ", joined)
        self.assertIn("Меъёри", joined)


if __name__ == "__main__":
    unittest.main()
