"""
Tests for the HybridChunker pipeline.
"""

import unittest

from app.services.hybrid_chunker import TextBlock, HybridChunker


class TestHybridChunkerBasics(unittest.TestCase):
    """Test core chunking behavior."""

    def setUp(self):
        self.chunker = HybridChunker(
            target_tokens=100,
            max_tokens=200,
            min_tokens=30,
            overlap_tokens=0,
        )

    def test_empty_input(self):
        result = self.chunker.chunk([])
        self.assertEqual(result, [])

    def test_single_short_block(self):
        blocks = [TextBlock(text="Короткий текст.", page=1, order=0, source="txt")]
        chunks = self.chunker.chunk(blocks)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Короткий текст.", chunks[0].text)
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertEqual(chunks[0].page_start, 1)
        self.assertEqual(chunks[0].page_end, 1)

    def test_chunk_index_sequential(self):
        """chunk_index values should be 0, 1, 2, ..."""
        blocks = [
            TextBlock(text=f"Параграф {i}. " * 50, page=1, order=i, source="txt")
            for i in range(10)
        ]
        chunks = self.chunker.chunk(blocks)
        self.assertTrue(len(chunks) > 1, "Expected multiple chunks")
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.chunk_index, i, f"Expected chunk_index={i}, got {chunk.chunk_index}")


class TestHeadingDetection(unittest.TestCase):
    """Test that legal headings (СТАТЬЯ, ГЛАВА etc.) are detected and force chunk boundaries."""

    def setUp(self):
        self.chunker = HybridChunker(
            target_tokens=200,
            max_tokens=400,
            min_tokens=30,
            overlap_tokens=0,
        )

    def test_article_header_splits(self):
        """Each СТАТЬЯ should start a new chunk."""
        blocks = [
            TextBlock(
                text="СТАТЬЯ 1\nТекст первой статьи. Описание налоговых правил для граждан.",
                page=1, order=0, source="txt",
            ),
            TextBlock(
                text="СТАТЬЯ 2\nТекст второй статьи. Другие правила для юридических лиц.",
                page=1, order=1, source="txt",
            ),
        ]
        chunks = self.chunker.chunk(blocks)
        self.assertTrue(len(chunks) >= 2, f"Expected >=2 chunks, got {len(chunks)}")
        self.assertIn("СТАТЬЯ 1", chunks[0].text)
        self.assertIn("СТАТЬЯ 2", chunks[1].text)

    def test_chapter_header_splits(self):
        """ГЛАВА should force a new chunk."""
        blocks = [
            TextBlock(text="ГЛАВА 1\nОбщие положения.", page=1, order=0, source="txt"),
            TextBlock(
                text="ГЛАВА 2\nСпециальные нормы. " * 10,
                page=2, order=1, source="txt",
            ),
        ]
        chunks = self.chunker.chunk(blocks)
        self.assertTrue(len(chunks) >= 2)
        self.assertIn("ГЛАВА 1", chunks[0].text)
        self.assertIn("ГЛАВА 2", chunks[1].text)

    def test_tajik_header_moddai(self):
        """Tajik МОДДАИ should be recognized as heading."""
        blocks = [
            TextBlock(text="МОДДАИ 1\nМатни моддаи аввал.", page=1, order=0, source="txt"),
            TextBlock(text="МОДДАИ 2\nМатни моддаи дуюм.", page=1, order=1, source="txt"),
        ]
        chunks = self.chunker.chunk(blocks)
        self.assertTrue(len(chunks) >= 2)

    def test_section_path_tracking(self):
        """Section path should track nested headings."""
        blocks = [
            TextBlock(text="ГЛАВА 1\nОбщие положения.", page=1, order=0, source="txt"),
            TextBlock(text="СТАТЬЯ 1\nОпределения.", page=1, order=1, source="txt"),
            TextBlock(text="Содержание статьи 1.", page=1, order=2, source="txt"),
        ]
        chunks = self.chunker.chunk(blocks)
        # The chunk containing СТАТЬЯ 1 should have section_path with ГЛАВА 1
        article_chunk = next((c for c in chunks if "СТАТЬЯ 1" in c.text), None)
        self.assertIsNotNone(article_chunk, "Expected a chunk with СТАТЬЯ 1")
        self.assertTrue(
            any("ГЛАВА 1" in s for s in article_chunk.section_path),
            f"Expected ГЛАВА 1 in section_path, got {article_chunk.section_path}",
        )


class TestTokenBounds(unittest.TestCase):
    """Test that chunks respect token size limits."""

    def test_chunks_within_bounds(self):
        chunker = HybridChunker(
            target_tokens=100,
            max_tokens=200,
            min_tokens=30,
            overlap_tokens=0,
        )
        # Generate a large amount of text
        text = "Информация о налогах Республики Таджикистан. " * 200
        blocks = [TextBlock(text=text, page=1, order=0, source="txt")]
        chunks = chunker.chunk(blocks)
        self.assertTrue(len(chunks) > 1, "Expected multiple chunks for large text")

        for chunk in chunks:
            tokens = chunker._estimate_tokens(chunk.text)
            self.assertLessEqual(
                tokens,
                chunker.max_tokens + 50,  # small tolerance for boundary effects
                f"Chunk too large: {tokens} tokens (max {chunker.max_tokens})",
            )

    def test_long_paragraph_split_at_sentences(self):
        """Oversized paragraphs should be split at sentence boundaries."""
        chunker = HybridChunker(
            target_tokens=50,
            max_tokens=100,
            min_tokens=20,
            overlap_tokens=0,
        )
        text = "Первое предложение налогового кодекса. " * 30
        blocks = [TextBlock(text=text, page=1, order=0, source="txt")]
        chunks = chunker.chunk(blocks)
        self.assertTrue(len(chunks) > 1, "Expected split for oversized paragraph")

    def test_small_chunk_is_flushed_before_overflow(self):
        """Current chunk must not exceed max_tokens when the next unit would overflow it."""
        chunker = HybridChunker(
            target_tokens=60,
            max_tokens=100,
            min_tokens=80,
            overlap_tokens=0,
        )
        blocks = [
            TextBlock(text="A" * 280, page=1, order=0, source="txt"),
            TextBlock(text="B" * 120, page=1, order=1, source="txt"),
        ]

        chunks = chunker.chunk(blocks)

        self.assertEqual(len(chunks), 2)
        for chunk in chunks:
            tokens = chunker._estimate_tokens(chunk.text)
            self.assertLessEqual(tokens, chunker.max_tokens)

    def test_hard_split_without_sentence_boundaries_respects_max_tokens(self):
        """Character fallback splitting should still keep every chunk within max_tokens."""
        chunker = HybridChunker(
            target_tokens=60,
            max_tokens=100,
            min_tokens=20,
            overlap_tokens=0,
        )
        blocks = [TextBlock(text="X" * 900, page=1, order=0, source="txt")]

        chunks = chunker.chunk(blocks)

        self.assertTrue(len(chunks) > 1)
        for chunk in chunks:
            tokens = chunker._estimate_tokens(chunk.text)
            self.assertLessEqual(tokens, chunker.max_tokens)

    def test_max_chars_limit_is_enforced(self):
        """Chunks should also respect an explicit character ceiling."""
        chunker = HybridChunker(
            target_tokens=80,
            max_tokens=300,
            min_tokens=20,
            overlap_tokens=0,
            max_chars=120,
        )
        blocks = [TextBlock(text="Текст без точек " * 40, page=1, order=0, source="txt")]

        chunks = chunker.chunk(blocks)

        self.assertTrue(len(chunks) > 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 120)


class TestListItems(unittest.TestCase):
    """Test that list items are handled properly."""

    def test_list_items_detected(self):
        chunker = HybridChunker(
            target_tokens=200,
            max_tokens=400,
            min_tokens=30,
            overlap_tokens=0,
        )
        blocks = [
            TextBlock(
                text="Условия:\n- пункт первый\n- пункт второй\n- пункт третий",
                page=1, order=0, source="txt",
            ),
        ]
        chunks = chunker.chunk(blocks)
        # List items should be kept together in a single chunk (they're small)
        self.assertEqual(len(chunks), 1)
        self.assertIn("пункт первый", chunks[0].text)
        self.assertIn("пункт третий", chunks[0].text)


class TestHeaderFooterRemoval(unittest.TestCase):
    """Test that repeating headers/footers are removed."""

    def test_removes_repeated_lines(self):
        chunker = HybridChunker(
            target_tokens=100,
            max_tokens=200,
            min_tokens=20,
            overlap_tokens=0,
        )
        # Simulate 10 pages with same header/footer
        blocks = []
        for page in range(1, 11):
            blocks.append(TextBlock(
                text="Налоговый кодекс РТ",  # repeating header
                page=page, order=page * 3, source="pymupdf",
            ))
            blocks.append(TextBlock(
                text=f"Содержание страницы {page} с уникальным текстом.",
                page=page, order=page * 3 + 1, source="pymupdf",
            ))
            blocks.append(TextBlock(
                text="Страница документа",  # repeating footer
                page=page, order=page * 3 + 2, source="pymupdf",
            ))

        chunks = chunker.chunk(blocks)
        all_text = " ".join(c.text for c in chunks)
        # The repeated header/footer should be removed (or appear very few times)
        header_count = all_text.lower().count("налоговый кодекс рт")
        self.assertLessEqual(header_count, 2, f"Header repeated {header_count} times, expected <=2")


class TestOverlap(unittest.TestCase):
    """Test that overlap is applied when configured."""

    def test_overlap_prefix(self):
        chunker = HybridChunker(
            target_tokens=50,
            max_tokens=100,
            min_tokens=20,
            overlap_tokens=20,
        )
        text = "Предложение номер один. " * 30
        blocks = [TextBlock(text=text, page=1, order=0, source="txt")]
        chunks = chunker.chunk(blocks)
        if len(chunks) > 1:
            # Second chunk should start with "..." indicating overlap
            self.assertTrue(
                chunks[1].text.startswith("..."),
                "Expected overlap prefix '...' in consecutive chunks",
            )


class TestMixedPdfOcr(unittest.TestCase):
    """Test OCR-related utilities."""

    def test_page_needs_ocr_sparse(self):
        from app.services.ocr_service import OCRService
        self.assertTrue(OCRService.page_needs_ocr(""))
        self.assertTrue(OCRService.page_needs_ocr("abc"))
        self.assertTrue(OCRService.page_needs_ocr("   "))

    def test_page_needs_ocr_normal(self):
        from app.services.ocr_service import OCRService
        normal_text = "Налоговый кодекс Республики Таджикистан. " * 5
        self.assertFalse(OCRService.page_needs_ocr(normal_text))

    def test_page_needs_ocr_garbage(self):
        """Text with very few alpha chars should trigger OCR."""
        from app.services.ocr_service import OCRService
        garbage = "####$$$$@@@@!!!!" * 10
        self.assertTrue(OCRService.page_needs_ocr(garbage))


class TestDocumentServiceExtraction(unittest.TestCase):
    """Test document extraction to TextBlocks."""

    def test_extract_blocks_from_txt(self):
        """TXT extraction should split by paragraphs."""
        import tempfile
        import os
        from app.services.document_service import DocumentService

        content = "Первый параграф.\n\nВторой параграф.\n\nТретий параграф."
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            tmp_path = f.name

        try:
            blocks = DocumentService.extract_blocks(tmp_path, ".txt")
            self.assertEqual(len(blocks), 3)
            self.assertEqual(blocks[0].text, "Первый параграф.")
            self.assertEqual(blocks[1].text, "Второй параграф.")
            self.assertEqual(blocks[2].text, "Третий параграф.")
            self.assertTrue(all(b.source == "txt" for b in blocks))
        finally:
            os.unlink(tmp_path)

    def test_extract_and_chunk_txt(self):
        """Full pipeline: TXT → blocks → chunks."""
        import tempfile
        import os
        from app.services.document_service import DocumentService

        content = ("Налоговый кодекс. " * 100).strip()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            tmp_path = f.name

        try:
            chunks = DocumentService.extract_and_chunk(tmp_path, ".txt")
            self.assertTrue(len(chunks) >= 1, "Expected at least one chunk")
            for chunk in chunks:
                self.assertIsNotNone(chunk.chunk_index)
                self.assertEqual(chunk.page_start, 1)
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
