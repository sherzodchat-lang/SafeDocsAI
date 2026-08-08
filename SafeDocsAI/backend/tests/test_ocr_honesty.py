"""Честность извлечения PDF со страницами-сканами.

До правки страница-скан при недоступном OCR молча пропускалась: документ
получал статус 'indexed', а часть содержимого не существовала для поиска —
«не нашлось» было неотличимо от «не индексировалось». Отказ был виден только
в stdout через print, то есть нигде.
"""

import io
import os
import tempfile
import unittest
from unittest.mock import patch

import fitz

from app.core.exceptions import SourceErrors
from app.services.document_service import DocumentService, UploadValidationError
from app.services.ocr_service import OCRService

# Достаточно длинный текст, чтобы страница не считалась сканом
# (порог OCR_THRESHOLD_CHARS = 80 непробельных символов).
_PAGE_TEXT = (
    "Obychnaya stranitsa s polnotsennym tekstovym sloem, kotoraya ne "
    "trebuet raspoznavaniya i popadaet v indeks tselikom bez poter."
)


def _png_bytes() -> bytes:
    from PIL import Image

    image = Image.new("RGB", (64, 64), color=(120, 120, 120))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _make_pdf(pages: list[str]) -> str:
    """PDF из страниц трёх видов: 'text', 'image' (скан), 'blank'."""
    doc = fitz.open()
    for kind in pages:
        page = doc.new_page()
        if kind == "text":
            page.insert_text((72, 72), _PAGE_TEXT, fontsize=11)
        elif kind == "image":
            page.insert_image(fitz.Rect(0, 0, 300, 300), stream=_png_bytes())
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    doc.save(path)
    doc.close()
    return path


class ScannedPagesWithoutOcrTests(unittest.TestCase):
    def setUp(self):
        self._paths: list[str] = []

    def tearDown(self):
        for path in self._paths:
            if os.path.exists(path):
                os.remove(path)

    def _pdf(self, pages: list[str]) -> str:
        path = _make_pdf(pages)
        self._paths.append(path)
        return path

    def test_scan_page_without_ocr_refuses_with_code(self):
        """Скан + недоступный OCR — отказ с кодом, а не индекс с дырой."""
        path = self._pdf(["text", "image"])
        with patch.object(OCRService, "dependencies_available", return_value=False):
            with self.assertRaises(UploadValidationError) as ctx:
                DocumentService.extract_blocks(path, ".pdf")
        self.assertEqual(ctx.exception.error_code, SourceErrors.OCR_UNAVAILABLE)

    def test_blank_page_is_not_treated_as_scan(self):
        """Пустая страница (без картинок) — не скан: распознавать нечего,
        пропуск честен и отказа не заслуживает."""
        path = self._pdf(["text", "blank"])
        with patch.object(OCRService, "dependencies_available", return_value=False):
            blocks = DocumentService.extract_blocks(path, ".pdf")
        self.assertTrue(blocks)
        self.assertTrue(all(b.page == 1 for b in blocks))

    def test_scan_page_with_ocr_available_goes_through_ocr(self):
        path = self._pdf(["text", "image"])
        with patch.object(OCRService, "dependencies_available", return_value=True), \
                patch.object(
                    OCRService, "ocr_single_page", return_value="Matni sahifai skan"
                ) as ocr_mock:
            blocks = DocumentService.extract_blocks(path, ".pdf")
        ocr_mock.assert_called_once()
        ocr_blocks = [b for b in blocks if b.source == "ocr"]
        self.assertEqual(len(ocr_blocks), 1)
        self.assertEqual(ocr_blocks[0].page, 2)


class OcrFailureLoggingTests(unittest.TestCase):
    def test_ocr_failure_reaches_the_log(self):
        """Отказ OCR обязан оставлять след в журнале, а не в stdout."""
        with patch(
            "app.services.ocr_service.convert_from_path",
            side_effect=RuntimeError("poppler is missing"),
        ):
            with self.assertLogs("app.services.ocr_service", level="WARNING"):
                result = OCRService.ocr_single_page("/tmp/nonexistent.pdf", 1)
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
