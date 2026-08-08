"""
OCR Service — per-page OCR for mixed/scanned PDFs.

Returns TextBlock objects compatible with HybridChunker.
"""

import logging
import shutil
from typing import List

import pytesseract
from pdf2image import convert_from_path

from app.services.hybrid_chunker import TextBlock

logger = logging.getLogger(__name__)


class OCRService:
    # Minimum meaningful text length on a page (in characters).
    # Pages with less extractable text than this are considered scan-like.
    OCR_THRESHOLD_CHARS = 80

    @staticmethod
    def dependencies_available() -> bool:
        """Установлены ли системные пакеты, без которых OCR не работает.

        Проверка нужна ДО обработки: pytesseract и pdf2image — это python-
        обёртки, их наличие в venv ничего не говорит о самих tesseract и
        poppler. На стенде без них ocr_single_page падал на каждой странице,
        отказ гасился, и страницы-сканы молча выпадали из индекса при статусе
        документа 'indexed'. Не кэшируем: shutil.which — дёшево, а пакеты
        могут доставить без перезапуска процесса.
        """
        return bool(shutil.which("tesseract")) and bool(shutil.which("pdftoppm"))

    @staticmethod
    def page_needs_ocr(page_text: str, threshold: int | None = None) -> bool:
        """
        Check if a page's text layer is too sparse and needs OCR.

        Args:
            page_text: Text extracted from the page's text layer.
            threshold: Minimum number of non-whitespace characters to consider
                       the text layer adequate. Defaults to OCR_THRESHOLD_CHARS.
        """
        threshold = threshold or OCRService.OCR_THRESHOLD_CHARS
        stripped = (page_text or "").strip()
        if len(stripped) < threshold:
            return True
        # Additional check: if the ratio of letters/digits is very low
        # (e.g. garbage characters from broken text layers)
        alpha_count = sum(1 for c in stripped if c.isalnum())
        if len(stripped) > 0 and alpha_count / len(stripped) < 0.3:
            return True
        return False

    @staticmethod
    def ocr_single_page(file_path: str, page_num: int, lang: str = "rus+tgk") -> str:
        """
        OCR a single page of a PDF using Tesseract.

        Args:
            file_path: Path to the PDF file.
            page_num: 1-based page number.
            lang: Tesseract language code.

        Returns:
            Extracted text from OCR.
        """
        try:
            images = convert_from_path(
                file_path,
                first_page=page_num,
                last_page=page_num,
            )
            if images:
                return pytesseract.image_to_string(images[0], lang=lang)
        except Exception:
            # logger, а не print: print уходил в stdout процесса и в журнале
            # не оставался — потерянные страницы было не с чем связать.
            logger.warning(
                "OCR failed on page %s of %s", page_num, file_path, exc_info=True
            )
        return ""

    @staticmethod
    def extract_text_from_scanned_pdf(
        file_path: str, lang: str = "rus+tgk"
    ) -> List[TextBlock]:
        """
        Full-document OCR fallback. Converts every page to an image
        and runs Tesseract. Returns TextBlock objects.
        """
        blocks: List[TextBlock] = []
        try:
            images = convert_from_path(file_path)
            for i, image in enumerate(images):
                text = pytesseract.image_to_string(image, lang=lang)
                if text.strip():
                    blocks.append(TextBlock(
                        text=text,
                        page=i + 1,
                        order=i,
                        source="ocr",
                    ))
        except Exception:
            logger.warning(
                "Full-document OCR failed for %s", file_path, exc_info=True
            )
        return blocks
