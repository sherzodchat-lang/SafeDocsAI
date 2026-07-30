"""
OCR Service — per-page OCR for mixed/scanned PDFs.

Returns TextBlock objects compatible with HybridChunker.
"""

import pytesseract
from pdf2image import convert_from_path
from typing import List

from app.services.hybrid_chunker import TextBlock


class OCRService:
    # Minimum meaningful text length on a page (in characters).
    # Pages with less extractable text than this are considered scan-like.
    OCR_THRESHOLD_CHARS = 80

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
        except Exception as e:
            print(f"OCR Error on page {page_num}: {e}")
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
        except Exception as e:
            print(f"OCR Error: {e}")
        return blocks
