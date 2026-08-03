"""
Document Service — unified extraction pipeline.

All document types (PDF, DOCX, TXT) are converted to TextBlock objects,
then chunked with HybridChunker. No more dual-path agentic/semantic branching.
"""

import contextlib
import errno
import logging
import os
import re
import uuid
from pathlib import Path, PurePosixPath
from typing import List

import fitz  # PyMuPDF
from fastapi import UploadFile

from app.core.config import settings
from app.core.exceptions import SourceErrors
from app.services.hybrid_chunker import TextBlock, ChunkResult, HybridChunker
from app.services.ocr_service import OCRService

logger = logging.getLogger(__name__)

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Читаем загрузку кусками: файл в 50 МБ не должен оказаться в памяти целиком.
UPLOAD_CHUNK_SIZE = 1024 * 1024

# Предел длины ОДНОГО элемента пути (NAME_MAX), а не пути целиком: 255 байт у
# ext4, tmpfs и большинства файловых систем Linux. Тем же числом ограничиваем
# document.name — колонка под btree-индексом, и строка в тысячи байт роняет
# INSERT по размеру индексной записи (та же причина, что у deps.TITLE_MAX_LENGTH).
MAX_NAME_BYTES = 255

# uuid4().hex (32 символа) + "_" перед именем: префикс съедает 33 байта из
# лимита, и пользователю остаётся 222. Ровно на этой границе загрузка и падала
# с OSError [Errno 36] вместо осмысленного отказа.
STORED_NAME_PREFIX_BYTES = 33

# Управляющие символы, которых в текстовом файле быть не должно (\t \n \r \f \v
# разрешены). По ним отличаем текст от бинарника, переименованного в .txt.
_BINARY_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0e-\x1f\x7f]")

# Управляющие символы в имени файла: RFC 7578 имя ничем не ограничивает, а
# уходит оно и в список источников, и в заголовок Content-Disposition.
_CONTROL_CHARS_IN_NAME = re.compile(r"[\x00-\x1f\x7f]")


class UploadValidationError(ValueError):
    """Отказ при приёме или разборе файла, с машинным кодом для клиента.

    Наследуется от ValueError: вызывающий код уже ловит ValueError, а
    error_code читается через getattr и не обязателен.
    """

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


class DocumentService:
    ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
    GENERIC_MIME_TYPES = {"application/octet-stream", "binary/octet-stream"}
    MIME_BY_EXTENSION = {
        ".pdf": {"application/pdf"},
        ".docx": {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/zip",
        },
        ".txt": {"text/plain"},
    }

    # Кодировка в .txt не записана, определяем по содержимому. Запасные
    # варианты пробуются только после UTF-8 и только при проверках ниже.
    FALLBACK_TXT_ENCODINGS = ("cp1251",)
    # Порог доли «не-UTF-8» байтов, начиная с которого файл считаем текстом в
    # однобайтовой кодировке. С запасом: у cp1251 доля около 1.0, у битого
    # UTF-8 — сотые доли, а файл-смесь из обеих кодировок даёт ~0.5 и отвергается.
    UTF8_MISMATCH_RATIO = 0.8

    @staticmethod
    def get_extension(filename: str) -> str:
        return Path(filename).suffix.lower()

    @staticmethod
    def truncate_name_bytes(name: str, max_bytes: int) -> str:
        """Обрезать имя до max_bytes в UTF-8, сохранив расширение.

        Считаем байты, а не символы: NAME_MAX измеряется в байтах, кириллица
        занимает по два, и «255 символов» упёрлись бы в тот же ENAMETOOLONG.
        errors="ignore" на обратном декодировании отбрасывает многобайтовую
        последовательность, разрубленную срезом посередине.
        """
        if len(name.encode("utf-8")) <= max_bytes:
            return name

        suffix = Path(name).suffix
        suffix_bytes = suffix.encode("utf-8")
        # Расширение сохраняем, но не ценой всего имени: если оно само длиннее
        # лимита, сохранять уже нечего и режем строку как есть.
        if len(suffix_bytes) >= max_bytes:
            return name.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")

        stem_bytes = name[: len(name) - len(suffix)].encode("utf-8")
        head = stem_bytes[: max_bytes - len(suffix_bytes)]
        return head.decode("utf-8", errors="ignore") + suffix

    @classmethod
    def sanitize_display_name(cls, filename: str | None) -> str:
        """Имя источника для БД и API: без каталогов и управляющих символов.

        На путь записи не влияет — файл всё равно ложится под собственным
        uuid-именем (см. save_upload_file), и выхода за каталог загрузок не
        было. Чистим ради того, что уходит наружу: name показывается в списке
        источников и подставляется в Content-Disposition у GET /{id}/preview.
        Строка вида `../../../../etc/passwd.txt` там не значит ничего, кроме
        путаницы, а `..` и `.` именами файлов не являются вовсе.

        Обратные слэши приводим к прямым до разбора пути: клиент на Windows
        присылает filename с ними, а PurePosixPath разделителем их не считает.
        """
        name = _CONTROL_CHARS_IN_NAME.sub("", filename or "").replace("\\", "/")
        name = PurePosixPath(name).name.strip()
        if not name or set(name) == {"."}:
            return "document"
        return cls.truncate_name_bytes(name, MAX_NAME_BYTES)

    @staticmethod
    def _normalize_media_type(content_type: str | None) -> str:
        return (content_type or "").split(";", 1)[0].strip().lower()

    @classmethod
    def validate_upload_file(cls, upload_file: UploadFile) -> str:
        if not upload_file.filename:
            raise UploadValidationError(
                "Filename is required", SourceErrors.FILENAME_REQUIRED
            )

        ext = cls.get_extension(upload_file.filename)
        if ext not in cls.ALLOWED_EXTENSIONS:
            raise UploadValidationError(
                "Unsupported file type. Allowed: PDF, DOCX, TXT",
                SourceErrors.UNSUPPORTED_TYPE,
            )

        content_type = cls._normalize_media_type(upload_file.content_type)
        allowed_types = cls.MIME_BY_EXTENSION.get(ext, set())
        if (
            content_type
            and content_type not in allowed_types
            and content_type not in cls.GENERIC_MIME_TYPES
        ):
            raise UploadValidationError(
                f"Invalid content type '{upload_file.content_type}' for extension '{ext}'",
                SourceErrors.INVALID_CONTENT_TYPE,
            )

        # Размер, заявленный при разборе формы. Отказ здесь экономит запись на
        # диск, но решающая проверка — потоковая, в save_upload_file: заявленному
        # размеру верить нельзя.
        declared_size = getattr(upload_file, "size", None)
        if declared_size is not None and declared_size > settings.MAX_UPLOAD_SIZE_BYTES:
            raise UploadValidationError(
                cls._too_large_message(), SourceErrors.TOO_LARGE
            )
        return ext

    @staticmethod
    def _too_large_message() -> str:
        return (
            f"File exceeds the {settings.MAX_UPLOAD_SIZE_MB} MB limit "
            f"for a single source"
        )

    @classmethod
    async def save_upload_file(cls, upload_file: UploadFile) -> str:
        # Имя на диске обязано уместиться в NAME_MAX вместе с uuid-префиксом.
        # Обрезаем, а не отказываем: 222 байта — предел файловой системы, а не
        # осмысленное ограничение для пользователя, и его файл действительно
        # может так называться. Уникальность держит префикс, а не имя, поэтому
        # обрезка ничего не ломает; полное имя остаётся в document.name.
        stored_name = cls.truncate_name_bytes(
            cls.sanitize_display_name(upload_file.filename),
            MAX_NAME_BYTES - STORED_NAME_PREFIX_BYTES,
        )
        file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{stored_name}")
        max_bytes = settings.MAX_UPLOAD_SIZE_BYTES
        written = 0
        try:
            try:
                with open(file_path, "wb") as buffer:
                    while True:
                        chunk = await upload_file.read(UPLOAD_CHUNK_SIZE)
                        if not chunk:
                            break
                        written += len(chunk)
                        # Считаем по мере чтения, а не по Content-Length: заголовок
                        # клиента ничем не подтверждён, а чтобы узнать размер иначе,
                        # пришлось бы принять весь файл.
                        if written > max_bytes:
                            raise UploadValidationError(
                                cls._too_large_message(), SourceErrors.TOO_LARGE
                            )
                        buffer.write(chunk)
            except OSError as exc:
                # ENAMETOOLONG после обрезки означает файловую систему со
                # своим, более строгим NAME_MAX (eCryptfs — 143 байта). Это
                # отказ из-за имени, и клиенту о нём надо сказать кодом, а не
                # пустым 500. Остальные OSError (нет места, нет прав, сбой
                # диска) — отказ сервера, и выдавать их за 400 нельзя.
                if exc.errno != errno.ENAMETOOLONG:
                    raise
                raise UploadValidationError(
                    "Имя файла слишком длинное для файловой системы сервера",
                    SourceErrors.FILENAME_TOO_LONG,
                ) from exc
        except BaseException:
            # Записанный хвост убирать больше некому: документа в БД ещё нет,
            # и путь к файлу нигде не сохранён.
            with contextlib.suppress(OSError):
                os.remove(file_path)
            raise
        return file_path

    # ------------------------------------------------------------------
    # Unified extraction: any document → TextBlock[]
    # ------------------------------------------------------------------

    @classmethod
    def extract_blocks(cls, file_path: str, extension: str) -> List[TextBlock]:
        """
        Extract TextBlocks from any supported document type.

        PDF: per-page extraction with automatic OCR fallback for sparse pages.
        DOCX: paragraphs → TextBlocks.
        TXT: paragraph-split → TextBlocks.
        """
        if extension == ".pdf":
            return cls._extract_blocks_from_pdf(file_path)
        if extension == ".docx":
            return cls._extract_blocks_from_docx(file_path)
        if extension == ".txt":
            return cls._extract_blocks_from_txt(file_path)
        raise ValueError(f"Unsupported file extension: {extension}")

    @classmethod
    def extract_and_chunk(
        cls,
        file_path: str,
        extension: str,
        chunker: HybridChunker | None = None,
    ) -> List[ChunkResult]:
        """
        Full pipeline: extract blocks → chunk.
        Convenience method for callers that want chunks directly.
        """
        blocks = cls.extract_blocks(file_path, extension)
        if not blocks:
            return []
        if chunker is None:
            from app.modules.rag.chunker_config import (
                CHUNKER_TARGET_TOKENS,
                CHUNKER_MAX_TOKENS,
                CHUNKER_MIN_TOKENS,
                CHUNKER_OVERLAP_TOKENS,
                CHUNKER_MAX_CHARS,
            )
            chunker = HybridChunker(
                target_tokens=CHUNKER_TARGET_TOKENS,
                max_tokens=CHUNKER_MAX_TOKENS,
                min_tokens=CHUNKER_MIN_TOKENS,
                overlap_tokens=CHUNKER_OVERLAP_TOKENS,
                max_chars=CHUNKER_MAX_CHARS,
            )
        return chunker.chunk(blocks)

    # ------------------------------------------------------------------
    # PDF extraction (per-page, with mixed OCR support)
    # ------------------------------------------------------------------

    @classmethod
    def _extract_blocks_from_pdf(cls, file_path: str) -> List[TextBlock]:
        """
        Extract text from PDF using PyMuPDF blocks API.
        Falls back to OCR per-page when text layer is insufficient.
        """
        blocks: List[TextBlock] = []
        order = 0

        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                # Try text layer first using blocks API for better structure
                page_blocks = page.get_text(
                    "blocks"
                )  # (x0, y0, x1, y1, text, block_no, block_type)

                # Collect text from text blocks (block_type 0 = text)
                text_content = []
                block_infos = []
                for b in page_blocks:
                    if b[6] == 0:  # text block
                        text = b[4].strip()
                        if text:
                            text_content.append(text)
                            block_infos.append(b)

                page_text = "\n".join(text_content)

                # Check if this page needs OCR
                if OCRService.page_needs_ocr(page_text):
                    # OCR fallback for this specific page
                    ocr_text = OCRService.ocr_single_page(file_path, page_num)
                    if ocr_text.strip():
                        blocks.append(
                            TextBlock(
                                text=ocr_text,
                                page=page_num,
                                order=order,
                                source="ocr",
                            )
                        )
                        order += 1
                    continue

                # Add each text block separately (preserves structure)
                for b in block_infos:
                    text = b[4].strip()
                    if text:
                        blocks.append(
                            TextBlock(
                                text=text,
                                page=page_num,
                                order=order,
                                bbox=(b[0], b[1], b[2], b[3]),
                                source="pymupdf",
                            )
                        )
                        order += 1

        return blocks

    # ------------------------------------------------------------------
    # DOCX extraction
    # ------------------------------------------------------------------

    @classmethod
    def _extract_blocks_from_docx(cls, file_path: str) -> List[TextBlock]:
        """Extract paragraphs from DOCX as TextBlocks."""
        try:
            from docx import Document as DocxDocument
        except Exception as exc:
            raise RuntimeError("DOCX support requires python-docx package") from exc

        doc = DocxDocument(file_path)
        blocks: List[TextBlock] = []

        for i, paragraph in enumerate(doc.paragraphs):
            text = paragraph.text.strip()
            if text:
                blocks.append(
                    TextBlock(
                        text=text,
                        page=1,  # DOCX doesn't have page numbers
                        order=i,
                        source="docx",
                    )
                )

        return blocks

    # ------------------------------------------------------------------
    # TXT extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _utf8_mismatch_ratio(raw_content: bytes) -> float:
        """Доля не-ASCII байтов, не сложившихся в валидные UTF-8 последовательности.

        В тексте cp1251 почти каждый кириллический байт для UTF-8 невалиден
        (замер на русских текстах — около 1.0), а у UTF-8 с испорченным или
        обрезанным фрагментом доля близка к нулю. Это и позволяет отличить
        «другая кодировка» от «сломанный UTF-8».
        """
        non_ascii = sum(1 for byte in raw_content if byte > 0x7F)
        if not non_ascii:
            return 0.0
        broken = raw_content.decode("utf-8", errors="replace").count("\ufffd")
        return broken / non_ascii

    @staticmethod
    def _looks_like_plain_text(text: str) -> bool:
        """cp1251 расшифровывает почти любые байты, поэтому успех decode() сам
        по себе ничего не подтверждает — проверяем результат на бинарный мусор."""
        return _BINARY_CONTROL_CHARS.search(text) is None

    @classmethod
    def _read_txt_content(cls, file_path: str) -> str:
        with open(file_path, "rb") as f:
            raw_content = f.read()

        # UTF-8 самопроверяем: случайный текст в другой кодировке почти никогда
        # не декодируется без ошибок, поэтому порядок «сначала UTF-8» надёжен.
        # Таджикские ӯ ҳ ҷ ғ ӣ қ в cp1251 непредставимы вовсе, так что до
        # запасных кодировок таджикский файл дойти не может.
        for encoding in ("utf-8-sig", "utf-8"):
            try:
                return raw_content.decode(encoding)
            except UnicodeDecodeError:
                continue

        # Значительная часть архивных документов на русском лежит в cp1251, и
        # отказ по ним означает «загрузить нельзя вообще». Но подменять кодировку
        # у почти валидного UTF-8 (обрезанный или склеенный из двух файл) нельзя:
        # cp1251 молча превратит его в мохибейк, поэтому такие файлы отклоняем.
        if cls._utf8_mismatch_ratio(raw_content) >= cls.UTF8_MISMATCH_RATIO:
            for encoding in cls.FALLBACK_TXT_ENCODINGS:
                try:
                    decoded = raw_content.decode(encoding)
                except UnicodeDecodeError:
                    continue
                if cls._looks_like_plain_text(decoded):
                    logger.info(
                        "TXT %s decoded as %s (not valid UTF-8)", file_path, encoding
                    )
                    return decoded

        raise UploadValidationError(
            "Файл не является валидным UTF-8. Конвертируйте файл в кодировку UTF-8 перед загрузкой.",
            SourceErrors.ENCODING_NOT_UTF8,
        )

    @classmethod
    def _extract_blocks_from_txt(cls, file_path: str) -> List[TextBlock]:
        """Read TXT file, split by double newlines into TextBlocks."""
        content = cls._read_txt_content(file_path)

        # Абзацы не склеиваем: собирать их в куски нужного размера — работа
        # HybridChunker, который всё равно заново делит блоки по пустой строке.
        blocks: List[TextBlock] = []
        for paragraph in re.split(r"\n\s*\n", content):
            text = paragraph.strip()
            if not text:
                continue
            blocks.append(
                TextBlock(
                    text=text,
                    page=1,
                    order=len(blocks),
                    source="txt",
                )
            )

        return blocks

    # ------------------------------------------------------------------
    # Helpers (kept from original)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text (kept for backward compatibility)."""
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"-\s*\n\s*", "", normalized)
        normalized = re.sub(r"\n\s*\d{1,3}\s*\n", "\n", normalized)
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    @staticmethod
    def detect_language(text: str) -> str:
        sample = (text or "").lower()
        tajik_chars = ("ӯ", "қ", "ҳ", "ҷ", "ғ", "ӣ")
        if any(char in sample for char in tajik_chars):
            return "tj"
        return "ru"
