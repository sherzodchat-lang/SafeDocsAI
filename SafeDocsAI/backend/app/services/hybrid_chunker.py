"""
Hybrid Layout Chunker — unified, deterministic document chunking pipeline.

Replaces the old dual-path (agentic + semantic) chunking with a single
Blocks → Units → Chunks pipeline. No LLM calls at ingestion time.

Token estimation uses len(text) / 3.6 for Cyrillic text as a practical
approximation without requiring a tokenizer dependency.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TextBlock:
    """Raw extracted fragment from a document page."""
    text: str
    page: int
    order: int
    bbox: Optional[Tuple[float, float, float, float]] = None
    source: str = "unknown"  # "pymupdf", "ocr", "docx", "txt"


@dataclass
class Unit:
    """Classified segment after normalization and structure detection."""
    text: str
    kind: str  # heading | paragraph | list_item | table_like
    page_start: int
    page_end: int
    order: int
    section_path: List[str] = field(default_factory=list)


@dataclass
class ChunkResult:
    """Final chunk ready for storage and indexing."""
    chunk_index: int
    text: str
    page_start: int
    page_end: int
    section_path: List[str] = field(default_factory=list)

    @property
    def section_path_json(self) -> str:
        return json.dumps(self.section_path, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Regex patterns for structure detection
# ---------------------------------------------------------------------------

# Heading patterns (Russian / Tajik legal documents)
_HEADING_PATTERNS = [
    # "СТАТЬЯ 12", "Глава 3", "РАЗДЕЛ 1", "БОБИ 5", "МОДДАИ 2"
    re.compile(
        r"^(?:СТАТЬЯ|ГЛАВА|РАЗДЕЛ|БОБИ|МОДДАИ)\s+\d+",
        re.IGNORECASE,
    ),
    # Multi-level numbering: "1.2.3 Заголовок"
    re.compile(r"^\d+(?:\.\d+)+\s+\S+"),
    # Roman numerals: "IV. Заголовок"
    re.compile(r"^[IVXLCDM]+\.\s+\S+"),
    # Short ALL-CAPS line without trailing period (likely a heading)
    re.compile(r"^[A-ZА-ЯЁӮҚҲҶҒ\s\d\-]{3,80}$"),
]

# List item patterns
_LIST_PATTERN = re.compile(
    r"^(?:"
    r"[-–—•]\s+"           # bullet: - • – —
    r"|\d+[.)]\s+"         # numbered: 1) 1.
    r"|[a-zа-яёӯқҳҷғ][.)]\s+"  # lettered: а) a)
    r")",
    re.IGNORECASE,
)

# Table-like: lines with multiple pipe separators or tab-aligned columns
_TABLE_PATTERN = re.compile(r"(?:\|.*){2,}|(?:\t\S+){2,}")

# Sentence boundary (for splitting oversized units)
# Negative lookbehind protects abbreviations: ст. гл. п. др. т. н.
_SENTENCE_SPLIT = re.compile(r"(?<!ст)(?<!гл)(?<!др)(?<!\bп)(?<!\bт)(?<!\bн)(?<=[.!?։])\s+")

# Page number pattern (standalone number on its own line)
_PAGE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$")


# ---------------------------------------------------------------------------
# HybridChunker
# ---------------------------------------------------------------------------

class HybridChunker:
    """
    Token-based, structure-aware document chunker.

    Accepts a flat list of TextBlocks (from any source: PDF text layer, OCR,
    DOCX, TXT) and produces a list of ChunkResults ready for storage.
    """

    # Token estimation factor for Cyrillic text
    CHARS_PER_TOKEN = 3.6

    def __init__(
        self,
        target_tokens: int = 450,
        max_tokens: int = 800,
        min_tokens: int = 200,
        overlap_tokens: int = 0,
        max_chars: int | None = None,
    ):
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        self.max_chars = max_chars

    # -- public API ----------------------------------------------------------

    def chunk(self, blocks: List[TextBlock]) -> List[ChunkResult]:
        """Main entry point: Blocks → Units → Chunks."""
        if not blocks:
            return []

        blocks = self._normalize_blocks(blocks)
        blocks = self._remove_headers_footers(blocks)
        units = self._blocks_to_units(blocks)
        chunks = self._pack_units(units)
        chunks = self._postprocess(chunks)
        chunks = self._enforce_max_tokens(chunks)

        if self.overlap_tokens > 0:
            chunks = self._apply_overlap(chunks)

        # Assign sequential indices
        for i, c in enumerate(chunks):
            c.chunk_index = i

        return chunks

    # -- token estimation ----------------------------------------------------

    def _estimate_tokens(self, text: str) -> int:
        """Approximate token count from character length."""
        return max(1, int(len(text) / self.CHARS_PER_TOKEN))

    def _max_chunk_chars(self) -> int:
        token_based_limit = int(self.max_tokens * self.CHARS_PER_TOKEN)
        if self.max_chars is None:
            return token_based_limit
        return min(token_based_limit, self.max_chars)

    def _exceeds_limits(self, text: str) -> bool:
        if self._estimate_tokens(text) > self.max_tokens:
            return True
        return self.max_chars is not None and len(text) > self.max_chars

    # -- normalization -------------------------------------------------------

    def _normalize_blocks(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """Normalize whitespace, fix hyphenation, strip page numbers."""
        result = []
        for b in blocks:
            text = b.text
            # Normalize line endings
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            # Fix PDF hyphenation: "нало-\n гоплательщик" → "налогоплательщик"
            text = re.sub(r"-\s*\n\s*", "", text)
            # Normalize spaces
            text = re.sub(r"[ \t]+", " ", text)
            # Collapse excessive newlines
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = text.strip()
            if not text or _PAGE_NUMBER.match(text):
                continue
            result.append(TextBlock(
                text=text,
                page=b.page,
                order=b.order,
                bbox=b.bbox,
                source=b.source,
            ))
        return result

    def _remove_headers_footers(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """
        Remove repeating headers/footers across pages.

        Heuristic: first 2 and last 2 lines of each page that repeat on >60%
        of pages and are short (<100 chars) are considered headers/footers.
        """
        if not blocks:
            return blocks

        # Group blocks by page
        pages: dict[int, List[TextBlock]] = {}
        for b in blocks:
            pages.setdefault(b.page, []).append(b)

        if len(pages) < 3:
            # Not enough pages to detect repeating headers/footers
            return blocks

        # Collect candidate lines (first 2 and last 2 per page)
        candidate_counter: Counter = Counter()
        total_pages = len(pages)

        for page_blocks in pages.values():
            seen_on_page = set()
            # Sorted by order
            sorted_blocks = sorted(page_blocks, key=lambda b: b.order)

            candidates = sorted_blocks[:2] + sorted_blocks[-2:]
            for b in candidates:
                # Normalize for comparison
                normalized = re.sub(r"\s+", " ", b.text.strip().lower())
                if len(normalized) < 100 and normalized not in seen_on_page:
                    seen_on_page.add(normalized)
                    candidate_counter[normalized] += 1

        # Lines appearing on >60% of pages are headers/footers
        threshold = total_pages * 0.6
        header_footer_lines = {
            line for line, count in candidate_counter.items()
            if count >= threshold
        }

        if not header_footer_lines:
            return blocks

        # Filter out matching blocks
        result = []
        for b in blocks:
            normalized = re.sub(r"\s+", " ", b.text.strip().lower())
            if normalized not in header_footer_lines:
                result.append(b)
        return result

    # -- structure detection -------------------------------------------------

    def _classify_kind(self, text: str) -> str:
        """Classify a text block into heading/paragraph/list_item/table_like."""
        stripped = text.strip()
        lines = stripped.split("\n")

        # Table detection (multi-line with pipes or tabs)
        if len(lines) >= 2 and sum(1 for l in lines if _TABLE_PATTERN.search(l)) >= 2:
            return "table_like"

        # Single/short line checks
        first_line = lines[0].strip() if lines else ""

        # List item detection
        if _LIST_PATTERN.match(first_line):
            return "list_item"

        # Heading detection
        for pattern in _HEADING_PATTERNS[:3]:  # strong heading patterns
            if pattern.match(first_line):
                return "heading"

        # Weak heading: short uppercase line without period (require at least 2 letters)
        if (
            len(lines) <= 2
            and len(first_line) <= 80
            and first_line == first_line.upper()
            and not first_line.endswith(".")
            and len(first_line) > 3
            and sum(1 for c in first_line if c.isalpha()) >= 2
        ):
            return "heading"

        return "paragraph"

    def _blocks_to_units(self, blocks: List[TextBlock]) -> List[Unit]:
        """
        Convert TextBlocks into Units with structure classification
        and section path tracking.
        """
        units: List[Unit] = []
        section_stack: List[str] = []

        # Split each block into sub-blocks by double newlines (paragraphs)
        sub_blocks: List[TextBlock] = []
        order = 0
        for b in blocks:
            paragraphs = re.split(r"\n\s*\n", b.text)
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                sub_blocks.append(TextBlock(
                    text=para,
                    page=b.page,
                    order=order,
                    bbox=b.bbox,
                    source=b.source,
                ))
                order += 1

        for sb in sub_blocks:
            kind = self._classify_kind(sb.text)

            if kind == "heading":
                # Update section stack
                heading_text = sb.text.strip().split("\n")[0].strip()
                # Determine heading level (simple heuristic)
                level = self._heading_level(heading_text)
                # Trim stack to current level
                section_stack = section_stack[:level]
                section_stack.append(heading_text)

            units.append(Unit(
                text=sb.text,
                kind=kind,
                page_start=sb.page,
                page_end=sb.page,
                order=sb.order,
                section_path=list(section_stack),
            ))

        return units

    @staticmethod
    def _heading_level(heading_text: str) -> int:
        """
        Determine heading nesting level:
        - ГЛАВА/РАЗДЕЛ/БОБИ → level 0 (top)
        - СТАТЬЯ/МОДДАИ → level 1
        - Multi-level numbers (1.2.3) → depth of numbering
        - Everything else → level 2
        """
        h = heading_text.upper().strip()
        if re.match(r"^(?:ГЛАВА|РАЗДЕЛ|БОБИ)\s+\d+", h):
            return 0
        if re.match(r"^(?:СТАТЬЯ|МОДДАИ)\s+\d+", h):
            return 1
        m = re.match(r"^(\d+(?:\.\d+)*)\s+", h)
        if m:
            return m.group(1).count(".")
        return 2

    # -- token-based packing -------------------------------------------------

    def _pack_units(self, units: List[Unit]) -> List[ChunkResult]:
        """
        Pack Units into Chunks using token-based sizing.

        Boundary priorities:
        1. Heading boundary (always start new chunk)
        2. Paragraph boundary
        3. List boundary (don't split list in the middle if possible)
        4. Sentence boundary (last resort for oversized units)
        """
        if not units:
            return []

        chunks: List[ChunkResult] = []
        current_texts: List[str] = []
        current_tokens = 0
        current_page_start = units[0].page_start
        current_page_end = units[0].page_end
        current_section_path = units[0].section_path

        def _flush():
            nonlocal current_texts, current_tokens, current_page_start, current_page_end, current_section_path
            if current_texts:
                text = "\n\n".join(current_texts).strip()
                if text:
                    chunks.append(ChunkResult(
                        chunk_index=0,  # reassigned later
                        text=text,
                        page_start=current_page_start,
                        page_end=current_page_end,
                        section_path=list(current_section_path),
                    ))
            current_texts = []
            current_tokens = 0

        for unit in units:
            unit_tokens = self._estimate_tokens(unit.text)

            # Rule 1: Heading always starts a new chunk
            if unit.kind == "heading" and current_texts:
                _flush()
                current_page_start = unit.page_start
                current_page_end = unit.page_end
                current_section_path = unit.section_path

            # Rule 2: Would exceed max_tokens → flush first
            if current_tokens + unit_tokens > self.max_tokens and current_texts:
                if current_tokens >= self.min_tokens:
                    _flush()
                    current_page_start = unit.page_start
                    current_page_end = unit.page_end
                    current_section_path = unit.section_path
                else:
                    # Prefer a small chunk over exceeding the model's context limit.
                    _flush()
                    current_page_start = unit.page_start
                    current_page_end = unit.page_end
                    current_section_path = unit.section_path

            # Handle oversized units (bigger than max_tokens on their own)
            if unit_tokens > self.max_tokens:
                # Flush what we have
                if current_texts:
                    _flush()
                    current_page_start = unit.page_start
                    current_page_end = unit.page_end
                    current_section_path = unit.section_path

                # Split the oversized unit by sentences
                pieces = self._split_oversized(unit.text)
                for piece in pieces:
                    piece_tokens = self._estimate_tokens(piece)
                    if current_tokens + piece_tokens > self.max_tokens and current_texts:
                        _flush()
                        current_page_start = unit.page_start
                        current_page_end = unit.page_end
                        current_section_path = unit.section_path

                    current_texts.append(piece)
                    current_tokens += piece_tokens
                    current_page_end = unit.page_end
                continue

            # Check if approaching target and this is a good boundary
            if (
                current_tokens >= self.target_tokens
                and current_texts
                and unit.kind in ("heading", "paragraph")
            ):
                _flush()
                current_page_start = unit.page_start
                current_page_end = unit.page_end
                current_section_path = unit.section_path

            # Add unit to current chunk
            current_texts.append(unit.text)
            current_tokens += unit_tokens
            current_page_end = unit.page_end
            if not current_section_path:
                current_section_path = unit.section_path

        # Flush remaining
        _flush()

        return chunks

    def _split_oversized(self, text: str) -> List[str]:
        """Split oversized text by sentences, then hard-split if needed."""
        sentences = _SENTENCE_SPLIT.split(text)
        if len(sentences) <= 1 and self._exceeds_limits(text):
            # Hard split by characters
            max_chars = self._max_chunk_chars()
            return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

        result: List[str] = []
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            candidate = f"{current} {sentence}".strip() if current else sentence
            if self._exceeds_limits(candidate):
                if current:
                    result.append(current)
                if self._exceeds_limits(sentence):
                    # Hard split this sentence
                    max_chars = self._max_chunk_chars()
                    result.extend(
                        sentence[i:i + max_chars]
                        for i in range(0, len(sentence), max_chars)
                    )
                    current = ""
                else:
                    current = sentence
            else:
                current = candidate
        if current:
            result.append(current)
        return result

    # -- postprocessing ------------------------------------------------------

    def _postprocess(self, chunks: List[ChunkResult]) -> List[ChunkResult]:
        """Merge undersized trailing chunks into their predecessor.
        Never merge if the current chunk starts with a heading."""
        if len(chunks) <= 1:
            return chunks

        merged: List[ChunkResult] = [chunks[0]]

        for chunk in chunks[1:]:
            prev = merged[-1]
            prev_tokens = self._estimate_tokens(prev.text)
            curr_tokens = self._estimate_tokens(chunk.text)
            combined_tokens = self._estimate_tokens(
                f"{prev.text}\n\n{chunk.text}"
            )

            # Never merge if current chunk starts with a heading
            starts_with_heading = any(
                p.match(chunk.text.strip().split("\n")[0].strip())
                for p in _HEADING_PATTERNS[:3]
            )

            # Merge if current is undersized AND combined fits AND no heading
            if (
                curr_tokens < self.min_tokens
                and combined_tokens <= self.max_tokens
                and not starts_with_heading
            ):
                prev.text = f"{prev.text}\n\n{chunk.text}"
                prev.page_end = max(prev.page_end, chunk.page_end)
                if len(chunk.section_path) > len(prev.section_path):
                    prev.section_path = chunk.section_path
            else:
                merged.append(chunk)

        return merged

    def _enforce_max_tokens(self, chunks: List[ChunkResult]) -> List[ChunkResult]:
        """Split any chunk that still exceeds max_tokens after merging."""
        bounded: List[ChunkResult] = []

        for chunk in chunks:
            if not self._exceeds_limits(chunk.text):
                bounded.append(chunk)
                continue

            for piece in self._split_oversized(chunk.text):
                text = piece.strip()
                if not text:
                    continue
                bounded.append(
                    ChunkResult(
                        chunk_index=0,
                        text=text,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        section_path=list(chunk.section_path),
                    )
                )

        return bounded

    def _apply_overlap(self, chunks: List[ChunkResult]) -> List[ChunkResult]:
        """Prepend tail of previous chunk to each chunk for context continuity."""
        if len(chunks) <= 1 or self.overlap_tokens <= 0:
            return chunks

        overlap_chars = int(self.overlap_tokens * self.CHARS_PER_TOKEN)
        result = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_text = chunks[i - 1].text
            curr = chunks[i]

            if len(prev_text) > overlap_chars:
                tail = prev_text[-overlap_chars:]
                # Break on word boundary
                space_idx = tail.find(" ")
                if space_idx > 0:
                    tail = tail[space_idx + 1:]
                curr.text = f"...{tail}\n\n{curr.text}"

            result.append(curr)

        return result
