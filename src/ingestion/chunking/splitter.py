from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Matches the page marker emitted by the PDF loader: \x00PAGE\x00<n>\x00PAGE\x00
_PAGE_MARKER_RE = re.compile(r"\x00PAGE\x00(\d+)\x00PAGE\x00")


@dataclass
class StructuredChunk:
    """A chunk with the metadata leaders expect: page, section, offsets."""

    text: str
    page: int | None
    section_path: str | None
    char_start: int
    char_end: int


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split a paragraph that is larger than chunk_size without losing content."""
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Prefer a natural sentence/punctuation boundary inside the target window.
        if end < len(text):
            boundary = max(
                text.rfind(" ", start, end),
                text.rfind(".", start, end),
                text.rfind(",", start, end),
                text.rfind("\n", start, end),
            )
            if boundary > start + int(chunk_size * 0.5):
                end = boundary + 1

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= len(text):
            break

        next_start = max(0, end - overlap)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def _extract_header(line: str) -> str | None:
    """Detect Markdown or capitalized section headers."""
    match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
    if match:
        return match.group(2).strip()
    return None


def _strip_page_markers(text: str) -> str:
    """Remove page markers from chunk text before it reaches a user or LLM."""
    return _PAGE_MARKER_RE.sub("", text).strip()


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    preserve_structure: bool = True,
) -> list[StructuredChunk]:
    """
    Enterprise Structure-Aware Chunker producing structured chunks.

    Each chunk carries:
      - page: the source page number (1-based), or None for non-PDF sources
      - section_path: the most recent section heading seen, or None
      - char_start / char_end: offsets into the original document text

    Page markers (``\\x00PAGE<n>\\x00``) emitted by the PDF loader are
    consumed here to track page boundaries and are stripped from the chunk
    text itself.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    normalized = re.sub(r"\r\n?", "\n", text or "").strip()
    if not normalized:
        return []

    raw_paragraphs = [p for p in re.split(r"\n\s*\n+", normalized) if p.strip()]

    chunks: list[StructuredChunk] = []
    current_chunk = ""
    current_section = ""
    current_page: int | None = None
    chunk_page: int | None = None
    chunk_start_offset = 0
    running_offset = 0

    def _flush() -> None:
        nonlocal current_chunk, current_section, current_page, chunk_page, chunk_start_offset
        body = _strip_page_markers(current_chunk)
        if body:
            chunks.append(
                StructuredChunk(
                    text=body,
                    page=chunk_page,
                    section_path=current_section or None,
                    char_start=chunk_start_offset,
                    char_end=chunk_start_offset + len(body),
                )
            )
        current_chunk = ""
        chunk_start_offset = running_offset

    for paragraph in raw_paragraphs:
        # Detect page markers at the start of the paragraph.
        page_match = _PAGE_MARKER_RE.match(paragraph)
        if page_match:
            current_page = int(page_match.group(1))
            # Record the page only when this is the first paragraph of a
            # new chunk. A chunk that spans multiple pages is labeled with
            # its starting page, which is the most useful citation anchor.
            if not current_chunk:
                chunk_page = current_page
            paragraph = paragraph[page_match.end():].lstrip()

        # Check if this paragraph contains a section header
        lines = paragraph.split("\n")
        first_line_header = _extract_header(lines[0])
        if first_line_header:
            current_section = first_line_header

        # Check for oversized paragraph
        if len(paragraph) > chunk_size:
            if current_chunk.strip():
                _flush()

            sub_chunks = _split_long_text(paragraph, chunk_size, overlap)
            for sc in sub_chunks:
                chunks.append(
                    StructuredChunk(
                        text=_strip_page_markers(sc),
                        page=current_page,
                        section_path=current_section or None,
                        char_start=running_offset,
                        char_end=running_offset + len(sc),
                    )
                )
            running_offset += len(paragraph) + 2
            continue

        candidate = f"{current_chunk}\n\n{paragraph}".strip() if current_chunk else paragraph

        if len(candidate) <= chunk_size:
            if not current_chunk:
                chunk_start_offset = running_offset
                chunk_page = current_page
            current_chunk = candidate
        else:
            if current_chunk.strip():
                _flush()

            # Start next chunk with overlap from previous chunk
            if overlap and chunks:
                prefix = chunks[-1].text[-overlap:].strip()
                current_chunk = f"{prefix}\n\n{paragraph}".strip()
                chunk_start_offset = running_offset
                chunk_page = current_page
                if len(current_chunk) > chunk_size:
                    current_chunk = paragraph
                    chunk_start_offset = running_offset
                    chunk_page = current_page
            else:
                current_chunk = paragraph
                chunk_start_offset = running_offset
                chunk_page = current_page

        running_offset += len(paragraph) + 2

    if current_chunk.strip():
        _flush()

    valid = [c for c in chunks if c.text.strip()]

    logger.info(
        "Structure-aware chunking completed: chunks=%d max_len=%d",
        len(valid),
        max((len(c.text) for c in valid), default=0),
    )

    return valid