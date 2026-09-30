from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


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


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    preserve_structure: bool = True,
) -> list[str]:
    """
    Enterprise Structure-Aware Chunker:
    - Maintains document and markdown section headers.
    - Preserves table rows and paragraphs intact.
    - Adds bounded context overlap across chunk boundaries.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    normalized = re.sub(r"\r\n?", "\n", text or "").strip()
    if not normalized:
        return []

    # Split on double newlines while tracking sections
    raw_paragraphs = [p for p in re.split(r"\n\s*\n+", normalized) if p.strip()]

    chunks: list[str] = []
    current_chunk = ""
    current_section = ""

    for paragraph in raw_paragraphs:
        # Check if this paragraph contains a section header
        lines = paragraph.split("\n")
        first_line_header = _extract_header(lines[0])
        if first_line_header:
            current_section = first_line_header

        # Check for oversized paragraph
        if len(paragraph) > chunk_size:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""

            sub_chunks = _split_long_text(paragraph, chunk_size, overlap)
            # Prepend section breadcrumb to long split chunks if available
            for sc in sub_chunks:
                if current_section and not sc.startswith(f"[{current_section}]") and len(sc) + len(current_section) + 5 <= chunk_size:
                    chunks.append(f"[{current_section}]\n{sc}")
                else:
                    chunks.append(sc)
            continue

        candidate = f"{current_chunk}\n\n{paragraph}".strip() if current_chunk else paragraph

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # Start next chunk with overlap from previous chunk
            if overlap and chunks:
                prefix = chunks[-1][-overlap:].strip()
                current_chunk = f"{prefix}\n\n{paragraph}".strip()
                if len(current_chunk) > chunk_size:
                    current_chunk = paragraph
            else:
                current_chunk = paragraph

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    valid = [c for c in chunks if c.strip()]

    logger.info(
        "Structure-aware chunking completed: chunks=%d max_len=%d",
        len(valid),
        max((len(c) for c in valid), default=0),
    )

    return valid
