from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# Internal page delimiter. It is a control character that never appears in
# normal PDF text, so it survives the chunker and is stripped before any
# text is shown to a user or an LLM.
_PAGE_SEP = "\x00PAGE\x00"


def parse_pdf(file_path: str) -> str:
    """
    Extract text from a text-based PDF, preserving page boundaries.

    The returned string embeds ``\\x00PAGE\\x00`` markers between pages so
    the chunker can recover per-page metadata (page numbers, character
    offsets). The markers are stripped before any text reaches a user or an
    LLM.

    Uses pypdf first and pdfplumber as a fallback for pages where pypdf
    returns no text. OCR is intentionally not hidden inside this function;
    scanned/image-only PDFs should be routed to an OCR-capable parser later.
    """
    path = Path(file_path)

    try:
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        from pypdf import PdfReader

        reader = PdfReader(str(path))
        total_pages = len(reader.pages)

        text_by_page: dict[int, str] = {}
        blank_pages: list[int] = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                text_by_page[page_number] = text.strip()
            else:
                blank_pages.append(page_number)

        # Fallback only for pages pypdf could not extract.
        if blank_pages:
            try:
                import pdfplumber

                with pdfplumber.open(str(path)) as pdf:
                    for page_number in blank_pages:
                        fallback = pdf.pages[page_number - 1].extract_text() or ""
                        if fallback.strip():
                            text_by_page[page_number] = fallback.strip()

            except Exception:
                logger.exception(
                    "pdfplumber fallback failed: filename=%s blank_pages=%s",
                    file_path, blank_pages,
                )

        # Preserve original page order, separated by the page marker so the
        # chunker can recover page numbers and offsets.
        page_blocks = [
            f"{_PAGE_SEP}{page_number}{_PAGE_SEP}{text_by_page[page_number]}"
            for page_number in range(1, total_pages + 1)
            if page_number in text_by_page
        ]
        full_text = "\n\n".join(page_blocks)

        if not full_text.strip():
            logger.warning(
                "No text extracted; document may be image/scanned PDF: "
                "filename=%s pages=%d",
                file_path, total_pages,
            )

        logger.info(
            "PDF parsed: filename=%s pages=%d extracted_characters=%d",
            file_path, total_pages, len(full_text),
        )

        return full_text.strip()

    except Exception:
        logger.exception("PDF parsing failed: filename=%s", file_path)
        raise