from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def parse_text(file_path: str) -> str:
    """Extract UTF-8 text from a plain-text/Markdown file."""
    path = Path(file_path)
    try:
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        text = path.read_text(encoding="utf-8", errors="replace")
        return text.strip()

    except Exception:
        logger.exception("Text parsing failed: filename=%s", file_path)
        raise
