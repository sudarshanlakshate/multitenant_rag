from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def parse_html(file_path: str) -> str:
    """Extract readable text from a local HTML document."""
    path = Path(file_path)
    try:
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        from bs4 import BeautifulSoup

        content = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(content, "html.parser")

        # Remove non-content elements before extraction.
        for element in soup(["script", "style", "meta", "noscript", "template"]):
            element.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)

    except Exception:
        logger.exception("HTML parsing failed: filename=%s", file_path)
        raise
