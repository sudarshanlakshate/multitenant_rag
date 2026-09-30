from __future__ import annotations

import re
from typing import NamedTuple


class SecurityCheckResult(NamedTuple):
    is_safe: bool
    sanitized_text: str
    flagged_reasons: list[str]


# Common prompt injection signatures
PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|directives|prompts)", re.IGNORECASE),
    re.compile(r"(system\s+prompt|instructions)\s*:\s*reveal", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(an?\s+unfiltered|dan|jailbreak)", re.IGNORECASE),
    re.compile(r"disregard\s+(any|all)\s+guidelines", re.IGNORECASE),
    re.compile(r"output\s+(your\s+initial\s+instructions|the\s+system\s+message)", re.IGNORECASE),
]

# Sensitive PII Patterns
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")
PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
API_KEY_PATTERN = re.compile(r"\b(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|Bearer\s+[a-zA-Z0-9_\-\.]{20,})\b", re.IGNORECASE)


def redact_pii(text: str) -> str:
    """Redacts sensitive PII from text before persistent storage or display."""
    if not text:
        return ""
    sanitized = EMAIL_PATTERN.sub("[EMAIL REDACTED]", text)
    sanitized = SSN_PATTERN.sub("[SSN REDACTED]", sanitized)
    sanitized = CREDIT_CARD_PATTERN.sub("[CARD REDACTED]", sanitized)
    sanitized = API_KEY_PATTERN.sub("[SECRET REDACTED]", sanitized)
    sanitized = PHONE_PATTERN.sub("[PHONE REDACTED]", sanitized)
    return sanitized


def validate_input_query(query: str) -> SecurityCheckResult:
    """
    Validates user query against prompt injection and sanitizes input.
    Returns whether query is safe to process, along with sanitized query and flags.
    """
    flags: list[str] = []
    
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(query):
            flags.append("Suspected prompt injection / instruction override pattern detected.")
            break

    sanitized = redact_pii(query)
    is_safe = len(flags) == 0

    return SecurityCheckResult(
        is_safe=is_safe,
        sanitized_text=sanitized,
        flagged_reasons=flags,
    )
