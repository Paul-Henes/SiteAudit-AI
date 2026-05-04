from __future__ import annotations

import re
from urllib.parse import urlparse


MAX_BODY_CHARS = 3000


def normalize_url(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        raise ValueError("URL cannot be empty.")

    if "://" not in candidate:
        candidate = f"https://{candidate}"

    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Enter a valid http or https URL.")

    return candidate


def clean_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def clean_multiline_text(value: str) -> str:
    lines = [clean_whitespace(line) for line in value.splitlines()]
    return "\n".join(line for line in lines if line)


def truncate_text(value: str, limit: int = MAX_BODY_CHARS) -> str:
    normalized = clean_whitespace(value)
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = clean_whitespace(value)
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(normalized)
    return ordered
