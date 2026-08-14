import re
from typing import Iterable


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, target_tokens: int = 400, overlap_tokens: int = 50) -> list[str]:
    """Découpe un texte en morceaux de 300-500 tokens environ, avec overlap de 50."""
    if not text or not text.strip():
        return []

    words = normalize_whitespace(text).split()
    if not words:
        return []

    size = max(1, target_tokens)
    overlap = max(0, min(overlap_tokens, size // 2))
    step = size - overlap

    if step <= 0:
        return [" ".join(words)]

    chunks: list[str] = []
    for start in range(0, len(words), step):
        end = min(start + size, len(words))
        chunk = " ".join(words[start:end])
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break

    cleaned: list[str] = []
    for chunk in chunks:
        if chunk.strip() and not any(existing == chunk for existing in cleaned):
            cleaned.append(chunk)
    return cleaned
