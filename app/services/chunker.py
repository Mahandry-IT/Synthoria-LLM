import re


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
    step = max(1, size - overlap)

    # Le texte entier tient dans un seul morceau.
    if len(words) <= size:
        return [" ".join(words)]

    min_size = max(1, size - overlap)
    chunks: list[str] = []
    last_start = 0
    start = 0

    while start < len(words):
        end = min(start + size, len(words))

        # Dernier morceau trop court : on l'étend dans le morceau précédent.
        if chunks and len(words) - start < min_size:
            chunks[-1] = " ".join(words[last_start:])
            break

        chunks.append(" ".join(words[start:end]))
        last_start = start

        if end == len(words):
            break

        start += step

    cleaned: list[str] = []
    for chunk in chunks:
        if chunk.strip() and not any(existing == chunk for existing in cleaned):
            cleaned.append(chunk)
    return cleaned