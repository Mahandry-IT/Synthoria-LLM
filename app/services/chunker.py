import re


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _merge_short_chunks(chunks: list[str], min_tokens: int) -> list[str]:
    """Fusionne tout chunk `< min_tokens` avec son voisin plutôt que de le
    laisser isolé (fragment à embedding faible signal, ex: une liste de
    puces courte en fin de section).

    Stratégie : fusion avec le chunk précédent ; fusion avec le suivant
    si c'est le premier chunk (pas de précédent disponible).
    """
    if min_tokens <= 0 or len(chunks) <= 1:
        return chunks

    merged: list[str] = list(chunks)
    i = 0
    while i < len(merged):
        if len(merged[i].split()) >= min_tokens or len(merged) <= 1:
            i += 1
            continue
        if i == 0:
            merged[0] = f"{merged[0]} {merged[1]}"
            del merged[1]
        else:
            merged[i - 1] = f"{merged[i - 1]} {merged[i]}"
            del merged[i]
    return merged


def chunk_text(
    text: str, target_tokens: int = 400, overlap_tokens: int = 50, min_tokens: int = 0
) -> list[str]:
    """Découpe un texte en morceaux de 300-500 tokens environ, avec overlap de 50.

    Si `min_tokens > 0`, tout chunk résultant plus court que ce seuil est
    fusionné avec son voisin (défaut 0 = rétrocompatible, pas de fusion).
    """
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
    return _merge_short_chunks(cleaned, min_tokens)