from app.services.chunker import chunk_text


def test_chunk_text_uses_target_size_and_overlap():
    text = " ".join([f"mot{i}" for i in range(1, 601)])

    chunks = chunk_text(text, target_tokens=300, overlap_tokens=50)

    assert len(chunks) > 1
    assert all(250 <= len(chunk.split()) <= 500 for chunk in chunks)
    assert chunks[0] != chunks[1]


def test_chunk_text_min_tokens_default_is_backward_compatible():
    """min_tokens=0 (défaut) : aucune fusion, comportement identique à avant."""
    text = " ".join([f"mot{i}" for i in range(1, 601)])

    with_default = chunk_text(text, target_tokens=300, overlap_tokens=50)
    without_min = chunk_text(text, target_tokens=300, overlap_tokens=50, min_tokens=0)

    assert with_default == without_min


def test_chunk_text_merges_short_chunk_at_end():
    """Un dernier chunk trop court doit être fusionné avec le précédent."""
    words = [f"mot{i}" for i in range(1, 341)]  # 340 mots
    text = " ".join(words)

    # target=300, overlap=0 -> deux morceaux de 300 puis un reste de 40 mots
    chunks = chunk_text(text, target_tokens=300, overlap_tokens=0, min_tokens=80)

    assert all(len(c.split()) >= 80 for c in chunks)
    assert sum(len(c.split()) for c in chunks) == len(words)


def test_chunk_text_merges_short_chunk_at_start():
    """Le premier chunk, s'il est trop court, est fusionné avec le suivant
    faute de voisin précédent."""
    from app.services.chunker import _merge_short_chunks

    chunks = ["a b c", "mot " * 100, "mot " * 100]
    merged = _merge_short_chunks(chunks, min_tokens=80)

    assert len(merged) == 2
    assert merged[0].startswith("a b c")


def test_chunk_text_merges_short_chunk_in_middle():
    """Un chunk court au milieu de la liste est fusionné avec son précédent."""
    from app.services.chunker import _merge_short_chunks

    chunks = ["mot " * 100, "court", "mot " * 100]
    merged = _merge_short_chunks(chunks, min_tokens=80)

    assert len(merged) == 2
    assert "court" in merged[0]


def test_merge_short_chunks_noop_when_min_tokens_is_zero():
    chunks = ["a", "b", "c"]
    assert _merge_short_chunks_import(chunks, min_tokens=0) == chunks


def _merge_short_chunks_import(chunks, min_tokens):
    from app.services.chunker import _merge_short_chunks

    return _merge_short_chunks(chunks, min_tokens)
