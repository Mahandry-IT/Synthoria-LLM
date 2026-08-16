from app.services.chunker import chunk_text


def test_chunk_text_uses_target_size_and_overlap():
    text = " ".join([f"mot{i}" for i in range(1, 601)])

    chunks = chunk_text(text, target_tokens=300, overlap_tokens=50)

    assert len(chunks) > 1
    assert all(250 <= len(chunk.split()) <= 500 for chunk in chunks)
    assert chunks[0] != chunks[1]
