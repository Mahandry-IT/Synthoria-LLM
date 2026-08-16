import io
import re
from typing import Any

import fitz

from app.core.config import Settings
from app.services.chunker import chunk_text
from app.services.gemini_vision import extract_key_image_descriptions

try:
    import camelot
except Exception:  # pragma: no cover - dépendance optionnelle, tolérante
    camelot = None

def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_text_from_pdf(pdf_bytes: bytes) -> list[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: list[str] = []
    for page in doc:
        text = page.get_text("text")
        cleaned = _clean_text(text)
        if cleaned:
            pages.append(cleaned)
    doc.close()
    return pages


def _extract_tables_from_pdf(pdf_bytes: bytes) -> list[str]:
    if camelot is None:
        return []

    tables: list[str] = []
    try:
        pdf_stream = io.BytesIO(pdf_bytes)
        for page_number, table in enumerate(camelot.read_pdf(pdf_stream, pages="all", flavor="stream"), start=1):
            if table.df is not None and not table.df.empty:
                text = table.df.to_markdown(index=False, tablefmt="pipe")
                if text.strip():
                    tables.append(f"Table page {page_number}:\n{text}")
    except Exception:
        return []
    return tables


def extract_pdf_chunks(pdf_bytes: bytes, filename: str, settings: Settings) -> list[dict[str, Any]]:
    text_pages = _extract_text_from_pdf(pdf_bytes)
    table_pages = _extract_tables_from_pdf(pdf_bytes)
    image_descriptions = extract_key_image_descriptions(pdf_bytes, settings.gemini_api_key)
    content_parts = text_pages + table_pages + image_descriptions

    chunks: list[dict[str, Any]] = []
    for index, part in enumerate(content_parts):
        chunk_list = chunk_text(part, target_tokens=settings.pdf_chunk_target_tokens, overlap_tokens=settings.pdf_chunk_overlap_tokens)
        for chunk_index, chunk in enumerate(chunk_list):
            chunks.append(
                {
                    "id": f"{filename}::{index}::{chunk_index}",
                    "source": filename,
                    "page": index + 1,
                    "content": chunk,
                    "metadata": {"filename": filename, "page": index + 1},
                }
            )

    return chunks
