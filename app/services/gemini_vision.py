import base64
import logging
from typing import Any

import fitz

logger = logging.getLogger("Synthoria LLM")

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - dépendance optionnelle
    genai = None


def extract_key_image_descriptions(pdf_bytes: bytes, api_key: str | None) -> list[str]:
    """Retourne les descriptions des images clés d'un PDF via Gemini Vision si configuré."""
    if not api_key or genai is None:
        return []

    descriptions: list[str] = []
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        for page_index in range(min(len(doc), 5)):
            page = doc[page_index]
            images = page.get_images(full=True)
            if not images:
                continue
            for img_index, _ in enumerate(images[:2]):
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                encoded = base64.b64encode(pix.tobytes("png"))
                prompt = "Décris succinctement cette image et relève ses informations clés."
                response = model.generate_content([
                    {"mime_type": "image/png", "data": encoded.decode("utf-8")},
                    prompt,
                ])
                text = getattr(response, "text", "")
                if text.strip():
                    descriptions.append(f"Image page {page_index + 1} ({img_index + 1}): {text.strip()}")
        doc.close()
    except Exception as exc:  # pragma: no cover - échec optionnel
        logger.warning("gemini_vision_failed: %s", exc)
    return descriptions
