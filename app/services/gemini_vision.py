import base64
import json
import logging
from pathlib import Path

import fitz
from google.generativeai import types

logger = logging.getLogger("Synthoria LLM")

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - dépendance optionnelle
    genai = None

def _load_vision_instructions() -> str:
    candidates = [
        Path(__file__).resolve().parents[2] / "instruction" / "vision_instructions.md",
        Path(__file__).resolve().parents[1] / "instruction" / "vision_instructions.md",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return

def _parse_json_response(raw_text: str) -> dict:
    text = (raw_text or "").strip()
    if not text:
        return {"keep": False, "reason": "Réponse vide"}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {"keep": False, "reason": "Réponse invalide"}

def extract_key_image_descriptions(pdf_bytes: bytes, api_key: str | None) -> list[str]:
    """Retourne les descriptions des images clés d'un PDF après filtrage selon le fichier d'instruction Markdown."""
    if not api_key or genai is None:
        return []

    instructions = _load_vision_instructions()
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
                image_part = {"mime_type": "image/png", "data": encoded.decode("utf-8")}

                description_config = types.GenerateContentConfig(
                    system_instruction=instructions
                )
                response = model.generate_content([image_part], generation_config=description_config)
                text = getattr(response, "text", "")
                if text and text.strip():
                    descriptions.append(f"Image page {page_index + 1} ({img_index + 1}): {text.strip()}")
                else:
                    logger.info("skip_non_informative_image page=%s img=%s reason=%s", page_index + 1, img_index + 1, "réponse vide / image non informative")
        doc.close()
    except Exception as exc:  # pragma: no cover - échec optionnel
        logger.warning("gemini_vision_failed: %s", exc)
    return descriptions
