import logging
from pathlib import Path

import fitz

from app.core.exceptions import GeminiServiceError
from app.services.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


def _load_vision_instructions() -> str:
    candidates = [
        Path(__file__).resolve().parents[2] / "instruction" / "vision_instructions.md",
        Path(__file__).resolve().parents[1] / "instruction" / "vision_instructions.md",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return ""


async def extract_key_image_descriptions(pdf_bytes: bytes, gemini_client: GeminiClient | None) -> list[str]:
    """Retourne les descriptions des images clés d'un PDF après filtrage selon le fichier d'instruction Markdown.

    Passe par `gemini_client.describe_image` (modèle `gemini_model_flash_lite`, moins coûteux que
    `flash`) : les appels sont donc espacés par le même rate limiter que le reste de l'application
    (`app/services/gemini_rate_limit.py`) au lieu de partir en rafale non throttlée. Best-effort :
    une image dont la description échoue (quota, indisponibilité) est simplement ignorée.
    """
    if gemini_client is None or not gemini_client.is_configured:
        return []

    instructions = _load_vision_instructions()
    descriptions: list[str] = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        for page_index in range(min(len(doc), 5)):
            page = doc[page_index]
            images = page.get_images(full=True)
            if not images:
                continue

            for img_index, img_info in enumerate(images[:2]):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image.get("ext", "png")
                except Exception as exc:
                    logger.warning("image_extraction_failed page=%s img=%s error=%s", page_index + 1, img_index + 1, exc)
                    continue

                try:
                    text = await gemini_client.describe_image(image_bytes, f"image/{image_ext}", instructions)
                except GeminiServiceError as exc:
                    logger.warning("gemini_vision_describe_failed page=%s img=%s error=%s", page_index + 1, img_index + 1, exc)
                    continue

                if text and text.strip():
                    descriptions.append(f"Image page {page_index + 1} ({img_index + 1}): {text.strip()}")
                else:
                    logger.info("skip_non_informative_image page=%s img=%s reason=%s", page_index + 1, img_index + 1, "réponse vide / image non informative")
        doc.close()
    except Exception as exc:  # pragma: no cover - échec optionnel
        logger.warning("gemini_vision_failed: %s", exc)
    return descriptions
