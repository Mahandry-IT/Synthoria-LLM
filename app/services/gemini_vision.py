import hashlib
import logging
from pathlib import Path

import fitz
from pydantic import ValidationError

from app.core.exceptions import GeminiServiceError
from app.schemas.course_generation import ImageDescriptionItem, ImageDescriptionsSchema
from app.services.gemini_client import GeminiClient

logger = logging.getLogger(__name__)

_MAX_PAGES = 5
_MAX_IMAGES_PER_PAGE = 2
# Écarte les icônes/puces/séparateurs décoratifs avant même l'appel Gemini — jamais assez
# informatifs pour justifier un appel, quelle que soit la classification du modèle. Basé sur les
# dimensions (pas le poids en octets : une image décorative en aplat de couleur compresse tout
# aussi bien qu'un vrai contenu informatif, ce n'est pas un signal fiable).
_MIN_IMAGE_DIMENSION_PX = 50


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

    Préfiltre local (taille minimale, dédoublonnage par sha256) puis **un seul appel multimodal
    groupé** (`gemini_client.describe_images`) pour toutes les images candidates du PDF, au lieu
    d'un appel par image (jusqu'à 10 appels auparavant) — réduit la consommation de quota
    proportionnellement au nombre d'images. Best-effort : si l'appel groupé échoue (quota,
    indisponibilité), les images de ce PDF sont simplement ignorées.
    """
    if gemini_client is None or not gemini_client.is_configured:
        return []

    instructions = _load_vision_instructions()
    # (page, img_index, bytes, mime) — ordre = index envoyé à Gemini dans le prompt.
    candidates: list[tuple[int, int, bytes, str]] = []
    seen_hashes: set[str] = set()

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        for page_index in range(min(len(doc), _MAX_PAGES)):
            page = doc[page_index]
            images = page.get_images(full=True)
            if not images:
                continue

            for img_index, img_info in enumerate(images[:_MAX_IMAGES_PER_PAGE]):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image.get("ext", "png")
                    width = base_image.get("width")
                    height = base_image.get("height")
                except Exception as exc:
                    logger.warning("image_extraction_failed page=%s img=%s error=%s", page_index + 1, img_index + 1, exc)
                    continue

                if width is None or height is None:
                    # Jamais observé en pratique (PyMuPDF renseigne ces clés pour tout format
                    # raster extractible) mais distingué du cas "vraiment petite" dans les logs :
                    # un vrai schéma dont les dimensions manqueraient serait silencieusement perdu
                    # sinon, sans indice pour le diagnostiquer.
                    logger.info(
                        "skip_image_missing_dimensions", extra={"page": page_index + 1, "img": img_index + 1}
                    )
                    continue

                if width < _MIN_IMAGE_DIMENSION_PX or height < _MIN_IMAGE_DIMENSION_PX:
                    logger.info(
                        "skip_undersized_image",
                        extra={"page": page_index + 1, "img": img_index + 1, "width": width, "height": height},
                    )
                    continue

                digest = hashlib.sha256(image_bytes).hexdigest()
                if digest in seen_hashes:
                    logger.info("skip_duplicate_image", extra={"page": page_index + 1, "img": img_index + 1})
                    continue
                seen_hashes.add(digest)

                candidates.append((page_index + 1, img_index + 1, image_bytes, f"image/{image_ext}"))
        doc.close()
    except Exception as exc:  # pragma: no cover - échec optionnel
        logger.warning("gemini_vision_failed: %s", exc)
        return []

    if not candidates:
        return []

    numbered = "\n".join(f"{i}. Image {i}" for i in range(len(candidates)))
    prompt = (
        f"Voici {len(candidates)} image(s) extraites d'un document, numérotées ci-dessous dans "
        f"l'ordre où elles te sont envoyées :\n{numbered}\n\n"
        "Analyse chaque image selon les instructions et renvoie un item par image, dans le même ordre."
    )
    image_bytes_list = [(data, mime) for _, _, data, mime in candidates]

    try:
        structured = await gemini_client.describe_images(
            image_bytes_list, prompt, system_instruction=instructions, response_schema=ImageDescriptionsSchema
        )
    except GeminiServiceError as exc:
        logger.warning("gemini_vision_describe_failed error=%s", exc)
        return []

    raw_items = structured.get("items") if isinstance(structured, dict) else None
    if not isinstance(raw_items, list):
        logger.warning("gemini_vision_response_missing_items")
        return []

    descriptions: list[str] = []
    seen_indices: set[int] = set()
    for raw_item in raw_items:
        # Chaque item est validé individuellement : un item malformé (ex. description trop longue
        # renvoyée par Gemini pour UNE image) ne doit pas faire perdre les descriptions des autres
        # images du même appel groupé — best-effort par image, comme avant le passage au batching.
        try:
            item = ImageDescriptionItem.model_validate(raw_item)
        except ValidationError as exc:
            logger.warning("gemini_vision_item_invalid error=%s", exc)
            continue
        if item.index in seen_indices or not (0 <= item.index < len(candidates)):
            continue
        seen_indices.add(item.index)
        page, img = candidates[item.index][0], candidates[item.index][1]
        if not item.informative or not item.description.strip():
            logger.info("skip_non_informative_image", extra={"page": page, "img": img})
            continue
        descriptions.append(f"Image page {page} ({img}): {item.description.strip()}")

    return descriptions
