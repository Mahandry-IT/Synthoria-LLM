import asyncio

import pytest

from app.api.schemas import ApiContentBlock, CourseGenerationResponse, CourseMeta, CourseSection, CourseSubsection
from app.core.config import Settings
from app.services.media import visual_resolver
from app.services.media.storage import UnsupportedImageError
from app.services.media.visual_resolver import resolve_visuals, resolve_visuals_in_sections


class _FakeSessionFactory:
    """Jamais utilisée réellement en Lot 1 (aucun résolveur ne produit de résultat), juste un mock utilisable."""

    def __call__(self):
        return self

    async def __aenter__(self):
        return object()

    async def __aexit__(self, *exc):
        return False


def _text_block() -> ApiContentBlock:
    return ApiContentBlock(type="text", text="Un paragraphe.")


def _image_block(source: str | None = "web", query: str | None = "chat noir") -> ApiContentBlock:
    return ApiContentBlock(type="image", image_source=source, image_query=query, image_alt="Un chat noir")


def _section_with(blocks: list[ApiContentBlock]) -> CourseSection:
    return CourseSection(
        id="s1", title="Section", quoi="", pourquoi="", comment="",
        worked_example={"statement": "", "steps": [], "result": ""},
        subsections=[CourseSubsection(title="", blocks=blocks)],
    )


def _response(sections: list[CourseSection]) -> CourseGenerationResponse:
    return CourseGenerationResponse(
        mode="question_only", format="full_course", sources=[],
        meta=CourseMeta(title="T", subject="S", generated_at="2026-09-23T10:00:00Z"),
        introduction={}, sections=sections, summary="",
    )


@pytest.mark.asyncio
async def test_resolve_visuals_in_sections_drops_image_blocks_without_session_factory():
    section = _section_with([_text_block(), _image_block()])
    result = await resolve_visuals_in_sections([section], settings=Settings(), db_session_factory=None)
    types = [b.type for b in result[0].subsections[0].blocks]
    assert types == ["text"]  # aucune DB dispo : le bloc image ne peut pas être résolu, il est retiré


@pytest.mark.asyncio
async def test_resolve_visuals_in_sections_drops_unresolved_image_lot1(monkeypatch):
    """Lot 1 : aucun résolveur par source n'est encore branché ; le bloc IMAGE est donc toujours retiré."""
    section = _section_with([_text_block(), _image_block(source="web")])
    result = await resolve_visuals_in_sections(
        [section], settings=Settings(), db_session_factory=_FakeSessionFactory()
    )
    types = [b.type for b in result[0].subsections[0].blocks]
    assert types == ["text"]


@pytest.mark.asyncio
async def test_resolve_visuals_in_sections_drops_unknown_source():
    section = _section_with([_image_block(source="not_a_real_source")])
    result = await resolve_visuals_in_sections(
        [section], settings=Settings(), db_session_factory=_FakeSessionFactory()
    )
    assert result[0].subsections[0].blocks == []


@pytest.mark.asyncio
async def test_resolve_visuals_in_sections_ignores_image_block_without_source():
    """Un bloc IMAGE sans `image_source` (jamais émis intentionnellement par Gemini) est retiré, pas planté."""
    section = _section_with([_image_block(source=None)])
    result = await resolve_visuals_in_sections(
        [section], settings=Settings(), db_session_factory=_FakeSessionFactory()
    )
    assert result[0].subsections[0].blocks == []


@pytest.mark.asyncio
async def test_resolve_block_never_raises_on_resolver_exception(monkeypatch):
    async def boom(block, *, settings):
        raise RuntimeError("panne réseau simulée")

    monkeypatch.setitem(visual_resolver._RESOLVERS, "web", boom)
    section = _section_with([_image_block(source="web")])
    result = await resolve_visuals_in_sections(
        [section], settings=Settings(), db_session_factory=_FakeSessionFactory()
    )
    assert result[0].subsections[0].blocks == []  # best-effort : jamais de propagation


@pytest.mark.asyncio
async def test_resolve_block_never_raises_on_unsupported_image(monkeypatch):
    async def unsupported(block, *, settings):
        raise UnsupportedImageError("format invalide")

    monkeypatch.setitem(visual_resolver._RESOLVERS, "web", unsupported)
    section = _section_with([_image_block(source="web")])
    result = await resolve_visuals_in_sections(
        [section], settings=Settings(), db_session_factory=_FakeSessionFactory()
    )
    assert result[0].subsections[0].blocks == []


@pytest.mark.asyncio
async def test_resolve_block_respects_timeout_budget(monkeypatch):
    async def slow(block, *, settings):
        await asyncio.sleep(10)
        return None

    monkeypatch.setitem(visual_resolver._RESOLVERS, "web", slow)
    section = _section_with([_image_block(source="web")])
    settings = Settings(media_resolve_timeout_seconds=0.05)

    result = await asyncio.wait_for(
        resolve_visuals_in_sections([section], settings=settings, db_session_factory=_FakeSessionFactory()),
        timeout=2,
    )
    assert result[0].subsections[0].blocks == []


@pytest.mark.asyncio
async def test_resolve_visuals_preserves_non_image_blocks_and_order():
    a, b, c = _text_block(), _image_block(), ApiContentBlock(type="list", list_items=["x"])
    section = _section_with([a, b, c])
    result = await resolve_visuals_in_sections(
        [section], settings=Settings(), db_session_factory=None
    )
    types = [blk.type for blk in result[0].subsections[0].blocks]
    assert types == ["text", "list"]


@pytest.mark.asyncio
async def test_resolve_visuals_skips_response_without_sections():
    response = _response([])
    result = await resolve_visuals(response, settings=Settings(), db_session_factory=_FakeSessionFactory())
    assert result.sections == []


@pytest.mark.asyncio
async def test_resolve_visuals_on_full_response_drops_image_blocks():
    section = _section_with([_text_block(), _image_block()])
    response = _response([section])
    result = await resolve_visuals(response, settings=Settings(), db_session_factory=_FakeSessionFactory())
    types = [blk.type for blk in result.sections[0].subsections[0].blocks]
    assert types == ["text"]
