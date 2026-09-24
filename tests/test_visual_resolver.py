import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app.api.schemas import ApiContentBlock, CourseGenerationResponse, CourseMeta, CourseSection, CourseSubsection
from app.core.config import Settings
from app.db.models import MediaAsset
from app.services.media import visual_resolver
from app.services.media.providers import WebImageCandidate
from app.services.media.storage import NormalizedImage, UnsupportedImageError
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
    async def boom(block, *, settings, **_kwargs):
        raise RuntimeError("panne réseau simulée")

    monkeypatch.setitem(visual_resolver._RESOLVERS, "web", boom)
    section = _section_with([_image_block(source="web")])
    result = await resolve_visuals_in_sections(
        [section], settings=Settings(), db_session_factory=_FakeSessionFactory()
    )
    assert result[0].subsections[0].blocks == []  # best-effort : jamais de propagation


@pytest.mark.asyncio
async def test_resolve_block_never_raises_on_unsupported_image(monkeypatch):
    async def unsupported(block, *, settings, **_kwargs):
        raise UnsupportedImageError("format invalide")

    monkeypatch.setitem(visual_resolver._RESOLVERS, "web", unsupported)
    section = _section_with([_image_block(source="web")])
    result = await resolve_visuals_in_sections(
        [section], settings=Settings(), db_session_factory=_FakeSessionFactory()
    )
    assert result[0].subsections[0].blocks == []


@pytest.mark.asyncio
async def test_resolve_block_respects_timeout_budget(monkeypatch):
    async def slow(block, *, settings, **_kwargs):
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


# ─── Lot 3 : résolveur web (Wikimedia Commons → Openverse) ──────────────────


def _web_settings(**overrides) -> Settings:
    return Settings(media_web_user_agent="test-agent/1.0 (test)", **overrides)


def _media_asset(**overrides) -> MediaAsset:
    defaults = dict(
        id=uuid.uuid4(), kind="web", sha256="a" * 64, mime="image/webp", width=800, height=600,
        bytes=1234, storage_path="/data/media/aa/aaaa.webp", alt_text="Un chat noir",
        origin_url="https://commons.wikimedia.org/wiki/File:Cat.jpg", author="Jane Doe",
        license="by-sa", license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        created_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return MediaAsset(**defaults)


@pytest.mark.asyncio
async def test_resolve_web_disabled_without_user_agent_returns_none():
    """Sans MEDIA_WEB_USER_AGENT (exigé par la politique Wikimedia), le résolveur reste inactif —
    comportement par défaut de Settings(), donc rétrocompatible avec les tests Lot 1."""
    section = _section_with([_image_block(source="web")])
    result = await resolve_visuals_in_sections([section], settings=Settings(), db_session_factory=_FakeSessionFactory())
    assert result[0].subsections[0].blocks == []


@pytest.mark.asyncio
async def test_resolve_web_disabled_via_flag_returns_none():
    section = _section_with([_image_block(source="web")])
    settings = _web_settings(media_web_enabled=False)
    result = await resolve_visuals_in_sections([section], settings=settings, db_session_factory=_FakeSessionFactory())
    assert result[0].subsections[0].blocks == []


@pytest.mark.asyncio
async def test_resolve_web_positive_cache_hit_reuses_asset_without_search(monkeypatch):
    asset = _media_asset()
    monkeypatch.setattr(
        visual_resolver.media_query_cache_repository, "get_fresh",
        AsyncMock(return_value=visual_resolver.media_query_cache_repository.CacheHit(found=True, asset_id=asset.id)),
    )
    monkeypatch.setattr(visual_resolver.media_asset_repository, "get", AsyncMock(return_value=asset))

    async def must_not_be_called(*args, **kwargs):
        raise AssertionError("une recherche ne doit jamais avoir lieu sur un hit de cache positif")

    monkeypatch.setattr(visual_resolver, "_search_web_candidates", must_not_be_called)

    section = _section_with([_image_block(source="web", query="chat noir")])
    result = await resolve_visuals_in_sections(
        [section], settings=_web_settings(), db_session_factory=_FakeSessionFactory()
    )

    block = result[0].subsections[0].blocks[0]
    assert block.type == "image"
    assert block.image.asset_id == str(asset.id)
    assert block.image.attribution.license == "by-sa"


@pytest.mark.asyncio
async def test_resolve_web_negative_cache_hit_drops_block_without_search(monkeypatch):
    monkeypatch.setattr(
        visual_resolver.media_query_cache_repository, "get_fresh",
        AsyncMock(return_value=visual_resolver.media_query_cache_repository.CacheHit(found=True, asset_id=None)),
    )

    async def must_not_be_called(*args, **kwargs):
        raise AssertionError("une recherche ne doit jamais avoir lieu sur un hit de cache négatif")

    monkeypatch.setattr(visual_resolver, "_search_web_candidates", must_not_be_called)

    section = _section_with([_image_block(source="web", query="sujet introuvable")])
    result = await resolve_visuals_in_sections(
        [section], settings=_web_settings(), db_session_factory=_FakeSessionFactory()
    )
    assert result[0].subsections[0].blocks == []


@pytest.mark.asyncio
async def test_search_web_candidates_stops_at_wikimedia_when_enough(monkeypatch):
    """Openverse n'est interrogé que si Wikimedia n'a pas fourni assez de candidats."""
    calls = []

    class FakeWikimedia:
        name = "wikimedia"

        async def search(self, query, *, limit):
            calls.append("wikimedia")
            return [
                WebImageCandidate(
                    provider="wikimedia", thumb_url=f"https://upload.wikimedia.org/{i}.jpg",
                    download_url=f"https://upload.wikimedia.org/{i}.jpg", page_url="https://commons.wikimedia.org/x",
                    width=800, height=600, title=f"File:{i}.jpg", author="A", license="CC0 1.0", license_url=None,
                )
                for i in range(limit)
            ]

    class FakeOpenverse:
        name = "openverse"

        async def search(self, query, *, limit):
            calls.append("openverse")
            return []

    monkeypatch.setattr(visual_resolver, "_build_web_providers", lambda settings: [FakeWikimedia(), FakeOpenverse()])

    candidates = await visual_resolver._search_web_candidates("chat noir", settings=_web_settings(media_web_max_candidates=3))

    assert calls == ["wikimedia"]  # assez de candidats dès Wikimedia : Openverse jamais appelé
    assert len(candidates) == 3


@pytest.mark.asyncio
async def test_search_web_candidates_falls_back_to_openverse_when_insufficient(monkeypatch):
    class FakeWikimedia:
        name = "wikimedia"

        async def search(self, query, *, limit):
            return []  # rien trouvé

    class FakeOpenverse:
        name = "openverse"

        async def search(self, query, *, limit):
            return [
                WebImageCandidate(
                    provider="openverse", thumb_url="https://api.openverse.org/thumb", download_url="https://api.openverse.org/thumb",
                    page_url="https://openverse.org/x", width=800, height=600, title="Cat",
                    author="B", license="by", license_url=None,
                )
            ]

    monkeypatch.setattr(visual_resolver, "_build_web_providers", lambda settings: [FakeWikimedia(), FakeOpenverse()])

    candidates = await visual_resolver._search_web_candidates("chat noir", settings=_web_settings())

    assert len(candidates) == 1 and candidates[0].provider == "openverse"


@pytest.mark.asyncio
async def test_search_web_candidates_filters_disallowed_license_and_small_width(monkeypatch):
    def make(license_, width):
        return WebImageCandidate(
            provider="wikimedia", thumb_url="https://upload.wikimedia.org/x.jpg", download_url="https://upload.wikimedia.org/x.jpg",
            page_url="https://commons.wikimedia.org/x", width=width, height=600, title="X", author="A",
            license=license_, license_url=None,
        )

    class FakeWikimedia:
        name = "wikimedia"

        async def search(self, query, *, limit):
            return [make("CC0 1.0", 800), make("CC BY-NC 4.0", 800), make("CC0 1.0", 100)]

    monkeypatch.setattr(visual_resolver, "_build_web_providers", lambda settings: [FakeWikimedia()])

    candidates = await visual_resolver._search_web_candidates("q", settings=_web_settings(media_web_min_width=400))

    assert len(candidates) == 1  # NC rejeté (licence), 100px rejeté (trop petit)


@pytest.mark.asyncio
async def test_resolve_web_full_success_path_creates_asset_and_writes_positive_cache(monkeypatch):
    image = NormalizedImage(content=b"webp-bytes", sha256="b" * 64, mime="image/webp", width=800, height=600)
    candidate = WebImageCandidate(
        provider="wikimedia", thumb_url="https://upload.wikimedia.org/x.jpg", download_url="https://upload.wikimedia.org/x.jpg",
        page_url="https://commons.wikimedia.org/wiki/File:X.jpg", width=800, height=600, title="A black cat",
        author="Jane Doe", license="CC BY-SA 4.0", license_url=None,
    )
    monkeypatch.setattr(
        visual_resolver.media_query_cache_repository, "get_fresh",
        AsyncMock(return_value=visual_resolver.media_query_cache_repository.CacheHit(found=False, asset_id=None)),
    )
    monkeypatch.setattr(visual_resolver, "_search_web_candidates", AsyncMock(return_value=[candidate]))
    monkeypatch.setattr(visual_resolver, "_download_and_normalize", AsyncMock(return_value=[(image, candidate)]))
    created_asset = _media_asset(sha256=image.sha256, author="Jane Doe", license="by-sa")
    monkeypatch.setattr(visual_resolver.media_asset_repository, "get_or_create", AsyncMock(return_value=created_asset))
    upsert_calls = []
    monkeypatch.setattr(
        visual_resolver.media_query_cache_repository, "upsert",
        AsyncMock(side_effect=lambda db, key, query, asset_id: upsert_calls.append((key, query, asset_id))),
    )
    monkeypatch.setattr(visual_resolver, "write_to_disk", lambda settings, img: None)

    section = _section_with([_image_block(source="web", query="chat noir")])
    result = await resolve_visuals_in_sections(
        [section], settings=_web_settings(media_web_verify_enabled=False), db_session_factory=_FakeSessionFactory()
    )

    block = result[0].subsections[0].blocks[0]
    assert block.image.asset_id == str(created_asset.id)
    assert block.image.attribution.author == "Jane Doe"
    assert len(upsert_calls) == 1
    assert upsert_calls[0][2] == created_asset.id  # cache écrit avec l'asset_id final (après dédup sha256)


@pytest.mark.asyncio
async def test_resolve_web_no_candidates_writes_negative_cache_and_drops_block(monkeypatch):
    monkeypatch.setattr(
        visual_resolver.media_query_cache_repository, "get_fresh",
        AsyncMock(return_value=visual_resolver.media_query_cache_repository.CacheHit(found=False, asset_id=None)),
    )
    monkeypatch.setattr(visual_resolver, "_search_web_candidates", AsyncMock(return_value=[]))
    upsert_calls = []
    monkeypatch.setattr(
        visual_resolver.media_query_cache_repository, "upsert",
        AsyncMock(side_effect=lambda db, key, query, asset_id: upsert_calls.append(asset_id)),
    )

    section = _section_with([_image_block(source="web", query="sujet vraiment introuvable")])
    result = await resolve_visuals_in_sections(
        [section], settings=_web_settings(), db_session_factory=_FakeSessionFactory()
    )

    assert result[0].subsections[0].blocks == []
    assert upsert_calls == [None]
