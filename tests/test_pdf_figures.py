"""Stockage des figures de PDF à l'ingestion (`app.services.media.pdf_figures`)."""

import io
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from app.core.config import Settings
from app.repositories import ingested_file_figure_repository
from app.services.gemini_vision import KeyImage
from app.services.media import pdf_figures
from app.services.media.pdf_figures import FIGURE_REF_RE, figure_ref, store_pdf_figures
from app.services.media.storage import storage_path_for


def _png(color: str = "red") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (80, 60), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _image(data: bytes | None = None, description: str = "Schéma d'un graphe.") -> KeyImage:
    return KeyImage(page=2, index=1, data=data if data is not None else _png(), mime="image/png", description=description)


class _SessionFactory:
    def __call__(self):
        return self

    async def __aenter__(self):
        return object()

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(media_storage_dir=str(tmp_path))


@pytest.fixture
def repos(monkeypatch) -> SimpleNamespace:
    asset = SimpleNamespace(id=uuid.uuid4())
    get_or_create = AsyncMock(return_value=asset)
    add = AsyncMock()
    monkeypatch.setattr(pdf_figures.media_asset_repository, "get_or_create", get_or_create)
    monkeypatch.setattr(pdf_figures.ingested_file_figure_repository, "add", add)
    return SimpleNamespace(asset=asset, get_or_create=get_or_create, add=add)


def test_figure_ref_is_short_and_matches_the_reference_pattern():
    ref = figure_ref("0123456789abcdef" * 4)

    assert ref == "fig-0123456789ab"
    assert FIGURE_REF_RE.fullmatch(ref)


@pytest.mark.asyncio
async def test_stores_the_reencoded_figure_and_links_it_to_its_file(settings, repos):
    refs = await store_pdf_figures([_image()], "cours.pdf", settings=settings, session_factory=_SessionFactory())

    asset_kwargs = repos.get_or_create.await_args.kwargs
    assert refs == [figure_ref(asset_kwargs["sha256"])]
    assert asset_kwargs["kind"] == "source" and asset_kwargs["mime"] == "image/webp"
    assert (asset_kwargs["source_filename"], asset_kwargs["source_page"]) == ("cours.pdf", 2)
    assert asset_kwargs["alt_text"] == "Schéma d'un graphe."
    # Jamais l'octet brut du PDF : c'est l'image ré-encodée qui est écrite sur disque.
    stored = storage_path_for(settings, asset_kwargs["sha256"])
    assert stored.exists() and stored.read_bytes() != _png()
    assert repos.add.await_args.kwargs == {
        "filename": "cours.pdf", "ref": refs[0], "page": 2, "asset_id": repos.asset.id,
    }


@pytest.mark.asyncio
async def test_same_image_gets_the_same_reference_across_ingestions(settings, repos):
    factory = _SessionFactory()

    first = await store_pdf_figures([_image()], "a.pdf", settings=settings, session_factory=factory)
    second = await store_pdf_figures([_image()], "b.pdf", settings=settings, session_factory=factory)

    assert first == second


@pytest.mark.asyncio
async def test_unsupported_image_is_skipped_without_losing_the_others(settings, repos):
    refs = await store_pdf_figures(
        [_image(data=b"pas une image"), _image(data=_png("blue"))], "cours.pdf",
        settings=settings, session_factory=_SessionFactory(),
    )

    assert refs[0] is None and refs[1] is not None
    repos.add.assert_awaited_once()


@pytest.mark.asyncio
async def test_database_failure_never_fails_the_ingestion(settings, repos):
    repos.add.side_effect = RuntimeError("base indisponible")

    refs = await store_pdf_figures([_image()], "cours.pdf", settings=settings, session_factory=_SessionFactory())

    assert refs == [None]


@pytest.mark.asyncio
async def test_without_database_no_figure_is_stored(settings, repos):
    refs = await store_pdf_figures([_image(), _image()], "cours.pdf", settings=settings, session_factory=None)

    assert refs == [None, None]
    repos.get_or_create.assert_not_awaited()


@pytest.mark.asyncio
async def test_long_description_is_truncated_for_the_alt_text(settings, repos):
    await store_pdf_figures(
        [_image(description="x" * 1000)], "cours.pdf", settings=settings, session_factory=_SessionFactory()
    )

    assert len(repos.get_or_create.await_args.kwargs["alt_text"]) == 300


# ─── Repository ────────────────────────────────────────────────


def _session(first=None) -> MagicMock:
    session = MagicMock()
    session.commit = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.first.return_value = first
    session.execute = AsyncMock(return_value=result)
    return session


def _sql(session: MagicMock) -> str:
    return str(session.execute.await_args.args[0])


@pytest.mark.asyncio
async def test_repository_add_ignores_a_figure_already_linked_to_the_file():
    session = _session()

    await ingested_file_figure_repository.add(
        session, filename="cours.pdf", ref="fig-0123456789ab", page=1, asset_id=uuid.uuid4()
    )

    assert "ON CONFLICT" in _sql(session) and "DO NOTHING" in _sql(session)
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_repository_resolves_a_reference_to_its_asset_through_the_link_table():
    asset = object()
    session = _session(first=asset)

    assert await ingested_file_figure_repository.get_asset_by_ref(session, "fig-0123456789ab") is asset
    sql = _sql(session)
    assert "JOIN ingested_file_figures" in sql and "ingested_file_figures.ref" in sql


@pytest.mark.asyncio
async def test_repository_delete_only_unlinks_the_figures_of_that_file():
    session = _session()

    await ingested_file_figure_repository.delete_by_filename(session, "cours.pdf")

    sql = _sql(session)
    assert sql.startswith("DELETE FROM ingested_file_figures") and "filename" in sql
    session.commit.assert_awaited_once()
