import pytest

from app.core.config import Settings
from app.services.media.licenses import canonical_url, is_allowed, normalize_license, strip_html


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("CC0 1.0", "cc0"),
        ("cc0", "cc0"),
        ("Public domain", "pdm"),
        ("PD-old", "pdm"),
        ("CC BY 4.0", "by"),
        ("cc-by-3.0", "by"),
        ("CC BY-SA 4.0", "by-sa"),
        ("by-sa", "by-sa"),
        ("CC BY-NC 2.0", "by-nc"),
        ("CC BY-NC-SA 4.0", "by-nc-sa"),
        ("CC BY-NC-ND 2.0", "by-nc-nd"),
        ("CC BY-ND 4.0", "by-nd"),
    ],
)
def test_normalize_license_recognizes_variants(raw, expected):
    assert normalize_license(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "All rights reserved", "Copyrighted", "unknown"])
def test_normalize_license_rejects_unknown_or_empty(raw):
    assert normalize_license(raw) is None


def test_normalize_license_does_not_confuse_nc_sa_with_sa():
    """« by-nc-sa » ne doit jamais être classé « by-sa » (permissif) juste parce qu'il contient "sa"."""
    assert normalize_license("CC BY-NC-SA 4.0") == "by-nc-sa"
    assert normalize_license("CC BY-NC-SA 4.0") != "by-sa"


def test_is_allowed_rejects_nd_and_nc_by_default():
    settings = Settings()
    assert is_allowed("cc0", settings) is True
    assert is_allowed("by", settings) is True
    assert is_allowed("by-sa", settings) is True
    assert is_allowed("by-nc", settings) is False
    assert is_allowed("by-nd", settings) is False
    assert is_allowed("by-nc-sa", settings) is False
    assert is_allowed(None, settings) is False


def test_strip_html_removes_tags_and_decodes_entities():
    dirty = '<a href="https://example.org/user">Jane&nbsp;Doe</a> &amp; John'
    assert strip_html(dirty) == "Jane\xa0Doe & John"


def test_strip_html_handles_none_and_empty():
    assert strip_html(None) == ""
    assert strip_html("") == ""


def test_canonical_url_known_codes():
    assert canonical_url("by-sa").startswith("https://creativecommons.org/licenses/by-sa/")
    assert canonical_url("cc0").startswith("https://creativecommons.org/publicdomain/zero/")


def test_canonical_url_unknown_or_none():
    assert canonical_url(None) is None
    assert canonical_url("by-nc") is not None  # a une URL même si rejetée par is_allowed
