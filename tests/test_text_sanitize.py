from app.core.text_sanitize import sanitize_json, strip_control_chars


def test_strip_control_chars_removes_null_and_other_control_bytes_but_keeps_newlines():
    assert strip_control_chars("a\x00b\x1fc\nd") == "abc\nd"


def test_sanitize_json_recurses_through_nested_dicts_and_lists():
    value = {
        "content": "texte\x00 corrompu",
        "chunks": [{"metadata": {"label": "page\x001"}}, "brut\x00"],
        "count": 3,
        "ok": True,
        "nothing": None,
    }

    result = sanitize_json(value)

    assert result == {
        "content": "texte corrompu",
        "chunks": [{"metadata": {"label": "page1"}}, "brut"],
        "count": 3,
        "ok": True,
        "nothing": None,
    }
