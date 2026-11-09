"""Validateur du HTML d'un widget (`ApiWidget`) : un cas par famille de motifs interdits."""

import pytest
from pydantic import ValidationError

from app.api.schemas import ApiContentBlock, ApiWidget

REALISTIC = """
<style>
  .w { background: var(--w-bg); color: var(--w-fg); font-family: inherit; position: relative; top: 0; }
  header { margin-top: 4px; border: 1px solid var(--w-border); }
</style>
<div class="w"><label for="n">Valeur</label><input id="n" type="range" min="0" max="10" value="3">
<button id="go" aria-label="Lancer">Lancer</button><output id="out" aria-live="polite"></output>
<img alt="" src="data:image/png;base64,iVBORw0KGgo=">
<svg id="s" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"></svg></div>
<script>
  // Le parent du noeud courant. Un parent. Le top. suit.
  const NS = 'http://www.w3.org/2000/svg';
  const node = document.createElementNS(NS, 'circle');
  document.getElementById('go').addEventListener('click', () => {
    document.getElementById('out').textContent = String(Number(document.getElementById('n').value) * 2);
  });
</script>
"""


def _widget(html: str, **kw) -> ApiWidget:
    return ApiWidget(title="T", description="D", html=html, **kw)


def test_realistic_widget_is_accepted_including_css_top_and_svg_namespace():
    widget = _widget(REALISTIC, height_hint=300)

    assert widget.height_hint == 300
    assert ApiContentBlock(type="widget", widget=widget).model_dump(mode="json")["widget"]["title"] == "T"


@pytest.mark.parametrize("html", [
    "<html><body>x</body></html>", "<HEAD></HEAD>", "<body>x", "<meta charset='utf-8'>", "<base href='/x'>",
    "<iframe src='about:blank'></iframe>", "<object data='x'></object>", "<embed src='x'>", "<link rel='stylesheet'>",
    "<form action='/x'></form>", "< form>",
    "<script src='app.js'></script>", "<SCRIPT type='module' SRC='a.js'></SCRIPT>",
    "<img src='http://x.test/a.png'>", "<a href='HTTPS://x.test'>x</a>", "<img src=\"//cdn.test/a.png\">",
    "<style>.a{background:url(//cdn.test/a.png)}</style>", "<style>@import 'a.css';</style>",
    "<script>fetch('/x')</script>", "<script>new XMLHttpRequest()</script>", "<script>new WebSocket('x')</script>",
    "<script>new EventSource('x')</script>", "<script>navigator.sendBeacon('x')</script>",
    "<script>import('x')</script>",
    "<script>window.parent.document</script>", "<script>window.top.location='x'</script>",
    "<script>parent.postMessage(1)</script>", "<script>top.location='x'</script>",
    "<script>document.cookie='a=b'</script>", "<script>localStorage.x=1</script>",
    "<script>sessionStorage.x=1</script>", "<script>indexedDB.open('x')</script>",
    "<script>eval('1')</script>", "<script>new Function('1')()</script>", "<a href='javascript:void(0)'>x</a>",
    "<script>postMessage(1,'*')</script>", "<div srcdoc='x'></div>",
])
def test_forbidden_patterns_are_rejected(html):
    with pytest.raises(ValidationError) as excinfo:
        _widget(html)

    errors = excinfo.value.errors()
    assert "widget html refusé" in errors[0]["msg"]
    assert html not in errors[0]["msg"]  # le HTML n'est pas recopié dans le message d'erreur


@pytest.mark.parametrize("html", [
    ".a { top: 0; margin-top: 1px } .b{ position: sticky; top:10px }",
    "<p>Le noeud parent. Puis le top. Voir ci-dessus.</p>",
    "<header>x</header><script>const retrieval = 1; const x = obj.eval_count;</script>",
    "<script>const a = 1 // commentaire\nconst b = '//' ;</script>",
])
def test_false_positives_are_accepted(html):
    assert _widget(html).html == html


def test_html_size_limit_and_field_bounds():
    with pytest.raises(ValidationError):
        _widget("x" * 20001)
    assert _widget("x" * 20000)
    for kwargs in (
        {"title": "x" * 121, "description": "d", "html": "<p>x</p>"},
        {"title": "t", "description": "d" * 501, "html": "<p>x</p>"},
        {"title": "t", "description": "d", "html": ""},
        {"title": "t", "description": "d", "html": "<p>x</p>", "height_hint": 159},
        {"title": "t", "description": "d", "html": "<p>x</p>", "height_hint": 721},
    ):
        with pytest.raises(ValidationError):
            ApiWidget(**kwargs)
