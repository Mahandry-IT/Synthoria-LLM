"""Le filet de sécurité global ne doit jamais laisser un 500 sans corps JSON exploitable ni trace."""

from fastapi import APIRouter
from fastapi.testclient import TestClient

from app.main import create_app


def test_unhandled_exception_returns_clean_json_500():
    # `create_app()` sans démarrer le lifespan (pas de Postgres/Ollama réels ici) : la route
    # ajoutée ne dépend d'aucun état d'application, seul le handler global est exercé.
    app = create_app()

    boom_router = APIRouter()

    @boom_router.get("/__boom")
    async def boom() -> None:
        raise RuntimeError("crash inattendu")

    app.include_router(boom_router)

    client = TestClient(app, raise_server_exceptions=False)
    res = client.get("/__boom", headers={"X-Request-ID": "req-123"})

    assert res.status_code == 500
    assert res.json() == {"detail": "Erreur interne inattendue", "error_code": "internal_error", "request_id": "req-123"}
    assert res.headers["X-Request-ID"] == "req-123"
