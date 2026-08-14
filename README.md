# Synthoria LLM

API REST FastAPI communiquant avec un serveur Ollama, orchestrés via Docker Compose.

## Stack

- **FastAPI** + **httpx** (client async)
- **Ollama** (conteneur séparé, réseau Docker interne)
- Retry avec backoff exponentiel + timeout sur les appels Ollama
- Rate limiting par IP, CORS configurable
- Tests **pytest** (Ollama mocké)

## Démarrage

```bash
cp .env.example .env
docker compose up -d --build

# Télécharger un modèle dans le conteneur ollama
docker compose exec ollama ollama pull llama3.2
```

API disponible sur `http://localhost:8000`.

## Endpoints

| Méthode | Route       | Description                          |
| ------- | ----------- | ------------------------------------- |
| GET     | `/health`   | Statut de l'API + accessibilité Ollama |
| POST    | `/generate` | Génère une réponse à partir d'un prompt |

### Exemple

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explique Docker en une phrase"}'
```

## Développement local (sans Docker pour l'API)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

## Tests

```bash
pytest -v
```

## Architecture

```
app/
├── api/          # routes + DTOs (schemas)
├── core/         # config, exceptions, rate limiting
├── services/      # client Ollama (retry/backoff/timeout)
└── main.py        # bootstrap FastAPI
```

## Variables d'environnement

Voir `.env.example`. `OLLAMA_BASE_URL` doit pointer vers le nom du service Docker (`ollama`), pas `localhost`.
