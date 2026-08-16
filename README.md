# Synthoria LLM

API FastAPI dédiée à l’ingestion et à la recherche de documents PDF via une pipeline RAG locale : extraction de texte, tableaux, images, chunking, embeddings Ollama et stockage vectoriel Chroma.

## Stack

- **FastAPI**
- **PyMuPDF** pour l’extraction de texte PDF
- **camelot-py** pour les tableaux
- **Gemini Vision** pour les images clés (optionnel si `GEMINI_API_KEY` est fourni)
- **Ollama** pour les embeddings locaux (`nomic-embed-text`) et le modèle de génération
- **ChromaDB** pour le stockage vectoriel local
- **Rate limiting** et **CORS** configurables
- Tests **pytest**

## Architecture

```text
PDF
  ├─ texte: PyMuPDF
  ├─ tableaux: camelot
  ├─ images clés: Gemini Vision (si clé configurée)
  └─ chunking: 300-500 tokens, overlap 50
      └─ embeddings locaux: Ollama / nomic-embed-text
          └─ ChromaDB (stockage local)
```

## Démarrage rapide

```bash
cp .env.example .env
docker compose up -d --build
```

Le docker compose démarre :
- l’API FastAPI sur `http://localhost:8000`
- Ollama sur `http://localhost:11434`
- Chroma sur `http://localhost:8001`

Les modèles nécessaires sont pullés automatiquement dans le conteneur Ollama :
- `llama3.2`
- `nomic-embed-text`

## Endpoints

| Méthode | Route | Description |
| ------- | ----- | ----------- |
| GET | `/health` | Vérifie l’état de l’API et la disponibilité Ollama |
| POST | `/generate` | Génère une réponse à partir d’un prompt classique |
| POST | `/pdf/ingest` | Envoie un fichier PDF, extrait ses blocs, les découpe et les indexe dans Chroma |
| POST | `/pdf/search` | Recherche sémantique dans les documents déjà indexés |

### Exemple 1 : génération simple

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explique Docker en une phrase"}'
```

### Exemple 2 : ingestion d’un PDF

```bash
curl -X POST http://localhost:8000/pdf/ingest \
  -F "file=@document.pdf"
```

### Exemple 3 : recherche vectorielle

```bash
curl -X POST http://localhost:8000/pdf/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Quel est le point clé du document ?", "top_k": 5}'
```

## Variables d’environnement

Voir `.env.example`.

```env
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_DEFAULT_MODEL=llama3.2
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=synthoria_documents
PDF_CHUNK_TARGET_TOKENS=400
PDF_CHUNK_OVERLAP_TOKENS=50
GEMINI_API_KEY=
```

> `GEMINI_API_KEY` est optionnel. Sans clé, l’extraction des images clés est simplement ignorée.

## Développement local (sans Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

## Tests

```bash
pytest -v
```

## Structure du projet

```text
app/
├── api/          # routes + schémas FastAPI
├── core/         # configuration, exceptions, rate limiting
├── services/     # Ollama, chunking, extraction PDF, vector store
├── main.py       # bootstrap FastAPI
└── __init__.py
```
