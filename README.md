# Synthoria LLM

API FastAPI dédiée à l’ingestion et à la recherche de documents PDF via une pipeline RAG locale : extraction de texte, tableaux, images, chunking, embeddings Ollama et stockage vectoriel local.

## Stack

- **FastAPI**
- **PyMuPDF** pour l’extraction de texte PDF
- **camelot-py** pour les tableaux
- **Gemini Vision** pour les images clés (optionnel si `GEMINI_API_KEY` est fourni)
- **Ollama** pour les embeddings locaux (`nomic-embed-text`) et le modèle de génération
- **Stockage vectoriel local léger** (JSON + embeddings NumPy, sans dépendance native C++)
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
          └─ stockage vectoriel local (persist directory JSON)
```

## Démarrage rapide

```bash
cp .env.example .env
docker compose up -d --build
```

Le docker compose démarre :
- l’API FastAPI sur `http://localhost:8000`
- Ollama sur `http://localhost:11434`
- un conteneur d’initialisation qui télécharge les modèles nécessaires

Les modèles nécessaires sont pullés automatiquement dans le conteneur Ollama :
- `llama3.2`
- `nomic-embed-text`

## Endpoints

| Méthode | Route | Description |
| ------- | ----- | ----------- |
| GET | `/health` | Vérifie l’état de l’API et la disponibilité Ollama |
| POST | `/generate` | Génère une réponse à partir d’un prompt classique |
| POST | `/pdf/ingest` | Envoie un fichier PDF, extrait ses blocs, les découpe et les indexe dans le stockage local |
| POST | `/pdf/search` | Recherche sémantique dans les documents déjà indexés |
| POST | `/courses/generate` | Mode 2 : génère un cours structuré (JSON) à partir d'une question, groundé sur les documents indexés + recherche web (Gemini) |

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

### Exemple 4 : génération de cours (Mode 2 : fichier + question)

```bash
curl -X POST http://localhost:8000/courses/generate \
  -H "Content-Type: application/json" \
  -d '{"question": "Explique le principe de fonctionnement", "top_k": 6}'
```

> Nécessite `GEMINI_API_KEY`. Le pipeline enchaîne deux appels Gemini : (1) Flash + `google_search` pour une réponse groundée sur le contexte fichier et le web, (2) Flash-Lite + `response_schema` pour structurer le résultat en JSON (voir `app/api/schemas.py::CourseGenerationResponse`).

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
GEMINI_MODEL_FLASH=gemini-2.0-flash
GEMINI_MODEL_FLASH_LITE=gemini-2.0-flash-lite
GEMINI_MAX_RETRIES=3
GEMINI_TIMEOUT_SECONDS=30
# Course generation (Mode 2) configuration
COURSE_TOP_K_DEFAULT=6
COURSE_QUESTION_MAX_LENGTH=2000
```

> `GEMINI_API_KEY` est optionnel. Sans clé, l’extraction des images clés est ignorée. Les règles de sélection des images sont chargées depuis le fichier `instruction/vision_instructions.md` et Gemini retourne une réponse vide si une image n’est pas informative.

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
├── services/     # Ollama, chunking, extraction PDF, vector store, Gemini Vision
├── main.py       # bootstrap FastAPI
├── __init__.py
instruction/
├── vision_instructions.md  # instructions système Gemini pour séparer images utiles / non utiles
└── ...
```