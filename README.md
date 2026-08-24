# Synthoria LLM

API FastAPI dédiée à l'ingestion et à la recherche de documents PDF via une pipeline RAG locale : extraction de texte, tableaux, images, chunking, embeddings Ollama et stockage vectoriel local.

## Stack

- **FastAPI**
- **PyMuPDF** pour l'extraction de texte PDF
- **camelot-py** pour les tableaux
- **Gemini Vision** pour les images clés (optionnel si `GEMINI_API_KEY` est fourni)
- **Ollama** pour les embeddings locaux (`nomic-embed-text`) et le modèle de génération
- **Stockage vectoriel local léger** (JSON + embeddings NumPy, sans dépendance native C++)
- **PostgreSQL 16** pour l'historique des sessions de cours (SQLAlchemy async + asyncpg)
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
- l'API FastAPI sur `http://localhost:8000`
- Ollama sur `http://localhost:11434`
- PostgreSQL sur `localhost:5432`
- un conteneur d'initialisation qui télécharge les modèles nécessaires

Les modèles nécessaires sont pullés automatiquement dans le conteneur Ollama :
- `llama3.2`
- `nomic-embed-text`

## Endpoints

| Méthode | Route | Description |
| ------- | ----- | ----------- |
| GET | `/health` | Vérifie l'état de l'API et la disponibilité Ollama |
| POST | `/generate` | Génère une réponse à partir d'un prompt classique |
| POST | `/pdf/ingest` | Envoie un fichier PDF, extrait ses blocs, les découpe et les indexe dans le stockage local |
| POST | `/pdf/search` | Recherche sémantique dans les documents déjà indexés |
| POST | `/courses/generate` | Génère un cours structuré (JSON). **Mode 2** (fichier + question) si `filename` fourni, **Mode 3** (question seule + recherche web) sinon. Session persistée en DB (best-effort). Quiz inclus avec réponses multiples, difficulté et points. |
| GET | `/courses/history?page=1&limit=20` | Historique paginé des sessions de cours (UUID, date, question, fichiers, mode) |
| GET | `/courses/history/{id}` | Détail d'une session avec la réponse Gemini complète |

### Tester l'API

Une collection Postman pré-configurée est disponible dans [`docs/Synthoria-LLM.postman_collection.json`](docs/Synthoria-LLM.postman_collection.json). Importez-la dans Postman (Import → fichier) pour tester tous les endpoints avec des exemples de body réalistes.

> **Mode 3 (question seule)** : ne pas fournir de `filename` → Gemini utilise la recherche web. **Mode 2 (fichier + question)** : fournir `filename` → retrieval RAG sur le document indexé.
>
> **Quiz** : les questions supportent les réponses multiples (QCM). Chaque question a un niveau de difficulté (`facile`/`normale`/`difficile`) et des points calculés côté serveur pour un total de 20/20. Le frontend doit lire `correct_option_indices` (liste d'indices 0-based) au lieu de `correct_option_index` unique.

## Variables d'environnement

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
GEMINI_MODEL_FLASH=gemini-2.5-flash
GEMINI_MODEL_FLASH_LITE=gemini-2.5-flash-lite
GEMINI_MAX_RETRIES=3
GEMINI_TIMEOUT_SECONDS=30
COURSE_TOP_K_DEFAULT=6
COURSE_QUESTION_MAX_LENGTH=2000
DATABASE_URL=postgresql+asyncpg://synthoria:synthoria@postgres:5432/synthoria
```

> `DATABASE_URL` pointe vers le conteneur PostgreSQL du compose. Pour un dev local sans Docker, ajustez l'URL (ex. `postgresql+asyncpg://user:pass@localhost:5432/synthoria`).

> `GEMINI_API_KEY` est optionnel. Sans clé, l'extraction des images clés est ignorée. Les règles de sélection des images sont chargées depuis le fichier `instruction/vision_instructions.md` et Gemini retourne une réponse vide si une image n'est pas informative.

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
├── api/              # routes + schémas FastAPI
├── core/             # configuration, exceptions, rate limiting
├── db/               # SQLAlchemy models + session async
├── repositories/     # accès aux données (course sessions)
├── services/         # Ollama, chunking, extraction PDF, vector store, Gemini Vision
├── main.py           # bootstrap FastAPI
├── __init__.py
docs/
├── Synthoria-LLM.postman_collection.json  # collection Postman
instruction/
├── course_generation_instructions.md  # instructions LLM (quiz, cours)
├── vision_instructions.md  # instructions système Gemini
migrations/
├── versions/         # migrations Alembic (PostgreSQL)
└── ...
```

## Quiz — Schéma de réponse

Chaque question de quiz dans `CourseGenerationResponse.quiz` suit ce schéma :

```json
{
  "question": "Quelle est la formule de la régression linéaire ?",
  "options": ["y = ax + b", "y = a² + b", "y = a/b", "y = a - bx"],
  "correct_option_indices": [0],
  "difficulty": "facile",
  "points": 1.0,
  "explanation": "La régression linéaire simple suit y = ax + b...",
  "time_limit_seconds": 45
}
```

| Champ | Type | Description |
| ----- | ---- | ----------- |
| `correct_option_indices` | `list[int]` | Indices 0-based des bonnes réponses. 1 élément = réponse unique, >1 = QCM multiple |
| `difficulty` | `"facile"` / `"normale"` / `"difficile"` | Niveau de difficulté. Répartition attendue : ~50% difficile, ~25% normale, ~25% facile |
| `points` | `float` | Points alloués (calculé côté serveur). Total = 20/20, borne min 0.5 |
| `time_limit_seconds` | `int` | 45s par défaut, 80s si la question implique un calcul |

> **⚠️ Breaking change** : `correct_option_index` (int) a été remplacé par `correct_option_indices` (list[int]). Mettre à jour le frontend en conséquence.
