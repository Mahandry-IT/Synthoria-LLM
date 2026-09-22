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
- Ollama sur `http://localhost:11435`
- PostgreSQL sur `localhost:5432`
- `piper` (synthèse vocale, réseau interne uniquement) et `worker` (génération des podcasts)
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
| POST | `/courses/plan` | **Étape 1** de la génération en deux temps : retourne un plan détaillé du cours (sections `title`/`objective`/`subtopics`/`order`, sans contenu rédigé) + un `plan_id`. Le contexte de récupération (RAG / recherche web) est figé en DB avec le plan (expire après `COURSE_PLAN_TTL_MINUTES`). |
| POST | `/courses/generate/from-plan` | **Étape 2** : génère le cours complet à partir de `plan_id` + `sections` (plan validé ou édité par l'utilisateur). Génération par lots de sections DEVELOPMENT, sans plafond de sections. `404` plan inconnu, `410` plan expiré, `422` plan invalide (max 80 sections, au moins une `development`). Réponse identique à `/courses/generate`. |
| GET | `/courses/history?page=1&limit=20` | Historique paginé des sessions de cours (UUID, date, question, fichiers, mode) |
| GET | `/courses/history/{id}` | Détail d'une session avec la réponse Gemini complète |
| POST | `/courses/{session_id}/sections/{section_id}/recall` | Évalue la reformulation (« explique avec tes mots ») d'une section : body `{answer}` (≤ 1000 caractères) → `{verdict: correct\|partiel\|incorrect, feedback, missing_points}`. La section est lue en base ; la réponse est traitée comme une donnée. `404` session/section inconnue, `422` réponse vide ou trop longue, `429` (10/min). |
| GET | `/reviews/due?limit=20` | Flashcards à réviser aujourd'hui (jamais révisées ou échues), dérivées des questions « Vérifie » des cours récents |
| POST | `/reviews/{session_id}/{card_id}` | Enregistre `{result: correct\|incorrect}` et planifie la suite (Leitner J+1, J+3, J+7, J+21) → `{box, due_at}`. `404` carte inconnue |
| POST | `/podcasts/generate/{session_id}` | Met en file la génération d'un podcast à partir d'un cours persisté (`202` + `job_id`). Body optionnel `{style, target_minutes, force}`. `404` session inconnue, `422` cours sans contenu exploitable, `429` trop de demandes, `503` fonctionnalité désactivée. Idempotent : un job non échoué équivalent est renvoyé sauf `force=true`. |
| GET | `/podcasts/jobs/{job_id}` | État du job : `pending → scripting → synthesizing → mixing → done \| failed`, `stage`, `progress` (0-100), `error_message`, `duration_seconds` |
| GET | `/podcasts/{job_id}/audio` | MP3 (support `Range` pour le seek du lecteur). `409` si le job n'est pas `done`, `410` si le fichier a expiré |
| GET | `/podcasts/{job_id}/transcript` | Transcript WebVTT synchronisé (une réplique par cue) |
| GET | `/podcasts/{job_id}/script` | Script JSON du podcast (dès l'étape de scriptage) |
| GET | `/courses/history/{session_id}/podcasts` | Jobs podcast liés à une session |
| GET | `/podcasts?limit=3` | Podcasts les plus récents (tous statuts, `limit` 1-20, défaut 3) : état du job + `title` (script, sinon titre du cours, sinon question). Sert le dashboard |
| GET | `/courses/plans?page&limit` | Plans en cours : `pending`, non expirés, pas encore transformés en cours (`plan_id`, `question`, `title`, `subject`, `sections_count`, `created_at`, `expires_at`), paginés |
| GET | `/courses/plans/{plan_id}` | Plan proposé relu tel quel, avec la `question` et les `filenames` d'origine (reprise). `404` inconnu, `410` expiré |

### Tester l'API

Une collection Postman pré-configurée est disponible dans [`docs/Synthoria-LLM.postman_collection.json`](docs/Synthoria-LLM.postman_collection.json). Importez-la dans Postman (Import → fichier) pour tester tous les endpoints avec des exemples de body réalistes.

> **Mode 3 (question seule)** : ne pas fournir de `filename` → Gemini utilise la recherche web. **Mode 2 (fichier + question)** : fournir `filename` → retrieval RAG sur le document indexé.
>
> **Génération en deux temps** : `POST /courses/plan` → l'utilisateur valide ou modifie le plan → `POST /courses/generate/from-plan`. `/courses/generate` (génération directe en un appel) reste disponible. Le nombre de sections du cours final suit exactement le plan validé ; le plafond de 80 sections n'est qu'une protection anti-abus (coût Gemini), pas une limite pédagogique.
>
> **Quiz** : les questions supportent les réponses multiples (QCM). Chaque question a un niveau de difficulté (`facile`/`normale`/`difficile`) et des points calculés côté serveur pour un total de 20/20. Le frontend doit lire `correct_option_indices` (liste d'indices 0-based) au lieu de `correct_option_index` unique.

## Podcast

Un cours persisté peut être transformé en podcast audio français à deux voix (`HOST` / `EXPERT`). L'audio est produit localement (TTS CPU) ; seul le script passe par Gemini.

```text
course_sessions.gemini_response
  └─ A. Sérialisation déterministe du cours → sections sources
      └─ B. Script (Gemini, structuré, par lots) → PodcastScript JSON   [checkpoint en base]
          └─ C. Normalisation TTS (LaTeX, €, %, CIDR, sigles, code)
              └─ D. Synthèse Piper, un WAV par réplique (cache par hash)  [checkpoint disque]
                  └─ E. Assemblage ffmpeg : silences + loudnorm → MP3 + chapitres + VTT
```

- **File de jobs en base** (`podcast_jobs`, réclamation `FOR UPDATE SKIP LOCKED`) : le conteneur `worker` survit aux redémarrages, isole le CPU de l'API et peut être multiplié (`docker compose up -d --scale worker=2`).
- **Reprise** : un job en échec est remis en file (jusqu'à `PODCAST_MAX_ATTEMPTS`) et repart de son dernier checkpoint : le script n'est pas régénéré, les WAV en cache ne sont pas resynthétisés. Un job dont le worker a disparu est libéré après `PODCAST_JOB_STALE_MINUTES`.
- **Déclenchement automatique** : `generate_podcast: true` dans le body de `/courses/generate` ou `/courses/generate/from-plan` (défaut : `PODCAST_AUTO_GENERATE`). La réponse contient alors `session_id` et `podcast_job_id`. Un échec de mise en file ne fait jamais échouer la génération du cours.
- **Moteur TTS** : Piper dans un conteneur séparé, appelé en HTTP (interface `TTSEngine`, remplaçable). Le moteur est sous licence GPL-3.0 et chaque voix a sa propre licence : **vérifier le `MODEL_CARD` de chaque voix** avant tout usage. Les voix se règlent avec `PODCAST_VOICE_HOST` / `PODCAST_VOICE_EXPERT` au format `modèle[:locuteur]` (défaut : `fr_FR-upmc-medium:0` et `:1`, deux locuteurs d'un même modèle) ; le modèle téléchargé au démarrage est `PODCAST_TTS_MODEL`.

### CLI

```bash
docker compose exec api python -m app.cli podcast enqueue <session_id> [--force] [--style concise] [--minutes 8]
docker compose exec api python -m app.cli podcast run <job_id>      # synchrone, sans worker
docker compose exec api python -m app.cli podcast status <job_id>   # JSON
```

Codes de sortie : `0` succès, `1` échec du job, `2` ressource introuvable. Scriptable (cron, traitement en lot de l'historique).

### Variables

`PODCAST_ENABLED`, `PODCAST_AUTO_GENERATE`, `PODCAST_STORAGE_DIR`, `PODCAST_TTS_BASE_URL`, `PODCAST_VOICE_HOST`, `PODCAST_VOICE_EXPERT`, `PODCAST_DEFAULT_TARGET_MINUTES`, `PODCAST_MAX_MINUTES`, `PODCAST_MAX_SEGMENTS`, `PODCAST_SCRIPT_BATCH_SIZE`, `PODCAST_TTS_CONCURRENCY`, `PODCAST_TTS_MAX_CHARS`, `PODCAST_WORKER_POLL_SECONDS`, `PODCAST_JOB_STALE_MINUTES`, `PODCAST_MAX_ATTEMPTS`, `PODCAST_AUDIO_BITRATE`, `PODCAST_RETENTION_DAYS`, `PODCAST_GENERATE_RATE_LIMIT_PER_MINUTE` — valeurs par défaut dans `.env.example`. Les dossiers de jobs plus vieux que `PODCAST_RETENTION_DAYS` sont supprimés par le worker (l'audio renvoie alors `410`).

> **Sécurité** : l'API n'a pas d'authentification — toute personne connaissant l'UUID d'un job peut lire son audio. Acceptable en local, à traiter avant toute exposition. Le conteneur `piper` ne publie aucun port.

## Format du cours (pédagogie active)

- **Blocs typés** : chaque section expose `subsections[].blocks[]` (`text`, `definition`, `list`, `table`, `formula`, `code`, `worked_example`, `callout`, `pitfall`, `diagram` Mermaid, `chart`). `quoi/pourquoi/comment/tables` sont **dépréciés** (historique, podcast) et seront retirés dans une version ultérieure. Les diagrammes (≤ 4000 caractères) et graphiques (≤ 12 libellés, 4 séries) sont bornés.
- **Réponse directe** : `answer` = `summary` + `key_points` + `blocks`, distincte de l'introduction (garde-fou de similarité, `COURSE_ANSWER_INTRO_SIMILARITY_MAX`). L'ancien format `quoi/pourquoi/comment/worked_example` reste lisible.
- **Cycle par section** (développement) : `challenge` → Pourquoi → Quoi → Comment → `faded_example` (À toi) → `check_questions` (Vérifie, feedback par choix) → `recall_prompt` (reformulation). Aucun plancher de sections : la couverture décide.
- **Quiz final** : majorité de questions normale/difficile (`COURSE_QUIZ_MIN_HARD_SHARE`, 0.6), `section_refs` pour les questions mêlant plusieurs sections.
- **Pré-test** : `POST /courses/plan` renvoie `pretest` (1 question par section) ; une section envoyée à `/courses/generate/from-plan` avec `mastery: "known"` est générée en version condensée.
- **Flashcards** : `flashcards[]` dérivées des questions « Vérifie ». Migration `005_add_flashcard_reviews` (table `flashcard_reviews`) : `docker compose exec api alembic upgrade head` (la table est aussi créée au démarrage). Intervalles : `REVIEW_INTERVALS_DAYS`.
- **Podcast actif** : chaque segment se termine par une question de rappel posée par l'hôte (`think_pause`), tirée du défi ou des questions « Vérifie » de la section, suivie d'un silence de 5 s puis de la réponse de l'expert. Cette paire finale n'est jamais tronquée par le budget de mots.
- **Abus** : `/recall` (10/min), `/courses/plan/more-sections` (6/min, `MORE_SECTIONS_RATE_LIMIT_PER_MINUTE`) et `/reviews/...` ont une limite dédiée en plus de la limite globale ; `section_refs` est borné (1-500, 10 max).
- **Vidéos** : `videos[]` vient d'une vraie recherche YouTube (jamais d'ID inventé par Gemini) — voir [Vidéos YouTube](#vidéos-youtube).

## Variables d'environnement

Voir `.env.example`.

```env
OLLAMA_BASE_URL=http://ollama:11435
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
GEMINI_RPM_LIMIT=14
YOUTUBE_API_KEY=
COURSE_TOP_K_DEFAULT=6
COURSE_QUESTION_MAX_LENGTH=2000
COURSE_PLAN_BATCH_SIZE=2
COURSE_PLAN_TTL_MINUTES=120
DATABASE_URL=postgresql+asyncpg://synthoria:synthoria@postgres:5432/synthoria
```

> `DATABASE_URL` pointe vers le conteneur PostgreSQL du compose. Pour un dev local sans Docker, ajustez l'URL (ex. `postgresql+asyncpg://user:pass@localhost:5432/synthoria`).

> `GEMINI_API_KEY` est optionnel. Sans clé, l'extraction des images clés est ignorée. Les règles de sélection des images sont chargées depuis le fichier `instruction/vision_instructions.md` et Gemini retourne une réponse vide si une image n'est pas informative.

### Limite de débit Gemini (429 / RESOURCE_EXHAUSTED)

Google limite l'API sur trois axes (ex. `gemini-*-flash-lite` au palier gratuit : 15 requêtes/minute, 250k tokens/minute, 500 requêtes/jour). L'application espace ses appels (`app/services/gemini_rate_limit.py`, fenêtre glissante) pour rester sous `GEMINI_RPM_LIMIT` (14 par défaut) au lieu de heurter un 429 puis retenter — les requêtes en excès attendent leur tour (`gemini_rate_limit_throttled` dans les journaux) plutôt que d'échouer. C'est le seuil RPM qui est généralement atteint en premier : le TPM est large au regard du contexte envoyé par appel.

Cette limite est **par conteneur** : les conteneurs `api` et `worker` (podcast) ont chacun leur fenêtre, sans coordination entre eux. En usage courant ils ne se chevauchent pas assez pour dépasser le vrai quota de la clé API ; en cas d'usage intensif et simultané des deux, baissez `GEMINI_RPM_LIMIT` (ex. 7 pour partager 15 RPM en deux). Le quota journalier (RPD) n'est pas plafonné côté application : au-delà, Gemini renvoie un 429 que l'application retente puis remonte normalement.

### Vidéos YouTube

Gemini ne produit plus d'ID ni d'URL de vidéo (il en invente régulièrement) : il propose seulement `video_search_queries` (1-2 requêtes de recherche courtes), et les vidéos viennent uniquement d'une vraie recherche.

1. **YouTube Data API v3** (`app/services/youtube_data_client.py`), si `YOUTUBE_API_KEY` est configurée : `search.list` puis un seul `videos.list` groupé, filtrés (intégrable, public, pas un direct, durée entre `YOUTUBE_MIN_DURATION_SECONDS` et `YOUTUBE_MAX_DURATION_SECONDS`), mis en cache en base (table `youtube_search_cache`, TTL `YOUTUBE_CACHE_TTL_HOURS`, migration `006_add_youtube_search_cache`) et partagé entre cours proches.
2. **Repli** (pas de clé, quota épuisé, ou aucun résultat) : recherche groundée Gemini + vérification oEmbed (comportement historique).
3. Sinon, aucune vidéo n'est jointe au cours.

Un `quotaExceeded` ouvre un disjoncteur en mémoire jusqu'au reset du quota (minuit heure du Pacifique) : la Data API n'est plus appelée jusque-là, chaque cours retombe directement sur le repli. **Coût de quota** : `search.list` = 100 u, `videos.list` = 1 u ; avec 2 requêtes par cours, ~201 u, soit environ 49 cours/jour sans cache sur le quota gratuit (10 000 u/jour).

Créer la clé : [Google Cloud Console](https://console.cloud.google.com/apis/credentials) → nouveau projet (ou existant) → activer **YouTube Data API v3** → créer une clé API → la restreindre à cette seule API et, en production, à l'IP du serveur. La clé est envoyée en en-tête (`X-Goog-Api-Key`), jamais en query string ni journalisée.

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
├── repositories/     # accès aux données (course sessions, course plans)
├── services/         # Ollama, chunking, extraction PDF, vector store, Gemini Vision
│   └── podcast/      # sérialisation du cours, script, normalisation TTS, client Piper, assemblage ffmpeg, pipeline
├── workers/          # worker de jobs podcast (python -m app.workers.podcast_worker)
├── cli.py            # CLI (python -m app.cli podcast ...)
├── main.py           # bootstrap FastAPI
├── __init__.py
docs/
├── Synthoria-LLM.postman_collection.json  # collection Postman
instruction/
├── course_generation_instructions.md  # instructions LLM (quiz, cours)
├── course_plan_instructions.md  # instructions LLM (plan de cours)
├── vision_instructions.md  # instructions système Gemini
├── podcast_script_instructions.md  # instructions LLM (script de podcast)
docker/
├── piper/            # image du serveur TTS Piper
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
| `difficulty` | `"facile"` / `"normale"` / `"difficile"` | Niveau de difficulté. Répartition calculée : difficile = round(N/2), normale = round(N/4), facile = N − les deux. Réessayé puis rééquilibré automatiquement si Gemini échoue. |
| `points` | `float` | Points alloués (calculé côté serveur). Total = 20/20, borne min 0.5 |
| `time_limit_seconds` | `int` | 45s par défaut, 80s si la question implique un calcul |

> **⚠️ Breaking change** : `correct_option_index` (int) a été remplacé par `correct_option_indices` (list[int]). Mettre à jour le frontend en conséquence.
