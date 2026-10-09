# Synthoria LLM

[![CI / Publish images](https://github.com/Mahandry-IT/Synthoria-LLM/actions/workflows/publish.yml/badge.svg?branch=master)](https://github.com/Mahandry-IT/Synthoria-LLM/actions/workflows/publish.yml)

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
docker network create synthoria-net   # une seule fois : réseau partagé avec le compose de Synthoria Studio
docker compose up -d --build
```

L'api rejoint le réseau externe `synthoria-net` pour que le compose de développement de Synthoria Studio la joigne sous le nom `api` (`http://api:8000`). Sans ce réseau, `docker compose up` échoue (« network synthoria-net declared as external, but could not be found »).

Ce compose sert au **développement local** (images construites depuis le dépôt). Il démarre :

| Service | Rôle | Exposition |
| --- | --- | --- |
| `api` | API FastAPI | `http://localhost:8000` |
| `worker` | génération des podcasts (même image que `api`) | interne |
| `ollama` / `ollama-init` | LLM + embeddings / téléchargement des modèles | `http://localhost:11435` |
| `postgres` | sessions, plans, jobs, quotas (PostgreSQL 16) | interne |
| `chroma` | base vectorielle | `http://localhost:8001` |
| `piper` / `piper-init` | synthèse vocale / téléchargement des voix | interne |

`DATABASE_URL` est construite par le compose à partir de `POSTGRES_USER`, `POSTGRES_PASSWORD` et `POSTGRES_DB` (défaut `synthoria`, à remplacer hors dev ; mot de passe sans `@ : / ? #`). Elle prime sur une `DATABASE_URL` du `.env` : pour changer d'identifiants, modifier les variables `POSTGRES_*`, jamais `DATABASE_URL` seule. Le schéma est créé au démarrage de l'API (`create_all` + colonnes ajoutées idempotentes, `app/db/schema_sync.py`) : une nouvelle image n'altère ni ne supprime les tables existantes.

Pour le **déploiement**, utiliser le dépôt parapluie [`Mahandry-IT/Synthoria`](https://github.com/Mahandry-IT/Synthoria) (LLM + Studio, images GHCR, mise à jour automatique) plutôt que ce compose.

### Images publiées

Branches : les branches de travail partent de `develop` et leurs PR ciblent `develop` ; une PR `develop` → `master` livre en production.

Le workflow `.github/workflows/publish.yml` lance `pytest` (avec un PostgreSQL de service) sur chaque push et PR vers `develop` ou `master`, puis, sur un push de `develop` ou `master` uniquement et après tests verts, publie :

| Image | Dockerfile | Utilisée par |
| --- | --- | --- |
| `ghcr.io/mahandry-it/synthoria-llm` | `Dockerfile` | `api`, `worker` |
| `ghcr.io/mahandry-it/synthoria-piper` | `docker/piper/Dockerfile` | `piper`, `piper-init` |

Tags : `latest` (dernier `master` vert, stack de production), `develop` (dernier `develop` vert, stack de test) et `sha-<7>` (commit, pour revenir en arrière). Aucun secret n'est embarqué : les clés restent dans le `.env` de l'hôte.

Les modèles nécessaires sont pullés automatiquement dans le conteneur Ollama :
- `llama3.2`
- `nomic-embed-text`

## Endpoints

| Méthode | Route | Description |
| ------- | ----- | ----------- |
| GET | `/health` | Vérifie l'état de l'API et la disponibilité Ollama |
| POST | `/generate` | Génère une réponse à partir d'un prompt classique |
| POST | `/pdf/ingest` | Envoie un ou plusieurs fichiers PDF, extrait leurs blocs, les découpe et les indexe dans le stockage local. Champs multipart optionnels `folder` / `subfolder` : les fichiers ingérés sont rangés directement dans ce dossier (créé implicitement ; `subfolder` seul = dossier par défaut). `422` nom de dossier invalide |
| POST | `/pdf/search` | Recherche sémantique dans les documents déjà indexés |
| GET | `/pdf/files?page=1&limit=20` | Fichiers indexés (paginé) : `{id, filename, folder, subfolder}`. Un fichier jamais déplacé est dans le dossier par défaut (`Général` / `Non classé`) |
| DELETE | `/pdf/files/{filename}` | Supprime un fichier, ses chunks et son rangement. `204` ; `404` fichier inconnu |
| PUT | `/pdf/files/{filename}/folder` | Range un fichier (`filename` encodé dans l'URL) : body `{folder, subfolder?}` (texte nettoyé, 1-200 caractères ; `subfolder` absent ou vide = `Non classé`) → `{filename, folder, subfolder}`. Dossier créé implicitement, casse d'un dossier existant réutilisée (« sfi » rejoint « Sfi »). Idempotent. `404` fichier inconnu, `422` validation |
| DELETE | `/pdf/folders/{folder}` | Supprime un dossier de fichiers : ses fichiers rejoignent `Général` / `Non classé` (aucun fichier supprimé) → `{moved}`. `400` dossier par défaut |
| DELETE | `/pdf/folders/{folder}/subfolders/{subfolder}` | Supprime un sous-dossier : ses fichiers rejoignent `Non classé` du même dossier → `{moved}`. `400` sous-dossier par défaut |
| POST | `/courses/generate` | Génère un cours structuré (JSON). **Mode 2** (fichier + question) si `filename` fourni, **Mode 3** (question seule + recherche web) sinon. Session persistée en DB (best-effort). Quiz inclus avec réponses multiples, difficulté et points. `depth` facultatif (`express`\|`standard`\|`approfondi`, défaut `DEFAULT_COURSE_DEPTH` = `approfondi`), renvoyé dans `meta.depth`. |
| POST | `/courses/plan` | **Étape 1** de la génération en deux temps : retourne un plan détaillé du cours (sections `title`/`objective`/`subtopics`/`order`, sans contenu rédigé) + un `plan_id`. Le contexte de récupération (RAG / recherche web) est figé en DB avec le plan (expire après `COURSE_PLAN_TTL_MINUTES`). `depth` facultatif (voir [Modes de cours](#modes-de-cours)) : persisté avec le plan (`course_plans.depth`), renvoyé dans la réponse et appliqué à toute la génération du cours. `422` mode inconnu. |
| POST | `/courses/generate/from-plan` | **Étape 2** : génère le cours complet à partir de `plan_id` + `sections` (plan validé ou édité par l'utilisateur). Génération par lots de sections DEVELOPMENT, sans plafond de sections. `404` plan inconnu, `410` plan expiré, `422` plan invalide (max 80 sections, au moins une `development`). Réponse identique à `/courses/generate` (le mode vient du plan persisté, jamais du client ; `meta.depth`). |
| GET | `/courses/history?page=1&limit=20` | Historique paginé des sessions de cours (UUID, date, question, fichiers, mode) |
| GET | `/courses/history/{id}` | Détail d'une session avec la réponse Gemini complète |
| POST | `/courses/{session_id}/sections/{section_id}/recall` | Évalue la reformulation (« explique avec tes mots ») d'une section : body `{answer}` (≤ 1000 caractères) → `{verdict: correct\|partiel\|incorrect, feedback, missing_points}`. La section est lue en base ; la réponse est traitée comme une donnée. `404` session/section inconnue, `422` réponse vide ou trop longue, `429` (10/min). |
| POST | `/courses/{session_id}/sections/{section_id}/challenge` | Analyse la réponse de l'apprenant au défi d'une section, **avant** l'explication : body `{answer}` (≤ 1000 caractères) → `{verdict: on_track\|partial\|off_track, feedback, hint}`. Le retour oriente vers l'explication sans la révéler. Le défi et ses `challenge_key_points` sont lus en base (repli sur Pourquoi/Quoi pour les anciennes sessions) ; la réponse est une donnée balisée, envoyée à Gemini mais jamais persistée. `404` session/section inconnue ou sans défi, `422` réponse vide ou trop longue, `429` (`CHALLENGE_RATE_LIMIT_PER_MINUTE`, 10/min), `502`/`503` échec Gemini. |
| GET | `/courses/{session_id}/chat` | Historique du chatbot du cours (messages non supprimés, toutes versions), ordre chronologique : `{messages: [{id, role: user\|assistant, content, status: answered\|off_topic, sources: [{label, reference}], created_at, parent_id}], quota: {limit, used, remaining, resets_at}}`. Arbre de versions via `parent_id` : question → réponse précédente (`null` = racine), réponse → sa question ; les questions de même parent sont des versions l'une de l'autre. `404` session inconnue. |
| POST | `/courses/{session_id}/chat` | Pose une question au tuteur du cours : body `{message, section_id?, parent_id?}` (message 1-1000 caractères ; `section_id` facultatif = section en cours de lecture, prioritaire ; `parent_id` = réponse après laquelle s'insère la question, `null` = racine, absent = dernière réponse active) → `{user_message, assistant_message, quota}` (messages avec `parent_id`). **Éditer** une question = renvoyer le `parent_id` de la question éditée : nouvelle version sœur, l'ancienne reste consultable ; seule la branche racine → `parent_id` est envoyée au tuteur. `404` aussi si `parent_id` n'est pas une réponse active du cours. Conversation Gemini propre au chat, seuls le plan et les sections pertinentes (classement lexical local, sans token ; `CHAT_CONTEXT_MAX_CHARS` 8000, `CHAT_CONTEXT_TOP_SECTIONS` 2) en contexte, complétés par une recherche web seulement si `GEMINI_USE_SEARCH_GROUNDING=true` (sinon le tuteur répond sans web ; en cas de 429 du grounding, nouvel essai sans web) (sources renvoyées) ; question hors thème → `status=off_topic` et refus fixe. Quota : `CHAT_DAILY_LIMIT` (15) questions par cours et par jour UTC, consommé seulement si Gemini répond. `404` session inconnue, `422` message vide ou trop long, `429` quota du jour (`detail` explicite + `Retry-After` jusqu'à minuit UTC) ou `CHAT_RATE_LIMIT_PER_MINUTE` (10/min), `502`/`503` échec Gemini. Chaque POST, édition comprise, consomme une question. |
| POST | `/courses/{session_id}/quiz/attempts` | Démarre une tentative de quiz → `201 {attempt_id, questions[]}` : N questions (N = taille du quiz d'origine) tirées de la banque du cours (amorcée au premier appel avec le quiz d'origine), les moins servies d'abord en gardant la répartition des difficultés ; format `QuizQuestion` avec points recalculés (total 20) et **sans** bonnes réponses ni explications. Quand moins de N questions n'ont jamais été servies, une tâche de fond recharge la banque (1 appel Gemini, N nouvelles questions) sans faire attendre la tentative ; en cas d'échec, les moins servies sont réutilisées. `404` session inconnue ou cours sans quiz, `429` (`QUIZ_ATTEMPT_RATE_LIMIT_PER_MINUTE`, 10/min). |
| POST | `/courses/{session_id}/quiz/attempts/{attempt_id}/submit` | Body `{answers: number[][], aborted?, abort_reason?}` (une liste d'indices par question, dans l'ordre ; `[]` = sans réponse) → `{score, max_score: 20, status, results[]}` ; `results[i]` = `{question, answer, correct_option_indices, is_correct, points, points_earned, explanation, explanation_per_choice}`. Correction côté serveur (égalité exacte des ensembles d'indices). `aborted=true` (triche) : enregistrée `aborted` avec sa raison, `score: null`, `results: []`. `404` tentative inconnue pour ce cours, `409` déjà soumise, `422` réponses invalides, `429` (`QUIZ_SUBMIT_RATE_LIMIT_PER_MINUTE`, 20/min). |
| GET | `/courses/{session_id}/quiz/attempts?limit` | Historique des tentatives (`attempt_id`, `started_at`, `finished_at`, `score`, `max_score`, `status`, `abort_reason`), les plus récentes d'abord (`limit` 1-200, défaut 50). `404` session inconnue. |
| DELETE | `/courses/{session_id}/chat/messages/{message_id}` | Supprime (logiquement, `deleted_at`) une question, sa réponse et toute leur descendance ; les autres versions restent. `200` `{status: "deleted"}` ; `404` session inconnue ou message qui n'est pas une question active du cours. Ne rend pas de quota. |
| DELETE | `/courses/{session_id}/chat` | Supprime (logiquement) tout le chat du cours ; la question suivante repart d'une racine. `200` `{status: "deleted"}` ; `404` session inconnue. Ne rend pas de quota (le quota compte les questions supprimées). |
| POST | `/courses/{session_id}/sections/{section_id}/regenerate` | Régénère le contenu d'une section marquée `incomplete` (échec temporaire à la génération) → `CourseSection` mis à jour, validé selon le mode du cours (`meta.depth`) avec la même réparation bornée que la génération. `404` session/section inconnue, `409` la section n'est pas incomplète, `429` (6/min), `502`/`503` échec Gemini. |
| PUT | `/courses/{session_id}/sections/{section_id}/note` | Enregistre (ou efface, note vide) la note libre de l'apprenant sur une section (≤ 2000 caractères, jamais générée) → `{note, updated_at}`. `404` session/section inconnue, `422` trop longue, `429` (20/min). |
| GET | `/reviews/due?limit=20` | Flashcards à réviser aujourd'hui (jamais révisées ou échues), dérivées des questions « Vérifie » des cours récents. Chaque carte porte aussi `variant_no` (0 = question d'origine), `mode` (`qcm` si `box + variant_no` est pair et que la carte a des choix, `text` sinon), `choices`, `correct_indices` et `explanation` ; `back` reste présent |
| POST | `/reviews/{session_id}/{card_id}` | Enregistre `{result: correct\|incorrect}` et planifie la suite (Leitner J+1, J+3, J+7, J+21) → `{box, due_at, variant_no}`. `correct` fait passer la carte à la variante suivante (même notion, autre formulation) ; si elle n'existe pas encore, une tâche de fond génère en 1 appel Gemini par cours 2 variantes pour chaque carte sue qui n'en a plus (`FLASHCARD_VARIANTS_MAX_CARDS_PER_CALL`, 15) ; en attendant, la carte retombe sur sa variante courante. `404` carte inconnue |
| POST | `/podcasts/generate/{session_id}` | Met en file la génération d'un podcast à partir d'un cours persisté (`202` + `job_id`). Body optionnel `{style, target_minutes, force}`. `404` session inconnue, `422` cours sans contenu exploitable, `429` trop de demandes, `503` fonctionnalité désactivée. Idempotent : un job non échoué équivalent est renvoyé sauf `force=true`. |
| GET | `/podcasts/jobs/{job_id}` | État du job : `pending → scripting → synthesizing → mixing → done \| failed`, `stage`, `progress` (0-100), `error_message`, `duration_seconds` |
| GET | `/podcasts/{job_id}/audio` | MP3 (support `Range` pour le seek du lecteur). `409` si le job n'est pas `done`, `410` si le fichier a expiré |
| GET | `/podcasts/{job_id}/transcript` | Transcript WebVTT synchronisé (une réplique par cue) |
| GET | `/podcasts/{job_id}/script` | Script JSON du podcast (dès l'étape de scriptage) |
| GET | `/courses/history/{session_id}/podcasts` | Jobs podcast liés à une session |
| GET | `/podcasts?limit=3` | Podcasts les plus récents (tous statuts, `limit` 1-20, défaut 3) : état du job + `title` (script, sinon titre du cours, sinon question). Sert le dashboard |
| GET | `/courses/plans?page&limit` | Plans en cours : `pending`, non expirés, pas encore transformés en cours (`plan_id`, `question`, `title`, `subject`, `sections_count`, `created_at`, `expires_at`), paginés |
| GET | `/courses/plans/{plan_id}` | Plan proposé relu tel quel, avec la `question`, les `filenames` et le `depth` d'origine (reprise, « Régénérer le plan » garde le mode ; plan antérieur au mode = `approfondi`). `404` inconnu, `410` expiré |
| GET | `/media/{asset_id}` | Sert une image ré-hébergée (jamais de hotlink vers la source d'origine) : téléchargée, ré-encodée en WebP et servie depuis le stockage local. `404` image inconnue, `410` fichier expiré/supprimé, `422` `asset_id` invalide (doit être un UUID). `Cache-Control: public, max-age=31536000, immutable`. |

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
- **Cycle par section** (développement) : `challenge` (+ `challenge_key_points`, 2-4 idées attendues, analysées par `POST .../challenge`) → Pourquoi → Quoi → Comment → `faded_example` (À toi) → `check_questions` (Vérifie, feedback par choix) → `recall_prompt` (reformulation). Nombre de sections, taille des sections et du quiz : selon le mode (voir [Modes de cours](#modes-de-cours)).
- **Quiz final** : majorité de questions normale/difficile (`COURSE_QUIZ_MIN_HARD_SHARE`, 0.6), `section_refs` pour les questions mêlant plusieurs sections.
- **Pré-test** : `POST /courses/plan` renvoie `pretest` (1 question par section) ; une section envoyée à `/courses/generate/from-plan` avec `mastery: "known"` est générée en version condensée.
- **Flashcards** : `flashcards[]` dérivées des questions « Vérifie ». Migration `005_add_flashcard_reviews` (table `flashcard_reviews`) : `docker compose exec api alembic upgrade head` (la table est aussi créée au démarrage). Intervalles : `REVIEW_INTERVALS_DAYS`.
- **Podcast actif** : chaque segment se termine par une question de rappel posée par l'hôte (`think_pause`), tirée du défi ou des questions « Vérifie » de la section, suivie d'un silence de 5 s puis de la réponse de l'expert. Cette paire finale n'est jamais tronquée par le budget de mots.
- **Abus** : `/recall` (10/min), `/challenge` (10/min), `/chat` (10/min par IP + 15 questions/jour par cours ; le tuteur refuse le hors-sujet, n'exécute aucune commande, ne révèle ni prompt ni configuration, et traite leçon et message comme des données balisées), `/courses/plan/more-sections` (6/min, `MORE_SECTIONS_RATE_LIMIT_PER_MINUTE`) et `/reviews/...` ont une limite dédiée en plus de la limite globale ; `section_refs` est borné (1-500, 10 max).
- **Chat : versions et suppression** : migration `017_add_course_chat_message_versions` (`course_chat_messages.parent_id` UUID, FK sur la même table `ON DELETE CASCADE`, et `deleted_at` TIMESTAMPTZ). Idempotente et rejouée au démarrage de l'API (`app/db/schema_sync.py`), avec un rattrapage des chats antérieurs : chaque session dont aucun message n'a de parent est chaînée par ordre chronologique (une seule branche). La suppression est logique ; le quota compte les questions supprimées.
- **Vidéos** : `videos[]` vient d'une vraie recherche YouTube (jamais d'ID inventé par Gemini) — voir [Vidéos YouTube](#vidéos-youtube).
- **Régénération et notes** : une section `incomplete: true` (échec temporaire à la génération) peut être régénérée seule (`POST .../regenerate`), sans relancer tout le cours ; le contexte (fichiers ou recherche web) est ré-obtenu à partir de la session, jamais renvoyé silencieusement en cas d'échec (contrairement à la génération complète). Une section qui n'est pas incomplète ne peut pas être régénérée — à la place, l'apprenant peut y laisser une note libre (`note`, ≤ 2000 caractères, table séparée `course_section_notes`, jamais générée par le modèle) via `PUT .../note`.
- **Images** : un bloc `image` peut apparaître dans `subsections[].blocks[]`, résolu et ré-hébergé (jamais de lien direct vers la source) — voir [Supports visuels](#supports-visuels).

### Modes de cours

`depth` (`POST /courses/plan`, `POST /courses/generate`) choisit le niveau de détail ; les profils vivent dans un seul module, `app/services/course_depth.py`, lu par les prompts (plan, lots, réparation, régénération, ajout de sections, quiz) et par le validateur — aucun chiffre dans `instruction/*.md`.

| | `express` | `standard` | `approfondi` (défaut) |
|---|---|---|---|
| Sections `development` | 7-12 | 13-16 | pilotées par la couverture (plancher 18) |
| Blocs non textuels / section | ≥ 50 % | ≥ 50 % | ≥ 50 % |
| Mots de prose max / section | 150 | 300 | 500 |
| Blocs max / section | 5 | 8 | 12 |
| Questions « Vérifie » | 1-2 | 2-3 | 2-3 |
| Quiz final | 5-6 | 8-10 | 10-12+ |

- **Prose** = `text`, `definition`, `callout`, éléments de `list` et `worked_example` ; tableaux, formules, code, schémas et graphiques ne comptent pas dans les mots mais comptent dans les blocs. **Textuel** = `text`, `definition`, `callout` (un bloc `image` compte comme non textuel tant qu'aucun résolveur ne garantit son affichage). Un bloc `text` reste limité à 3 phrases.
- **Validation** (`app/services/visual_validation.py`, `course_plan_generator.section_issues`) : problèmes **bloquants** (sous-section Pourquoi/Quoi/Comment vide, sous-thème du plan non traité) et **soft** (ratio, mots, blocs, `text` trop long), avec des messages chiffrés (« 412 mots de prose pour un budget de 300 ») recopiés dans le prompt de réparation. Réparation bornée à 2 appels Gemini par lot ; un remplaçant n'est retenu que si son score pondéré (bloquant × 10 + soft) est strictement meilleur. Après réparation : bloquant restant → section `incomplete` ; soft seulement → la meilleure version est gardée et `course_section_budget_exceeded` est journalisé (suivre ce taux après déploiement : chaque dépassement coûte jusqu'à 2 appels).
- **Arbitrage** : si la couverture des sous-thèmes et le budget entrent en conflit, la couverture gagne.
- **Rétro-compatibilité** : un plan ou une session sans mode (antérieurs) est traité comme `approfondi` (`meta.depth` absent, colonne `course_plans.depth` à `approfondi` par défaut). `DEFAULT_COURSE_DEPTH` ne s'applique qu'aux nouvelles requêtes sans `depth`.
- **Migration** `015_add_course_plan_depth` (`course_plans.depth VARCHAR(12) NOT NULL DEFAULT 'approfondi'`) : idempotente, et la même colonne est ajoutée au démarrage de l'API (`app/db/schema_sync.py`) car le conteneur ne joue pas les migrations Alembic (`create_all` n'ajoute jamais de colonne à une table existante).

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
COURSE_QUESTION_MAX_LENGTH=15000
COURSE_PLAN_BATCH_SIZE=2
COURSE_PLAN_TTL_MINUTES=120
DEFAULT_COURSE_DEPTH=approfondi
DATABASE_URL=postgresql+asyncpg://synthoria:synthoria@postgres:5432/synthoria
MEDIA_STORAGE_DIR=/data/media
MEDIA_MAX_BYTES=5242880
```

> Sous Docker, `DATABASE_URL` est ignorée au profit de `POSTGRES_USER/PASSWORD/DB` (voir Démarrage rapide). Pour un dev local sans Docker, ajustez l'URL (ex. `postgresql+asyncpg://user:pass@localhost:5432/synthoria`).

> `GEMINI_API_KEY` est indispensable pour générer des cours — c'est la seule variable réellement obligatoire de l'application. Elle sert aussi, en plus de la génération de cours, à l'extraction optionnelle des images clés d'un PDF : sans clé, cette extraction précise est simplement ignorée (le reste de l'ingestion PDF continue normalement). Les règles de sélection des images sont chargées depuis le fichier `instruction/vision_instructions.md` et Gemini retourne une réponse vide si une image n'est pas informative. Cette extraction (jusqu'à 5 pages, 2 images/page par PDF ingéré) préfiltre les candidats (taille minimale, dédoublonnage par hash) puis les décrit en **un seul appel groupé** via `GeminiClient.describe_images` (`gemini_model_flash_lite`, moins coûteux que `flash`) plutôt qu'un appel par image — passe par le même rate limiter que les autres appels Gemini (`GEMINI_RPM_LIMIT`) ; si l'appel échoue (quota, indisponibilité), les images de ce PDF sont simplement ignorées.
>
> **Créer la clé** : [Google AI Studio](https://aistudio.google.com/apikey) → « Create API key » → choisir un projet Google Cloud (ou en créer un) → copier la clé générée. Palier gratuit disponible (voir [Limite de débit Gemini](#limite-de-débit-gemini-429--resource_exhausted) ci-dessous pour les quotas).

### Limite de débit Gemini (429 / RESOURCE_EXHAUSTED)

Google limite l'API sur plusieurs axes **par projet et par modèle** (ex. `gemini-*-flash-lite` au palier gratuit : 15 requêtes/minute, 250k tokens/minute, 500 requêtes/jour) — partagés entre tous les appels, quel que soit le conteneur qui les émet. L'application espace ses appels (`app/services/gemini_rate_limit.py`, fenêtre glissante) pour rester sous `GEMINI_RPM_LIMIT` (14 par défaut) au lieu de heurter un 429 puis retenter.

**État partagé Postgres** (table `gemini_model_quota`, migration `013`) : les conteneurs `api` et `worker` (podcast) partagent la même vue de ce qui a réellement été consommé, au lieu d'avoir chacun leur compteur en mémoire isolé. `GEMINI_RPM_SHARE` répartit `GEMINI_RPM_LIMIT` entre les deux process (différencié par service dans `docker-compose.yml`, ex. 0.7 api / 0.3 worker) plutôt que de laisser chacun croire disposer de la totalité du quota. Un cache mémoire de quelques secondes (`GEMINI_QUOTA_CACHE_TTL_SECONDS`) évite une requête DB à chaque appel — fuite de budget bornée assumée, pas une coordination parfaite entre process.

**Disjoncteur par modèle** (`app/services/gemini_quota_manager.py`) : un 429 identifié comme quota **journalier** marque le modèle indisponible jusqu'au prochain minuit Pacifique (fuseau des quotas Google), sans consommer de tentative ni de RPM partagé supplémentaire — un 429 **minute** retente normalement avec backoff. `GEMINI_MODEL_RPD_LIMITS` (ex. `{"gemini-3.6-flash": 100}`) permet aussi de plafonner volontairement un modèle en dessous de son vrai quota Google.

**Chaînes de repli configurables** (`GEMINI_CHAIN_GENERATION`/`GEMINI_CHAIN_LIGHT`/`GEMINI_CHAIN_SEARCH`, dérivées par défaut de `GEMINI_MODEL_FLASH_LITE`/`GEMINI_MODEL_FLASH`, lite d'abord) : chaque appel essaie les modèles dans l'ordre, sautant directement ceux marqués indisponibles par le disjoncteur — jamais de tentative réseau sur un modèle déjà su épuisé. Si même le modèle le plus robuste de la chaîne est indisponible (**mode dégradé**), les appels Gemini optionnels (complétion de couverture, classement de vidéos, relance ciblée « pour aller plus loin ») sont sautés plutôt que tentés en vain — le flux principal de génération continue.

**Réduire le volume d'appels** : cache Postgres des réponses (`gemini_response_cache`, `GEMINI_RESPONSE_CACHE_TTL_HOURS`) sur `reformulate_query`/`describe_images`/`rank_images` — réingérer un contenu déjà vu (même PDF, même question) évite un nouvel appel Gemini.

**Observer l'état** : `GET /health` expose l'état par modèle des chaînes configurées (`available`, `requests_today`, `exhausted_until`), absent (`gemini: null`) si la DB de quota est indisponible — `/health` ne dépend jamais de cette optimisation. Un 429 renvoyé par l'API porte un en-tête `Retry-After` (secondes jusqu'au reset si un quota jour a été identifié, une valeur courte par défaut sinon).

### Vidéos YouTube

Gemini ne produit plus d'ID ni d'URL de vidéo (il en invente régulièrement) : il propose seulement `video_search_queries` (1-2 requêtes de recherche courtes), et les vidéos viennent uniquement d'une vraie recherche.

1. **YouTube Data API v3** (`app/services/youtube_data_client.py`), si `YOUTUBE_API_KEY` est configurée : `search.list` puis un seul `videos.list` groupé, filtrés (intégrable, public, pas un direct, durée entre `YOUTUBE_MIN_DURATION_SECONDS` et `YOUTUBE_MAX_DURATION_SECONDS`), mis en cache en base (table `youtube_search_cache`, TTL `YOUTUBE_CACHE_TTL_HOURS`, migration `006_add_youtube_search_cache`) et partagé entre cours proches.
2. **Repli** (pas de clé, quota épuisé, ou aucun résultat) : recherche groundée Gemini + vérification oEmbed (comportement historique).
3. Sinon, aucune vidéo n'est jointe au cours.

Un `quotaExceeded` ouvre un disjoncteur en mémoire jusqu'au reset du quota (minuit heure du Pacifique) : la Data API n'est plus appelée jusque-là, chaque cours retombe directement sur le repli. **Coût de quota** : `search.list` = 100 u, `videos.list` = 1 u ; avec 2 requêtes par cours, ~201 u, soit environ 49 cours/jour sans cache sur le quota gratuit (10 000 u/jour).

Créer la clé : [Google Cloud Console](https://console.cloud.google.com/apis/credentials) → nouveau projet (ou existant) → activer **YouTube Data API v3** → créer une clé API → la restreindre à cette seule API et, en production, à l'IP du serveur. La clé est envoyée en en-tête (`X-Goog-Api-Key`), jamais en query string ni journalisée.

**Nombre de vidéos** : `COURSE_VIDEOS_MIN` (5) et `COURSE_VIDEOS_MAX` (10) donnent une **cible**, jamais une garantie — YouTube peut renvoyer moins de résultats pertinents. `COURSE_VIDEOS_MIN` sert aussi au repli groundé (« Trouve entre 5 et 10 vidéos… ») ; `COURSE_VIDEOS_MAX` borne l'affichage final dans tous les cas.

**Classement pédagogique (optionnel)** : si `COURSE_VIDEOS_RANKING_ENABLED=true` (défaut) et qu'une clé Data API a trouvé ≥ 2 candidats, un appel Flash-Lite supplémentaire (`rank_videos`) catégorise chaque vidéo (`category` : cours / exercices_corriges / intuition / demonstration / methode, `level` : debutant / intermediaire / avance, `relevance_score` 0-100) à partir d'un vivier plus large que `COURSE_VIDEOS_MAX` (+5 candidats). Gemini ne reçoit et ne renvoie que des index numérotés et des enums — jamais d'URL ni de texte libre non borné — donc un titre ou une description de vidéo malveillante ne peut pas injecter d'instruction. Les candidats sous `YOUTUBE_RANKING_MIN_SCORE` (40 par défaut) sont écartés ; la sélection finale privilégie la diversité (meilleur score de chaque catégorie d'abord). Best-effort : en cas d'échec, l'ordre V1 est conservé. Ce classement ajoute 1 appel Gemini par cours (dans la fenêtre de `GEMINI_RPM_LIMIT`) ; désactiver `COURSE_VIDEOS_RANKING_ENABLED` si le quota est trop juste.

### Supports visuels

Comme pour les vidéos, Gemini ne produit jamais d'URL ni de nom de fichier d'image (il en invente régulièrement) : un bloc `image` porte seulement une **intention** — `image_source` (`pdf` / `web` / `generated`), `image_query` (recherche web) ou `image_reference` (figure d'un PDF source), `image_alt` — résolue déterministiquement côté serveur (`app/services/media/visual_resolver.py`).

- **Jamais de hotlink** : toute image retenue est téléchargée, vérifiée par sa signature réelle (pas par le `Content-Type` déclaré), ré-encodée en WebP (EXIF retiré, dimension bornée) et stockée localement (table `media_assets`, migration `008_add_media_assets`) ; le cours ne référence que `GET /media/{asset_id}`.
- **Best-effort** : la résolution (téléchargement, re-recherche, génération) a un budget de temps et de concurrence dédié (`MEDIA_RESOLVE_TIMEOUT_SECONDS`, `MEDIA_RESOLVE_CONCURRENCY`) ; un bloc `image` dont la résolution échoue est simplement retiré du cours, jamais laissé sans image. À terme (une fois `pdf` et `generated` branchés aussi), un bloc `image` ne devra pas non plus compter comme le visuel de la règle « visuel d'abord » (`app/services/visual_validation.py`) — désactivé pour l'instant, ce contrôle ferait encore payer un appel Gemini de régénération pour les sources non branchées sans jamais pouvoir aboutir.
- **Déduplication** : les images sont indexées par sha256 du contenu ré-encodé ; deux blocs qui résolvent vers la même image (même figure PDF réutilisée, même image web) partagent une seule ligne.
- **Attribution** : les images sous licence CC BY / CC BY-SA affichent obligatoirement leur auteur et leur licence ; un bloc sans licence connue n'est pas retenu.

Lot 1 (fondations) pose le schéma, le stockage et le routage. `pdf` (figures de PDF sources) et `generated` (génération IA) restent non branchés.

**Lot 3 — images web (Wikimedia Commons → Openverse), actif** :
- `image_source: "web"` avec une `image_query` déclenche une recherche Wikimedia Commons, puis Openverse seulement si Commons n'a pas fourni assez de candidats (`MEDIA_WEB_MAX_CANDIDATES`). Chaque candidat est téléchargé depuis un **hôte en liste blanche uniquement** (`upload.wikimedia.org`, `api.openverse.org` — jamais l'hôte d'origine d'une photo Openverse, ex. Flickr), filtré par licence (`MEDIA_WEB_ALLOWED_LICENSES`, ND/NC exclus par défaut) et taille minimale (`MEDIA_WEB_MIN_WIDTH`).
- **Vérification de pertinence** : un appel Gemini Flash-Lite (`rank_images`) reçoit les miniatures déjà téléchargées + `image_alt`/`image_query`/le titre de section, et choisit la meilleure (ou aucune) — jamais l'URL des candidats, jamais leurs métadonnées traitées comme des instructions (`instruction/image_ranking_instructions.md`). Désactivable (`MEDIA_WEB_VERIFY_ENABLED=false`) : repli sur le premier candidat filtré.
- **Cache** (`media_query_cache`, TTL `MEDIA_WEB_CACHE_TTL_HOURS`) : un résultat positif comme négatif est mis en cache par requête normalisée — un hit ne fait ni appel réseau ni appel Gemini.
- `MEDIA_WEB_USER_AGENT` est **obligatoire** (nom de l'app, URL du repo, contact — exigé par la politique Wikimedia) : le résolveur reste inactif tant qu'il est vide, **sans erreur visible** — les cours se génèrent normalement mais sans aucune image web. Pas de clé à créer, juste une valeur texte à définir soi-même, ex. `MEDIA_WEB_USER_AGENT="Synthoria LLM/1.0 (https://github.com/<org>/<repo>; contact@example.com)"`.
- `OPENVERSE_CLIENT_ID`/`OPENVERSE_CLIENT_SECRET` sont optionnels (quota anonyme sinon, ou repli automatique si l'authentification échoue).

**Créer les identifiants Openverse** (optionnel) : [api.openverse.org/v1/auth_tokens/register](https://api.openverse.org/v1/auth_tokens/register/) → remplir le formulaire (nom, description, email) → `client_id` et `client_secret` renvoyés immédiatement, à copier dans `OPENVERSE_CLIENT_ID`/`OPENVERSE_CLIENT_SECRET`. Augmente le quota de requêtes par rapport à l'accès anonyme ; sans identifiants, le résolveur fonctionne quand même (Wikimedia Commons seul, ou Openverse en quota anonyme).

Les lots suivants brancheront les figures de PDF sources (Lot 2) et la génération IA (Lot 5).

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

Les tests de routes démarrent le lifespan FastAPI et exigent un PostgreSQL joignable via `DATABASE_URL` (fourni par un service PostgreSQL en CI).

## Structure du projet

```text
app/
├── api/              # routes + schémas FastAPI
├── core/             # configuration, exceptions, rate limiting
├── db/               # SQLAlchemy models + session async
├── repositories/     # accès aux données (course sessions, course plans)
├── services/         # Ollama, chunking, extraction PDF, vector store, Gemini Vision
│   ├── podcast/      # sérialisation du cours, script, normalisation TTS, client Piper, assemblage ffmpeg, pipeline
│   └── media/        # supports visuels : normalisation/stockage (Pillow), résolveur (GET /media/{asset_id})
│       └── providers/  # fournisseurs d'images web (Wikimedia Commons, Openverse)
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
├── image_ranking_instructions.md  # instructions LLM (vérification de pertinence des images web)
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
