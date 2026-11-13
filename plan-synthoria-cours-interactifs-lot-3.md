# Plan — Synthoria : cours interactifs, lot 3 (exercices corrigés : données, génération, onglet)

## Contexte
Lot 3 sur 4. Il crée le stockage et la génération des exercices corrigés, leur correction par l'IA et l'onglet qui les liste. L'énoncé riche, le minuteur, les indices et le lien section vers exercice sont au lot 4. Règles : RG-1, RG-4, RG-5, RG-6, RG-8 (côté données), RG-9.

## Constats sur l'existant
- Le podcast fait référence : file en base `podcast_jobs`, worker séparé (`app/workers/podcast_worker.py`, `FOR UPDATE SKIP LOCKED`), job idempotent par `params_hash`, statuts, reprise sur interruption, `release_stale`. Route de lancement 202 et route de lecture d'état (`app/api/podcast_routes.py`).
- La correction « Explique avec tes mots » est un bon modèle : `app/services/recall_evaluator.py`, schéma `app/schemas/recall.py`, réponse du client encadrée par `<learner_answer>`, limiteur `SlidingWindowLimiter`.
- Dernière migration : `021`. Modèles SQLAlchemy 2.0 dans `app/db/models.py`.
- Aucune authentification ni propriétaire de cours : le contrôle d'accès se limite à l'existence de la session (404), comme le podcast. Les limiteurs sont par IP, en mémoire.
- Frontend : onglet « Exercices corrigés » en placeholder dans `CourseView.tsx` (~l.193). Hooks de suivi de job dans `src/features/podcast/hooks/`, champ LaTeX `RichTextEditor`, modale `components/Modal.tsx`.
- Tests backend : routes avec `TestClient` et Gemini mocké, repositories avec session mockée ; pas de base réelle.

## Architecture retenue
On reprend le schéma du podcast, sans nouvelle couche. Table de travail pour la génération, table pour les exercices, worker dédié. Un seul appel Gemini produit tout le lot d'exercices du cours (RG-9) : la même réponse choisit les sections qui en ont besoin (RG-6).

## Décisions à confirmer avant de coder
1. File + worker (comme le podcast) plutôt que `BackgroundTasks` : survit à un redémarrage et gère proprement le 429. Coût : un worker de plus à lancer (`python -m app.workers.exercise_worker`, service Docker).
2. Un seul appel Gemini pour le lot : l'IA reçoit les titres et sous-thèmes des sections, choisit celles qui en ont besoin et rédige tous les exercices. Plafond par mode : express 2, standard 3, approfondi 4 (à ajuster).
3. Contrat d'un exercice : `id`, `section_id` (facultatif), `position`, `title`, `covers` (ce qu'il couvre), `statement` (liste de blocs du format existant : texte, tableau, graphe, courbe, widget), `hints` (3, du général au précis), `duration_minutes`, `expected_solution` (démarche). `expected_solution` ne quitte jamais le serveur en dehors de la réponse de correction.
4. RG-8 côté serveur : la liste renvoie titre, section, `covers`, état ; l'énoncé et les indices ne viennent qu'avec la route de détail, appelée au clic sur « Commencer ».
5. La réponse de l'apprenant n'est pas enregistrée en base (aucun point, RG-1). Elle reste dans le navigateur ; le serveur ne stocke que l'état et les exercices.
6. Sans comptes ni propriétaire, le contrôle d'accès reste « la session existe », comme le podcast, avec limiteurs par IP.

## Étapes

### 15. Conception des exercices
- [ ] Figer le contrat de la décision 3 et les états : `absent` (aucune ligne), `pending`, `generating`, `done`, `failed`.
- [ ] Contrat des routes : lancer, état, liste, détail, correction (voir section 18).
- [ ] Critère RG-6 : donné à l'IA dans le prompt (section qui contient un calcul, une méthode ou un cas à appliquer), validé par un plafond côté serveur.

### 16. Modèle et migration
- [ ] `app/db/models.py` : table `exercise_jobs` (session_id unique, status, params_hash, attempts, locked_at, error_message, timestamps) et table `course_exercises` (session_id, section_id nullable, position, title, covers, statement JSONB, hints JSONB, duration_minutes, expected_solution, timestamps).
- [ ] `migrations/versions/022_add_course_exercises.py`, réversible, index `idx_course_exercises_session`.
- [ ] Repositories `app/repositories/exercise_job_repository.py` et `exercise_repository.py` sur le modèle de `podcast_job_repository.py`.

### 17. Génération asynchrone
- [ ] Schéma Gemini `app/schemas/exercise_generation.py` (champs requis dans le JSON schema, comme pour les sections) et service `app/services/exercise_generator.py` : un appel, validation (plafond, `section_id` connu, énoncé non vide, 3 indices), blocs d'énoncé nettoyés avec les mêmes contrôles que les sections (blocs sans contenu écartés).
- [ ] `app/workers/exercise_worker.py` : réclamation `SKIP LOCKED`, tentatives bornées, requeue sur interruption, 429 renvoyé à une nouvelle tentative avec attente, `release_stale`.
- [ ] Un seul lot par cours (RG-9) : la ligne de job par session est unique ; relancer ne régénère que si l'état est `failed`.
- [ ] Prérequis RG-4 : le cours et son quiz sont terminés ; sinon 409.

### 18. Routes des exercices
- [ ] `app/api/exercise_routes.py` : `POST /courses/{session_id}/exercises/generate` (202, sur demande seulement, RG-5), `GET /courses/{session_id}/exercises` (état et liste sans énoncé), `GET /courses/{session_id}/exercises/{exercise_id}` (énoncé et indices), `POST .../{exercise_id}/answer` (correction).
- [ ] Limiteurs par route (génération ≈ 2/min, correction ≈ 10/min), 404 sur session ou exercice inconnu, vérification que l'exercice appartient bien à la session.
- [ ] Brancher le routeur dans l'application et documenter dans OpenAPI.

### 19. Correction de la réponse par l'IA
- [ ] `app/services/exercise_evaluator.py` sur le modèle de `recall_evaluator.py` : énoncé et solution lus en base, réponse encadrée par `<learner_answer>` et balises neutralisées, taille bornée.
- [ ] Sortie : ce qui est juste, ce qui manque ou est faux, démarche attendue. Aucun score (RG-1).
- [ ] Échec ou 429 : erreur explicite qui invite à réessayer ; le client garde la réponse.

### 20. Frontend : fenêtre et onglet Exercices corrigés
- [ ] `src/features/exercises/` : `exercise.schema.ts`, `exercise.api.ts`, hooks de lancement et de suivi (modèle `usePodcastJob`).
- [ ] Fenêtre « Générer les exercices ? » (`Modal`) une fois le cours et le quiz prêts ; « Non » ne génère rien.
- [ ] Onglet : bouton « Générer les exercices » si aucun exercice, avancement pendant la génération, message d'échec avec « Réessayer », liste (titre, section liée, ce qu'il couvre) sans énoncé. Remplacer le placeholder de `CourseView.tsx`.

### 21. Tests du lot 3
- [ ] Backend : schéma requis, plafond et sections retenues, génération en un seul appel, idempotence du job, états, routes (404, 409, limiteurs), énoncé absent de la liste, solution absente des routes de lecture, correction (injection dans la réponse, échec).
- [ ] Non-régression de quota : générer un cours n'appelle jamais l'API des exercices.
- [ ] Frontend : fenêtre, bouton de génération, états, liste sans énoncé.
- [ ] Essai réel avec Gemini : un lot d'exercices sur un cours existant.

## Points de vigilance sécurité
- Texte du modèle et réponse de l'apprenant non fiables : affichés sans HTML, balises neutralisées, taille bornée.
- Pas d'authentification dans le projet : IDOR non couvert par un propriétaire, vérifier au moins que l'exercice demandé appartient à la session de l'URL.
- Limiteurs par IP en mémoire (non partagés entre processus) : suffisants pour ce lot, à durcir avec les comptes.
- `expected_solution` jamais renvoyée avant la correction.

## Tests à prévoir
- Deux lancements de génération sur le même cours : un seul lot.
- 429 pendant la génération : le job repart, pas d'exercices partiels.
- Cours sans section qui en ait besoin : état `done` avec zéro exercice et message adapté.
- Les anciens cours s'ouvrent sans régression, onglet vide avec bouton.
