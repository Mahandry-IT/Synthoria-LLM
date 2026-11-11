# Plan — Synthoria : cours interactifs, lot 1 (quiz enrichi et navigation)

## Contexte
Lot 1 sur 4. Il ajoute au quiz deux types de question (réponse numérique, appariement) et la navigation à trois onglets sous le podcast. Périmètre : backend Synthoria LLM et frontend Synthoria Studio. Cahier des charges : US 8, 9 et 10, règles RG-2 et RG-3.

## Architecture retenue
On garde l'existant, sans nouvelle couche.
- Backend : la question Gemini (`QuizQuestion`, `app/schemas/course_generation.py`) devient une question d'API (`ApiQuizQuestion`, `app/api/schemas.py`) via `_map_quiz_question` (`app/services/course_generator.py`). Les questions de la banque sont stockées en JSON (`quiz_questions.payload`) et les réponses des tentatives en JSON (`quiz_attempts.answers`). La note est calculée par `grade` dans `app/services/quiz_bank.py`.
- Frontend : `src/features/course/` (`QuizQuestionStep`, `QuizQuestionRadioGroup`, `useQuizFlow`, `quizAttempt`, `quizGrading`, `QuizResults`).
- Conséquence : probablement **aucune migration** (le contenu est en JSON). À confirmer à l'étape 2.
- Une question garde son format actuel (`options`, `correct_option_indices`). Un champ `kind` (`choice` par défaut, `numeric`, `matching`) distingue les nouveaux types, pour que les anciens quiz restent lisibles.

## Étapes

### 1. Conception du quiz
- [ ] Champ `kind` : `choice` (défaut), `numeric`, `matching`.
- [ ] Numérique : `expected_value` (nombre) et `unit` facultative. Entier ou décimal se déduit de la valeur (RG-2) : `expected_value` entier → égalité stricte, sinon écart de 0,0001.
- [ ] Appariement : `subjects` (sujets) et `options`, de même longueur ; la bonne réponse est `correct_matches` (indice d'option par sujet). Tout ou rien, comme le choix multiple actuel.
- [ ] Format de réponse d'une tentative : choix = liste d'indices (inchangé) ; numérique = un nombre ; appariement = liste d'indices d'option par sujet, `null` si vide.
- [ ] La question publique (avant correction) ne contient ni `expected_value` ni `correct_matches`.
- [ ] Périmètre des nouveaux types : quiz final et banque de questions. Pas dans les questions de fin de section, qui sont refaites au lot 2.

### 2. Backend : schémas et stockage
- [ ] `QuizQuestion` (Gemini) : ajouter `kind`, `expected_value`, `unit`, `subjects`, `correct_matches`, tous facultatifs avec défaut, sans changer les champs existants.
- [ ] `ApiQuizQuestion` : mêmes champs ; `_map_quiz_question` les propage.
- [ ] `public_question` (`quiz_bank.py`) : masquer les nouvelles réponses.
- [ ] Lecture des anciennes questions : `kind` absent = `choice`. `original_quiz` accepte les nouveaux types (aujourd'hui il écarte une question sans `correct_option_indices`).
- [ ] Confirmer qu'aucune migration n'est nécessaire (JSON) ; sinon `migrations/versions/022_*`.

### 3. Backend : génération
- [ ] `instruction/course_generation_instructions.md` et la consigne du quiz : quand produire un numérique (calcul avec un résultat unique) ou un appariement (sujets à associer à des options), et en quelle proportion.
- [ ] Validation côté serveur : appariement avec autant d'options que de sujets, au moins 3 ; numérique avec une valeur finie.
- [ ] Rafraîchissement de la banque (`build_refill_prompt`, `QuizRefillSchema`) : accepter les nouveaux types. Même nombre d'appels qu'aujourd'hui.
- [ ] Essai réel avant de figer : 2 appels Gemini pour mesurer que le schéma enrichi reste supporté par le modèle léger (une section a déjà été vidée par un schéma trop permissif).

### 4. Backend : correction et tentatives
- [ ] `validate_answers` : valider le format de réponse selon `kind` (indice hors bornes, option utilisée deux fois dans un appariement, nombre non fini).
- [ ] `grade` : corriger les trois types ; numérique selon RG-2 (entier strict, décimal à 0,0001) ; renvoyer la bonne réponse dans le détail de correction.
- [ ] `quiz_routes.py` : les routes de démarrage et de soumission passent déjà par ces fonctions ; adapter les schémas de requête et de réponse (`QuizAttemptSubmit`, résultats).
- [ ] Conserver la limite de débit existante (`_limit_submit`).

### 5. Frontend : nouvelles questions
- [ ] `course.types.ts` et le schéma de lecture : ajouter `kind` et les champs, tolérants (valeur par défaut `choice`).
- [ ] `QuizQuestionNumeric` : champ nombre, virgule ou point acceptés, valeur vide autorisée, unité affichée.
- [ ] `QuizQuestionMatching` : un `select` par sujet ; une option prise disparaît des autres listes ; la valeur vide la remet (RG-3) ; autant d'options que de sujets.
- [ ] `quizGrading.ts` : mêmes règles que le serveur (RG-2) pour le retour immédiat ; le serveur reste l'autorité pour la note du quiz.
- [ ] `quizAttempt.ts`, `useQuizFlow`, `QuizQuestionStep`, `QuizResults` : formats de réponse et affichage de la correction pour les trois types.
- [ ] Vérifier la compatibilité de `review/` (cartes de révision) avec un `kind` inconnu : ignorer les nouveaux types plutôt que casser.

### 6. Frontend : navigation à trois onglets
- [ ] Repérer le composant qui affiche le podcast et son parent dans la page du cours.
- [ ] Composant d'onglets Leçons, Exercices corrigés, Quiz sous le podcast ; onglet actif signalé et utilisable au clavier.
- [ ] Onglet Exercices corrigés : état vide (« pas encore d'exercices »), sans bouton de génération (lot 3).
- [ ] Mémoriser l'onglet actif au rechargement.
- [ ] Le quiz s'affiche dans son onglet ; la leçon reste affichée dans son onglet.

### 7. Tests, documentation, déploiement
- [ ] Backend : questions numérique et appariement (validation, correction, entier strict, décimal à 0,0001, option dupliquée, `kind` absent), non-régression des anciens quiz, masquage des réponses.
- [ ] Frontend : appariement (options exclusives, valeur vide), numérique (virgule, vide), onglets, lecture d'une ancienne question.
- [ ] Collection Postman : exemples de question numérique et d'appariement et de leur réponse.
- [ ] README : décrire les nouveaux types.
- [ ] Reconstruire les images `api` et `worker`.

## Points de vigilance sécurité
- La question publique ne doit jamais contenir `expected_value` ni `correct_matches` avant la correction.
- Valider toutes les réponses côté serveur (format, bornes, doublons) ; le frontend ne fait que l'ergonomie.
- Texte des sujets et des options : texte non fiable, affiché sans HTML.

## Tests à prévoir
- Un quiz enregistré avant ce lot se lit et se corrige comme avant.
- Réponse numérique `4` pour `4` accepté, `4,0001` refusé ; `0,3333` pour `0,33333` accepté.
- Appariement : tout ou rien, un sujet vide compte comme faux.
- Quota : le nombre d'appels Gemini pour générer un quiz est inchangé.

## Décisions à confirmer avant de coder
1. Les nouveaux types vivent dans le quiz final et la banque, pas dans les questions de fin de section (reprises au lot 2).
2. Appariement noté « tout ou rien ». Une note partielle par sujet est possible mais le cahier des charges ne la demande pas.
3. Une réponse décimale qui tombe sur un entier (4,0 attendu) est traitée comme entière.
