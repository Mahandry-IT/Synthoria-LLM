# Plan — Synthoria : cours interactifs, lot 4 (exercices corrigés : expérience)

## Contexte
Lot 4 sur 4, presque entièrement frontend. Le backend du lot 3 expose déjà le détail d'un exercice (énoncé, indices, durée) et la correction. Ce lot câble l'expérience : énoncé riche, « Commencer », minuteur, indices, réponse LaTeX, correction, lien section vers exercice. Règles : RG-1, RG-7, RG-8. Aucune migration.

## Constats sur l'existant
- `src/features/exercises/` : `getExercise` et `answerExercise` existent mais ne sont appelés nulle part ; pas de hook pour le détail ni la correction. La liste (`ExercisesPanelView.tsx`) n'est pas cliquable.
- Énoncé : `BlockRenderer` (`src/features/course/components/blocks/BlockRenderer.tsx`) rend déjà texte, tableau, diagramme, graphique, widget ; `statement` est déjà typé en blocs de cours. Les widgets restent sandboxés (iframe sans `allow-same-origin`).
- Réponse : `RichTextEditor` (props `value`, `onChange`, `ariaLabel` obligatoire) avec extensions LaTeX, sortie Markdown. Motif de retour : `RecallBox.tsx` et `useRecallFeedback.ts`.
- Section : `SectionReader.tsx` affiche les encadrés et les vidéos en fin de section (conditions `atEnd` / progressif ~l.210-236 et ~l.365-370). `CourseView.tsx` ne connaît pas les exercices par section ; l'onglet actif se change par `storeTab('exercises')` (`courseTabs.ts`).
- Backend : la correction renvoie 429 (limiteur ou quota), 502, 503 avec le texte « Réessayez dans un instant : votre réponse est conservée. ».
- Tests : vitest en environnement node, sans jsdom ni testing-library ; le motif est `renderToStaticMarkup` et des fonctions pures. Un minuteur se teste avec `vi.useFakeTimers()`.

## Architecture retenue
Même dossier `src/features/exercises/`, sans nouvelle dépendance. La logique (minuteur, indices, brouillon) vit dans des fonctions pures testées ; les composants restent fins.

## Décisions à confirmer avant de coder
1. Fin du minuteur : décompte depuis `duration_minutes`. À zéro, un indicateur « temps écoulé » apparaît, sans bloquer la saisie ni la correction. Le minuteur s'arrête quand la correction arrive.
2. Refaire : après la correction, un bouton « Refaire l'exercice » ramène à l'écran « Commencer » (énoncé de nouveau masqué, minuteur remis à zéro, brouillon effacé). Pas de limite autre que le limiteur de correction.
3. Indices : les 3 indices restent masqués ; un bouton « Indice suivant » les affiche un par un, du général au précis. Compteur en mémoire, remis à zéro par « Refaire ».
4. Navigation : l'exercice s'ouvre dans l'onglet « Exercices corrigés » (liste vers détail, par un état local), sans nouvelle page. Un rechargement revient à la liste.
5. Brouillon : la réponse en cours est gardée dans `localStorage` par exercice (dans un `try/catch`) pour survivre à une erreur ou à un changement d'onglet, effacée à la correction réussie ou à « Refaire ».
6. Tests : pas de jsdom ni testing-library. Logique en fonctions pures, rendu statique pour l'affichage. Le clic, le focus et le clavier restent à vérifier à la main.
7. Corrections simultanées : un seul envoi en cours par apprenant (bouton désactivé pendant l'appel) ; le limiteur serveur de 10/min fait le reste.

## Étapes

### 22. Frontend : énoncé riche
- [ ] `ExerciseStatement.tsx` : rend `statement` avec `BlockRenderer`, sans HTML brut ; un bloc invalide est ignoré.
- [ ] Mise en page lisible sur mobile : tableaux et graphiques défilent dans leur cadre, texte à largeur contenue, widget à hauteur bornée (réutiliser le comportement existant).
- [ ] Hook `useExerciseDetail` (React Query, clé `["course-exercise", sessionId, exerciseId]`), déclenché seulement au clic sur « Commencer ».

### 23. Frontend : minuteur et « Commencer »
- [ ] Liste cliquable (`ExercisesPanelView.tsx`) : chaque exercice mène à une carte « avant » avec titre, section liée, ce qu'il couvre, durée et bouton « Commencer ». Aucun énoncé ni indice dans le DOM avant le clic (RG-8).
- [ ] `exerciseTimer.ts` (fonctions pures : temps restant, format mm:ss, état « écoulé ») et `ExerciseTimer.tsx` visible pendant l'exercice, `role="timer"`, sans annonce vocale à chaque seconde.
- [ ] Comportement de fin de temps : décision 1.

### 24. Frontend : réponse, indices, correction
- [ ] `ExerciseAnswerForm.tsx` : `RichTextEditor` avec LaTeX, compteur de longueur aligné sur la limite serveur (4000), envoi désactivé si vide, trop long ou en cours.
- [ ] `ExerciseHints.tsx` : indices masqués, révélés un par un (décision 3).
- [ ] `useAnswerExercise` (mutation sur `answerExercise`) et `ExerciseFeedbackView.tsx` : ce qui est juste, ce qui manque, démarche attendue, sans score (RG-1). Erreur : message clair, réponse conservée, bouton pour renvoyer.
- [ ] Brouillon et « Refaire l'exercice » : décisions 2 et 5.

### 25. Frontend : lien section vers exercice
- [ ] `CourseView.tsx` : lire `useExercises(sessionId)` et construire la table `section_id` vers exercice ; la passer à `SectionReader`.
- [ ] `SectionReader.tsx` : bouton « Faire l'exercice » en fin de section, seulement si un exercice rédigé existe pour cette section et que la liste est à l'état `done` (RG-7) ; aucun bouton sinon.
- [ ] Clic : `storeTab('exercises')` puis ouverture de l'exercice ; dans l'exercice, « Retour à la section » rouvre l'onglet Leçons et fait défiler jusqu'à la section (ancre `id` sur l'élément de section).

### 26. Résilience au quota
- [ ] Messages dédiés pour 429 (limite par minute ou quota), 502 et 503, avec « votre réponse est conservée » et un bouton « Renvoyer ».
- [ ] Un seul envoi à la fois (décision 7) ; pas de renvoi automatique en boucle.
- [ ] Détail introuvable (404) : retour à la liste avec message ; chargement et erreur du détail avec « Réessayer ».

### 27. Tests, documentation, déploiement
- [ ] Tests purs : minuteur, indices, brouillon, table section vers exercice, messages d'erreur. Rendu statique : carte « avant » sans énoncé, énoncé avec blocs, retour de correction sans score, bouton de section présent ou absent.
- [ ] README backend : décrire les routes d'exercices et le parcours ; ajouter les routes d'exercices à la collection Postman (`/postman`).
- [ ] Reconstruire les images, appliquer la migration `022`, lancer le worker `exercise-worker`, régénérer un cours de test de bout en bout avec Gemini, vérifier à la main dans le navigateur (clavier, focus, mobile).

## Points de vigilance sécurité
- Énoncé et correction viennent du modèle : affichés par `BlockRenderer` et `LatexText`, jamais en `dangerouslySetInnerHTML` ; les widgets gardent la sandbox existante.
- Le DOM ne contient ni énoncé ni indices tant que « Commencer » n'est pas cliqué (RG-8) ; la solution ne vient que de la correction.
- Brouillon en `localStorage` : texte de l'apprenant uniquement, aucune donnée de compte, lecture et écriture dans un `try/catch`.
- Aucun point attribué ni conservé (RG-1).

## Tests à prévoir
- Exercice sans indice ou sans énoncé exploitable : message adapté, pas de page cassée.
- Minuteur à zéro : indicateur affiché, saisie toujours possible.
- 429 sur la correction : réponse gardée, renvoi possible.
- Cours sans exercice ou liste `failed` : aucun bouton dans les sections.
- Anciens cours : onglet et sections sans régression.
