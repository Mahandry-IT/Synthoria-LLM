# Todos — Synthoria : cours interactifs et exercices corrigés

Source : `docs/cahier-des-charges-synthoria-cours-interactifs.md`. Backend : Synthoria LLM. Frontend : Synthoria Studio.
Contrainte transversale : le quota Gemini est limité, donc chaque lot borne le nombre d'appels par cours et ne déclenche rien sans demande de l'apprenant.

## Lot 1 — Quiz enrichi et navigation (V1)

### 1. Conception du quiz
- [x] Définir les types de question : choix multiples, réponse numérique, appariement
- [x] Fixer le contrat de réponse de l'API pour chaque type (bonne réponse, tolérance, options)
- [x] Décider où vit la règle entier strict / décimal à 0,0001 (RG-2) : interface seule, valeur attendue fournie par le serveur

### 2. Backend : schémas et migration
- [x] Étendre les schémas de question (numérique, appariement) avec défaut pour les questions existantes
- [x] Migration de la banque de questions si le stockage change
- [x] Rendre la lecture des anciens quiz inchangée

### 3. Backend : génération des nouvelles questions
- [x] Adapter les consignes de génération du quiz (quand produire un numérique ou un appariement)
- [x] Valider côté serveur : autant d'options que de sujets, valeur numérique exploitable
- [x] Garder un seul appel pour le quiz entier (pas d'appel par question)

### 4. Backend : correction et tentatives
- [x] Corriger les trois types de question à la soumission
- [x] Enregistrer les tentatives avec les nouveaux types
- [x] Renvoyer la bonne réponse pour l'affichage de la correction

### 5. Frontend : nouvelles questions
- [x] Composant réponse numérique avec la règle de RG-2
- [x] Composant appariement : listes déroulantes, option prise retirée des autres, valeur vide qui la remet
- [x] Correction affichée pour chaque type

### 6. Frontend : navigation à trois onglets
- [x] Onglets Leçons, Exercices corrigés, Quiz sous le podcast
- [x] Onglet Exercices corrigés accessible même sans exercice (état vide)
- [x] Conserver l'onglet actif au rechargement

### 7. Tests, documentation, déploiement du lot 1
- [x] Tests backend (validation, correction) et frontend (appariement, numérique)
- [x] Mettre à jour la collection Postman et le README si le contrat change
- [x] Appliquer la migration, reconstruire les images

## Lot 2 — Nouveau format de leçon (V1)

### 8. Conception du format de section
- [ ] Définir l'ordre : Pourquoi/Quoi/Comment, tableau ou exemple, démonstration pas à pas, « À toi », fin de section
- [ ] Définir « Ce qu'il faut retenir » et « Ce qu'il faut éviter » comme éléments de fin de section
- [ ] Décider comment l'IA choisit entre mini-QCM et « Explique avec tes mots » (point ouvert du cahier des charges)

### 9. Backend : schémas de section
- [ ] Ajouter retenir et piège au schéma de section avec défaut pour les cours existants
- [ ] Ajouter le mini-QCM de fin de section
- [ ] Étendre la complétude : une section sans ses éléments obligatoires est incomplète

### 10. Backend : génération et budget d'appels
- [ ] Revoir les consignes (résumés simples, exemple, démonstration, bilan)
- [ ] Adapter les règles par mode (express, standard, approfondi)
- [ ] Vérifier avec le vrai Gemini que le schéma reste supportable par le modèle léger

### 11. Backend : vidéos par section
- [ ] Rattacher chaque vidéo à sa section (le classement sait déjà associer des sections)
- [ ] Garder les notes de vidéo existantes
- [ ] Section sans vidéo : aucun espace vide

### 12. Frontend : rendu d'une section
- [ ] Affichage de retenir et piège en fin de section
- [ ] Mini-QCM de fin de section
- [ ] Démonstration pas à pas : réutiliser les widgets et les étapes du graphique

### 13. Frontend : vidéos par section
- [ ] Bloc vidéos à la fin de chaque section
- [ ] Ajout, modification et suppression de notes depuis la section

### 14. Tests, documentation, déploiement du lot 2
- [ ] Tests de schéma, de complétude et de rendu
- [ ] Anciens cours : vérifier qu'ils s'affichent sans régression
- [ ] Décision sur la régénération des cours existants
- [ ] Mettre à jour le README (nouveaux types de question du lot 1, nouveau format de section)

## Lot 3 — Exercices corrigés : données, génération, onglet (V2)

### 15. Conception des exercices
- [ ] Contrat d'un exercice : titre, ce qu'il couvre, énoncé riche, indices, section liée, correction attendue
- [ ] États de génération : absent, en cours, prêt, en échec
- [ ] Budget d'appels par cours (RG-9) et critère « la section en a besoin » (RG-6)

### 16. Modèle et migration
- [ ] Stockage des exercices par cours, lien optionnel vers une section
- [ ] Stockage de l'état de génération
- [ ] Migration réversible

### 17. Génération asynchrone
- [ ] Tâche en arrière-plan sur le modèle du podcast
- [ ] Un seul lot par cours, après la leçon et le quiz (RG-4)
- [ ] Arrêt propre et reprise possible en cas de 429

### 18. Routes des exercices
- [ ] Lancer la génération (sur demande seulement, RG-5)
- [ ] Lire l'état, la liste et le détail d'un exercice
- [ ] Contrôle d'accès et limites de débit

### 19. Correction de la réponse par l'IA
- [ ] Corriger une réponse en LaTeX sur le modèle de « Explique avec tes mots »
- [ ] Aucun point attribué (RG-1)
- [ ] Message de reprise sans perte de la réponse en cas d'échec

### 20. Frontend : fenêtre et onglet Exercices corrigés
- [ ] Fenêtre « Générer les exercices ? » à la fin de la génération du cours
- [ ] Bouton « Générer les exercices » dans l'onglet quand il n'y en a pas
- [ ] Liste des exercices avec la section liée et ce qu'ils couvrent, suivi de l'avancement

### 21. Tests du lot 3
- [ ] Tests de génération (budget, sections retenues), de routes et de correction
- [ ] Test de non-régression des quotas sur la génération du cours

## Lot 4 — Exercices corrigés : expérience (V2)

### 22. Frontend : énoncé riche
- [ ] Énoncé mêlant texte, tableau, graphe et courbe
- [ ] Lisible sur mobile

### 23. Frontend : minuteur et « Commencer »
- [ ] Énoncé masqué avant « Commencer », seul ce que l'exercice couvre est visible (RG-8)
- [ ] Minuteur visible pendant l'exercice
- [ ] Décider du comportement à la fin du temps (point ouvert)

### 24. Frontend : réponse, indices, correction
- [ ] Champ LaTeX pour répondre
- [ ] Indices masqués, affichés un par un
- [ ] Affichage de la correction

### 25. Frontend : lien section vers exercice
- [ ] Bouton dans la section, visible seulement si l'exercice est rédigé (RG-7)
- [ ] Retour de l'exercice vers sa section

### 26. Résilience au quota
- [ ] Messages clairs sur 429, bouton de reprise
- [ ] Limiter les corrections simultanées par apprenant

### 27. Tests, documentation, déploiement du lot 4
- [ ] Tests de composants et de parcours complet
- [ ] Postman et README à jour
- [ ] Reconstruire les images, régénérer un cours de test de bout en bout
