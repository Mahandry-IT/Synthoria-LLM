# Plan — Synthoria : cours interactifs, lot 2 (nouveau format de leçon)

## Contexte
Lot 2 sur 4. Il ajoute à chaque section « Ce qu'il faut retenir » et « Ce qu'il faut éviter », fiabilise la démonstration pas à pas, et rattache les vidéos à leur section. Backend Synthoria LLM et frontend Synthoria Studio. Cahier des charges : US 1 à 7, règles RG-6 non concernée.

## Constats sur l'existant (lus dans le code)
- Le mini-QCM de fin de section existe déjà : `check_questions`, affiché « Vérifie » (`SectionCheck.tsx`). « Explique avec tes mots » = `recall_prompt`, « À toi » = `faded_example`. On ne crée donc pas de nouveau mini-QCM : on garde ces deux éléments.
- Rien n'existe pour « retenir » ni pour « piège » au niveau de la section. Le bloc `pitfall` (description, pourquoi, comment l'éviter) existe dans le contenu ; la section `common_pitfalls` est une section de cours entière, pas un encadré de fin de section.
- Les vidéos sont dans le JSONB `gemini_response.videos`, sans table. Le `section_id` n'est posé que par `rank_videos`. Si le classement est désactivé ou échoue, la vidéo reste au niveau du cours.
- Les sections sont en JSONB : pas de migration SQL, mais un défaut est requis pour les anciens cours.
- Les règles par mode vivent dans `app/services/course_depth.py`, pas dans les `.md`.

## Architecture retenue
Aucune nouvelle couche. On étend `Section` (`app/schemas/course_generation.py`), `CourseSection` (`app/api/schemas.py`), le rendu `SectionReader.tsx` et `course.schema.ts` / `course.types.ts`. Aucun appel Gemini ajouté : retenir et piège sortent dans l'appel de la section, les vidéos gardent l'appel de classement existant.

## Décisions à confirmer avant de coder
1. Fin de section : on garde « Vérifie » (mini-QCM) et « Explique avec tes mots » tels quels, sans nouveau mini-QCM. L'IA ne choisit pas : les deux restent, comme aujourd'hui.
2. « Retenir » = 2 à 4 puces courtes ; « Éviter » = 1 à 2 pièges, chacun avec « comment l'éviter ». Champs de la section, distincts du bloc `pitfall` et de la section « Pièges courants ».
3. Anciens cours : gardent leur format, sans régénération. Les deux encadrés ne s'affichent que s'ils existent.
4. Vidéo sans `section_id` : affichée dans le bloc de vidéos du cours (comportement actuel), jamais d'espace vide.

## Étapes

### 8. Conception du format de section
- [ ] Ordre d'une section : Challenge, Pourquoi, Quoi, Comment (tableau ou exemple, démonstration), « À toi », « Vérifie » ou « Explique », puis Retenir et Éviter, puis vidéos.
- [ ] Contrat : `key_takeaways: list[str]` (2 à 4) et `pitfalls_to_avoid: list[{mistake, how_to_avoid}]` (1 à 2), défaut vide.
- [ ] Confirmer les décisions 1 à 4 ci-dessus.

### 9. Backend : schémas de section
- [ ] `app/schemas/course_generation.py` : ajouter les deux champs à `Section` avec défaut ; les rendre requis dans le JSON schema envoyé à Gemini (même mécanisme que `_require_development_content`, pour éviter les sections vides).
- [ ] `app/api/schemas.py` : `CourseSection` expose les deux champs, défaut vide pour les anciens cours ; texte traité comme non fiable (affiché sans HTML).
- [ ] Complétude : `_structural_gaps` (`course_plan_generator.py`) signale une section de développement sans retenir ni piège (problème bloquant pour les sections nouvelles). `is_incomplete_section_dict` (`course_generator.py`) ne marque pas incomplets les anciens cours sans ces champs : seule une section générée après ce lot est contrôlée.
- [ ] `section_regenerator.py` : le prompt de réparation redemande les champs manquants.

### 10. Backend : génération et budget d'appels
- [ ] `instruction/course_generation_instructions.md` : décrire Retenir et Éviter, exiger un exemple complet, et rappeler que la démonstration pas à pas passe par un widget ou un graphique à étapes quand elle apporte quelque chose.
- [ ] `app/services/course_depth.py` : bornes par mode (express 2 retenir et 1 piège ; standard 3 et 1 ; approfondi 3-4 et 2). Pas de chiffre dans les `.md`.
- [ ] Essai réel avec Gemini (Docker, DNS surchargé) : 2 cours, vérifier que le schéma enrichi tient sur le modèle léger et que le nombre d'appels est inchangé.

### 11. Backend : vidéos par section
- [ ] `course_videos.py` : fiabiliser le rattachement. Sans classement, associer la vidéo à une section par repli (sujet de la section), sinon laisser `section_id` à null.
- [ ] Conserver le plafond `course_videos_max_per_section` (2) et les notes existantes (`course_video_notes`) ; aucune migration.
- [ ] Section sans vidéo : rien dans la réponse, pas de bloc vide.

### 12. Frontend : rendu d'une section
- [ ] `course.types.ts` et `course.schema.ts` : champs `key_takeaways` et `pitfalls_to_avoid`, tolérants (défaut vide).
- [ ] `SectionReader.tsx` : deux encadrés en fin de section (« Ce qu'il faut retenir », « Ce qu'il faut éviter »), dévoilés avec `sectionProgress.ts` à la fin des étapes ; absents si vides.
- [ ] Démonstration pas à pas : réutiliser `WidgetBlock` et `ChartStepControls` ; aucune nouvelle logique.
- [ ] « Vérifie » inchangé.

### 13. Frontend : vidéos par section
- [ ] `SectionVideos.tsx` placé après les deux encadrés, en fin de section ; masqué si la section n'a pas de vidéo.
- [ ] Notes : ajouter, modifier, supprimer fonctionnent déjà (`VideoNoteButton`, `useSaveVideoNote`) ; vérifier qu'elles marchent depuis la section.
- [ ] `videoGrouping.ts` : les vidéos sans section restent au niveau du cours.

### 14. Tests, documentation, déploiement
- [ ] Backend : schéma (champs requis dans le JSON schema, défaut pour les anciens cours), complétude, rattachement vidéo, prompt de réparation.
- [ ] Frontend : rendu des encadrés présents et absents, ancien cours sans régression, vidéos par section.
- [ ] Anciens cours : ouvrir un cours enregistré avant le lot et vérifier l'affichage.
- [ ] Décision sur la régénération des cours existants : proposition = ne pas régénérer.
- [ ] README : mettre à jour la section « Format du cours » (l.176-190), les modes de cours (l.195-202) et « Quiz — Schéma de réponse » (l.365-383) pour les types numérique et appariement du lot 1 et les nouveaux encadrés.
- [ ] Reconstruire les images `api`, `worker` et frontend (étape faite à la main par l'utilisateur au lot 1).

## Points de vigilance sécurité
- Les textes retenir et piège viennent du modèle : affichés sans HTML, longueur bornée côté API.
- Aucun nouvel accès réseau ; l'API YouTube et la limite de débit des notes restent inchangées.
- La route de note vérifie déjà que la vidéo appartient au cours ; ne pas l'affaiblir.

## Tests à prévoir
- Section générée sans `key_takeaways` : signalée incomplète et réparée.
- Cours ancien sans les champs : se lit, s'affiche, n'est pas marqué incomplet.
- Vidéo sans `section_id` : affichée au niveau du cours ; section sans vidéo : aucun espace vide.
- Quota : nombre d'appels Gemini par cours inchangé.
