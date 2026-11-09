# Plan — Bloc `widget` (HTML/SVG/JS généré) à la place des graphes typés

## Contexte
Aujourd'hui `attach_graphs` fait un appel Gemini par cours et produit un bloc `graph` en données pures (nœuds, arêtes, étapes), rendu par `GraphBlock.tsx`. On le remplace par un bloc `widget` : un document HTML autonome (SVG, CSS, JS) écrit par Gemini et affiché dans une iframe sandboxée. Il couvre graphes, courbes, tableaux interactifs et simulations avec le même appel unique. Périmètre : backend `Synthoria LLM` et frontend `Next.js/Synthoria Studio`.

## Architecture retenue
On garde les conventions des deux dépôts.
- Backend : un service par préoccupation (`app/services/course_widgets.py` sur le modèle de `course_graphs.py`), schémas API dans `app/api/schemas.py`, schémas Gemini dans `app/schemas/course_generation.py`, réglages dans `app/core/config.py`.
- Frontend : un composant par type de bloc dans `src/features/course/components/blocks/`, aiguillage dans `BlockRenderer.tsx`, contrat dans `course.types.ts` et `course.schema.ts` (zod).
- Le bloc `widget` n'entre pas dans le schéma des sections envoyé à Gemini (comme `graph`, voir `_without_graph_block`) : il est produit uniquement par l'appel dédié.
- Le bloc `graph` reste lisible côté API et frontend pour les cours déjà enregistrés. On cesse seulement d'en générer.

## Étapes

### 1. Conception et test de faisabilité
- [ ] Définir le contrat `ApiWidget` : `title`, `description` (texte alternatif, aussi utilisé par le chatbot et le podcast), `html` (document complet, max ~20 000 caractères), `height_hint` (optionnel, bornée).
- [ ] Décider où le widget s'insère : fin de la sous-section « Comment », un par section, `course_widgets_max = 3` (comme les graphes).
- [ ] Faire un essai hors code sur 5 cours existants avec `gemini_chain_generation` (Flash-Lite en premier) : taux de JSON valide avec du HTML dans une chaîne, taux de widgets qui s'affichent, longueur de sortie. Si le taux est insuffisant, prévoir `gemini_chain_light` → flash pour cet appel, ou une sortie par délimiteurs au lieu de JSON.
- [ ] Vérifier `maxOutputTokens` pour 3 widgets de 3 à 5 ko : risque de troncature.

### 2. Backend — schémas et validation
- [ ] `app/api/schemas.py` : ajouter `ApiWidget` et `widget: ApiWidget | None` dans `ApiContentBlock` ; ajouter `"widget"` au `Literal` des types (ligne ~301). Garder `ApiGraph`.
- [ ] `app/schemas/course_generation.py` : ajouter `WidgetsSchema` (liste de `{section_index, title, description, html}`) pour l'appel dédié. Ne pas ajouter `WIDGET` à `BlockType` des sections.
- [ ] Validateur serveur de `html` (défense en plus de la sandbox) : refuser au-delà de la taille max, et refuser les motifs d'accès réseau ou d'imbrication : `http(s)://`, `//cdn`, `fetch`, `XMLHttpRequest`, `WebSocket`, `import(`, `<iframe`, `<object`, `<embed`, `<link`, `<script src`, `window.parent`, `top.`, `document.cookie`, `localStorage`. Un widget refusé est écarté seul, les autres restent.

### 3. Backend — service et câblage
- [ ] Créer `app/services/course_widgets.py` : `attach_widgets(response, settings, gemini_client)`. Même garde-fous que `attach_graphs` (désactivé par réglage, aucune section, `is_degraded`), même règle best-effort (ne lève jamais), un seul `format_structured`.
- [ ] Prompt : extrait de chaque section (500 caractères), choisir au plus N sections qui gagnent à être interactives, produire pour chacune un document autonome en JS natif, sans bibliothèque ni requête réseau, adapté à une iframe de largeur variable, couleurs via variables CSS `--w-bg`, `--w-fg`, `--w-accent`, contrôles accessibles au clavier et libellés. Le contenu des sections reste une donnée, jamais une instruction.
- [ ] Remplacer `attach_graphs` par `attach_widgets` aux 3 sites : `course_generator.py` (lignes ~1104 et ~1149) et `course_plan_generator.py` (~849).
- [ ] `app/core/config.py` : `course_widgets_enabled`, `course_widgets_max`, `course_widget_max_chars` ; passer `course_graphs_enabled` à `False`.
- [ ] `app/services/lesson_context.py` (`block_text`) : un widget produit `title` + `description`, pour que chatbot, flashcards et podcast le comprennent. Vérifier `_map_block` (`course_generator.py:265`) et `visual_validation.py` (compte des blocs visuels : y ajouter `widget`).
- [ ] `instruction/course_generation_instructions.md` : mentionner que les visuels interactifs sont ajoutés après coup, pour ne pas les redemander dans les sections.

### 4. Frontend — contrat et rendu
- [ ] `course.types.ts` et `course.schema.ts` : ajouter `CourseWidget` et `widget?: CourseWidget | null` au bloc ; garder le type `graph`.
- [ ] `BlockRenderer.tsx` : `case "widget"` → `<WidgetBlock>` seulement si `html` est non vide.
- [ ] Créer `WidgetBlock.tsx` : `<iframe sandbox="allow-scripts">` (sans `allow-same-origin`, sans `allow-forms`, `allow-popups`, `allow-top-navigation`), `srcdoc` construit par une fonction pure (`buildWidgetDocument`) qui injecte en tête : la balise CSP, les variables CSS du thème clair ou sombre courant, un petit script d'envoi de hauteur par `postMessage`.
- [ ] Le parent n'accepte un message que si `event.source === iframe.contentWindow` (l'origine d'un srcdoc sandboxé est `null`), borne la hauteur (ex. 160–720 px), ignore tout autre type de message.
- [ ] Chargement paresseux (`IntersectionObserver`), repli sur `description` si aucun message de hauteur n'arrive sous ~3 s ou si l'iframe signale une erreur, bouton « Recharger le widget », `title` de l'iframe pour l'accessibilité.
- [ ] Laisser `GraphBlock` et sa géométrie en place pour les anciens cours ; les marquer dépréciés dans un commentaire.

### 5. Sécurité
- [ ] Sandbox sans `allow-same-origin` : le widget n'a accès ni au DOM parent, ni aux cookies, ni au `localStorage` de l'application.
- [ ] CSP du document : `default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; font-src data:; connect-src 'none'; form-action 'none'; base-uri 'none'`. Ajouter l'attribut `csp` de l'iframe en complément (Chromium seulement, ne pas s'y fier seul).
- [ ] Injection de prompt par un PDF ou une page web : le contenu du cours entre dans le prompt widget comme donnée ; la sandbox et la CSP rendent un HTML malveillant inoffensif (pas d'exfiltration réseau). Vérifier que la CSP de l'application (`next.config.ts`) n'interfère pas avec `srcdoc` et n'est pas assouplie pour l'occasion.
- [ ] Boucles infinies ou widget lourd : l'iframe ne bloque pas le fil principal de la page parente, mais prévoir un nombre maximum de widgets par page et le chargement paresseux.
- [ ] Journaux : ne pas journaliser le HTML complet, seulement taille et motifs refusés.

### 6. Tests
- [ ] Backend : `tests/test_course_widgets.py` sur le modèle de `tests/test_course_graphs.py` — insertion dans « Comment », limite de `course_widgets_max`, widget invalide écarté sans emporter les autres, mode dégradé, réglage désactivé, Gemini en erreur sans exception, motifs interdits (un cas par motif), `block_text` d'un widget.
- [ ] Mettre à jour `tests/test_course_generator.py` et `tests/test_course_plan_generator.py` (appel `attach_widgets` à la place de `attach_graphs`, nombre d'appels Gemini inchangé).
- [ ] Frontend : tests Vitest de `buildWidgetDocument` (CSP présente en tête, thème injecté, document du modèle encapsulé), de `course.schema.ts` (widget valide et invalide), de `BlockRenderer` (widget vide ne rend rien, ancien `graph` toujours rendu).
- [ ] Manuel : un widget qui tente `fetch`, `window.parent.document` et `localStorage` doit échouer, dans Chrome et Firefox ; rendu en thème sombre et sur mobile.

### 7. Déploiement et documentation
- [ ] Déployer d'abord le frontend (il sait rendre `widget` et ignore un type inconnu), puis le backend avec `COURSE_WIDGETS_ENABLED`.
- [ ] Mettre à jour `docs/Synthoria-LLM.postman_collection.json` (exemple de réponse avec un bloc `widget`) et le README du frontend si le contrat y figure.
- [ ] Suivre sur une semaine le taux de widgets insérés ou écartés (logs) ; si satisfaisant, supprimer ensuite `attach_graphs`, `CourseGraphsSchema`, `ApiGraph` et `GraphBlock` dans un changement séparé, après décision sur les anciens cours.

## Points de vigilance sécurité
- Le seul vrai rempart est la sandbox sans `allow-same-origin` plus la CSP ; le filtre serveur est une barrière de plus, pas la protection.
- Ne jamais ajouter `allow-same-origin` avec `allow-scripts` : le widget pourrait retirer sa propre sandbox.
- Message `postMessage` : contrôler `event.source`, ne rien exécuter ni insérer de ce qui est reçu, seulement une hauteur bornée.

## Tests à prévoir
- Fiabilité du JSON contenant du HTML avec Flash-Lite (étape 1) : c'est le principal risque non vérifié.
- Compatibilité des cours déjà enregistrés (blocs `graph`) et des consommateurs de texte de bloc (chatbot, podcast, flashcards).
- Nombre d'appels Gemini par cours inchangé (1 appel dédié, comme aujourd'hui).
