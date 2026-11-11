# Cahier des charges — Synthoria : cours interactifs et exercices corrigés

## 1. Contexte
Synthoria génère des cours à partir d'une question ou de documents : leçon découpée en sections (Pourquoi, Quoi, Comment), QCM, podcast, vidéos YouTube avec notes, « Explique avec tes mots » et exercice à trous « À toi ». Cette évolution reprend la forme d'un cours bien construit : explications simples, démonstration pas à pas, points à retenir, pièges à éviter, puis exercices corrigés. Le quota de l'IA est limité : la génération doit rester économe.

## 2. Objectifs
- Rendre chaque section plus pédagogique : résumés clairs, exemple ou tableau, démonstration interactive pas à pas, exercice « À toi », bilan en fin de section.
- Ajouter une partie « Exercices corrigés », distincte de la leçon et du quiz, où l'apprenant s'entraîne et reçoit une correction.
- Enrichir le quiz avec des réponses numériques et des questions d'appariement.
- Naviguer facilement entre leçons, exercices corrigés et quiz.
- Ne générer les exercices que si l'apprenant le souhaite, pour ménager le quota.

## 3. Acteurs / Rôles
- **Apprenant** — suit les cours, répond aux quiz, prend des notes, s'entraîne sur les exercices corrigés. Il n'y a pas encore de comptes : aucun suivi de points.

## 4. Périmètre
- Nouveau format de section dans la leçon (résumés, exemple ou tableau, démonstration interactive, « À toi », fin de section, « Ce qu'il faut retenir », « Ce qu'il faut éviter »).
- Vidéos YouTube rattachées à chaque section, avec notes.
- Nouveaux types de questions dans le quiz : réponse numérique et appariement.
- Navigation à trois onglets sous le podcast : Leçons, Exercices corrigés, Quiz.
- Partie « Exercices corrigés » : génération à la demande, énoncé riche, minuteur, indices, correction de la réponse, lien avec les sections.

## 5. Règles métier
- **RG-1** — Les exercices corrigés ne donnent aucun point. L'attribution de points viendra avec les comptes utilisateurs.
- **RG-2** — Une question à réponse numérique est juste selon la nature de la bonne réponse. Si elle est entière, la valeur saisie doit lui être strictement égale. Si elle est décimale, la valeur saisie est juste à 0,0001 près. Dans les deux cas, hors de cette limite, la question ne rapporte aucun point.
- **RG-3** — Dans une question d'appariement, il y a autant d'options que de sujets. Une option déjà choisie pour un sujet n'est plus proposée aux autres. Choisir la valeur vide pour un sujet remet son option à disposition.
- **RG-4** — Les exercices corrigés se génèrent après la leçon et le quiz, jamais en même temps.
- **RG-5** — Les exercices ne sont générés que si l'apprenant le demande.
- **RG-6** — Seules les sections qui en ont vraiment besoin reçoivent un exercice : toutes les sections n'en ont pas.
- **RG-7** — Le bouton qui mène d'une section à son exercice n'apparaît que si cet exercice est rédigé.
- **RG-8** — Avant de commencer un exercice, l'apprenant ne voit pas l'énoncé, seulement ce que l'exercice couvre.
- **RG-9** — La génération reste économe : un seul lot d'exercices par cours, pas un exercice par section.

## 6. Fonctionnalités

### V1 — Nouveau format de leçon et quiz enrichi

#### Epic : Structure d'une section

**User story 1** — En tant qu'apprenant, je veux lire dans chaque section des parties Pourquoi, Quoi et Comment avec des résumés simples et bien expliqués, afin de comprendre avant d'approfondir.

Critères d'acceptation :
- chaque section garde les trois parties Pourquoi, Quoi, Comment ;
- chaque partie est lisible seule et reste courte ;
- les notions importantes sont expliquées, pas seulement citées.

**User story 2** — En tant qu'apprenant, je veux voir après les explications un tableau ou un exemple, afin de relier la théorie à un cas concret.

Critères d'acceptation :
- la section présente un tableau ou un exemple travaillé quand il aide à comprendre ;
- l'exemple est complet, avec ses données et son résultat.

**User story 3** — En tant qu'apprenant, je veux une démonstration interactive étape par étape (graphe, courbe ou autre visuel), afin de voir comment la méthode se déroule.

Critères d'acceptation :
- chaque étape est expliquée en une phrase ;
- je peux avancer et reculer d'une étape ;
- si une section n'a pas de visuel pertinent, elle n'en affiche pas.

**User story 4** — En tant qu'apprenant, je veux faire un petit exercice « À toi » dans la section, afin de m'exercer tout de suite.

Critères d'acceptation :
- l'exercice « À toi » existant est conservé tel quel dans le nouveau format.

**User story 5** — En tant qu'apprenant, je veux terminer chaque section par un mini-QCM ou par « Explique avec tes mots », afin de vérifier ce que j'ai retenu.

Critères d'acceptation :
- chaque section se termine par l'un des deux ;
- « Explique avec tes mots » garde son fonctionnement actuel.

**User story 6** — En tant qu'apprenant, je veux trouver à la fin de chaque section « Ce qu'il faut retenir » et « Ce qu'il faut éviter », afin de fixer l'essentiel et de repérer les pièges.

Critères d'acceptation :
- les deux encadrés apparaissent à la fin de chaque section ;
- « Ce qu'il faut retenir » liste les points clés ;
- « Ce qu'il faut éviter » décrit les erreurs fréquentes et comment les éviter.

#### Epic : Vidéos par section

**User story 7** — En tant qu'apprenant, je veux des vidéos YouTube pour aller plus loin à la fin de chaque section, afin d'approfondir le sujet que je viens de lire.

Critères d'acceptation :
- chaque section propose ses vidéos ;
- je peux ajouter, modifier et supprimer une note sur une vidéo, comme aujourd'hui ;
- une section sans vidéo trouvée n'affiche pas d'espace vide.

#### Epic : Quiz enrichi

**User story 8** — En tant qu'apprenant, je veux répondre à certaines questions par un nombre, afin d'être testé sur des calculs.

Critères d'acceptation :
- la question propose un champ pour saisir un nombre ;
- une bonne réponse entière n'accepte que la valeur exacte, une bonne réponse décimale est juste à 0,0001 près (RG-2), sinon aucun point ;
- la correction affiche la bonne valeur.

**User story 9** — En tant qu'apprenant, je veux des questions d'appariement où chaque sujet a une liste déroulante, afin d'associer chaque sujet à l'option qui lui convient.

Critères d'acceptation :
- il y a autant d'options que de sujets (RG-3) ;
- une option déjà choisie disparaît des autres listes ;
- choisir la valeur vide remet l'option à disposition ;
- un sujet laissé vide compte comme une mauvaise réponse ;
- la correction montre l'association attendue pour chaque sujet.

#### Epic : Navigation

**User story 10** — En tant qu'apprenant, je veux une navigation à trois onglets sous le podcast (Leçons, Exercices corrigés, Quiz), afin de passer d'une partie à l'autre sans chercher.

Critères d'acceptation :
- les trois onglets sont visibles sous le podcast ;
- l'onglet actif est signalé ;
- l'onglet Exercices corrigés est accessible même quand aucun exercice n'est encore généré (voir V2).

### V2 — Exercices corrigés

#### Epic : Génération des exercices

**User story 11** — En tant qu'apprenant, je veux qu'à la fin de la génération d'un cours on me demande si je veux générer les exercices corrigés, afin de décider si j'en ai besoin.

Critères d'acceptation :
- une fenêtre apparaît une fois la leçon et le quiz prêts (RG-4) ;
- si je réponds oui, la génération se lance en arrière-plan ;
- si je réponds non, rien n'est généré (RG-5).

**User story 12** — En tant qu'apprenant, je veux un bouton « Générer les exercices » dans l'onglet Exercices corrigés quand il n'y en a pas encore, afin de les demander plus tard.

Critères d'acceptation :
- le bouton apparaît seulement si le cours n'a pas d'exercices ;
- il génère l'ensemble des exercices du cours en une fois, pas section par section (RG-9) ;
- seules les sections qui en ont besoin reçoivent un exercice (RG-6) ;
- l'avancement de la génération est visible.

#### Epic : Faire un exercice

**User story 13** — En tant qu'apprenant, je veux voir la liste des exercices du cours avec ce que chacun couvre, afin de choisir celui que je veux faire.

Critères d'acceptation :
- la liste montre le titre de l'exercice, la section liée et ce qu'il couvre ;
- l'énoncé n'est pas visible avant de commencer (RG-8).

**User story 14** — En tant qu'apprenant, je veux un minuteur et un bouton « Commencer », afin de m'entraîner dans des conditions de temps.

Critères d'acceptation :
- l'énoncé s'affiche seulement après avoir cliqué sur « Commencer » ;
- le minuteur démarre au même moment et reste visible ;
- le temps écoulé est visible pendant l'exercice.

**User story 15** — En tant qu'apprenant, je veux des énoncés qui contiennent des graphes, des tableaux ou des courbes, afin de traiter des cas réels (plus court chemin, équilibre de Nash, etc.).

Critères d'acceptation :
- un énoncé peut mêler texte, tableau, graphe et courbe ;
- ces éléments sont lisibles sur mobile.

**User story 16** — En tant qu'apprenant, je veux demander des indices pendant l'exercice, afin d'être débloqué sans voir la solution complète.

Critères d'acceptation :
- les indices sont masqués par défaut ;
- je les affiche un par un, du plus général au plus précis.

**User story 17** — En tant qu'apprenant, je veux saisir ma réponse dans un champ LaTeX et recevoir une correction, afin de savoir où je me suis trompé.

Critères d'acceptation :
- le champ accepte le texte et les formules LaTeX ;
- ma réponse est corrigée par l'IA, comme dans « Explique avec tes mots » ;
- la correction indique ce qui est juste, ce qui manque ou est faux, et la démarche attendue ;
- aucun point n'est attribué (RG-1) ;
- en cas d'échec de la correction, un message m'invite à réessayer sans perdre ma réponse.

#### Epic : Lien avec les sections

**User story 18** — En tant qu'apprenant, je veux un bouton dans une section qui mène à son exercice corrigé, afin de m'exercer juste après avoir lu.

Critères d'acceptation :
- le bouton apparaît seulement si l'exercice de cette section est rédigé (RG-7) ;
- une section sans exercice n'affiche pas de bouton ;
- depuis l'exercice, je peux revenir à sa section.

## 7. Hors périmètre
- Attribution de points ou classement pour les exercices (prévu après l'ajout des comptes utilisateurs).
- Exercice généré section par section à la demande.
- Génération automatique des exercices sans accord de l'apprenant.

## 8. Points ouverts / à confirmer
- Durée du minuteur : fixée par exercice ou choisie par l'apprenant ? Que se passe-t-il à la fin du temps (simple indicateur ou blocage de la saisie) ?
- Nombre d'exercices visé par cours et critère pour qu'une section « en ait besoin ».
- Démonstration interactive : une par section quand c'est pertinent, ou limitée à quelques sections du cours ?
- Choix entre mini-QCM et « Explique avec tes mots » en fin de section : au choix de l'IA, ou un seul des deux par défaut ?
- Nombre d'indices par exercice et possibilité de refaire un exercice après correction.
- Les cours déjà générés : gardent-ils leur ancien format, ou proposer de les régénérer ?
