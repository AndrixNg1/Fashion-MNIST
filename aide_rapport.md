# Aide pour le rapport — CNN Fashion-MNIST

Ce document est un aide-mémoire pour préparer le rapport de quatre pages et la
défense. Il contient les résultats effectivement obtenus, les difficultés
rencontrées, leurs solutions, les limites du projet et les améliorations
possibles. Il ne remplace pas le rapport final : celui-ci doit être plus court
et sélectionner les éléments les plus importants.

## 1. Objectif du projet

L'objectif est d'implémenter un CNN personnalisé pour classer les dix classes
de Fashion-MNIST, une fois avec PyTorch et une fois avec TensorFlow/Keras. Les
deux implémentations utilisent une architecture et des hyperparamètres aussi
comparables que possible afin que la comparaison soit juste.

Les dix classes sont : `T-shirt/top`, `Trouser`, `Pullover`, `Dress`, `Coat`,
`Sandal`, `Shirt`, `Sneaker`, `Bag` et `Ankle boot`.

## 2. Données et protocole expérimental

- Fashion-MNIST contient 60 000 images d'entraînement et 10 000 images de test.
- Les images sont en niveaux de gris et mesurent 28 × 28 pixels.
- Les 60 000 images d'entraînement sont séparées de façon reproductible en :
  - 54 000 images pour l'entraînement ;
  - 6 000 images pour la validation.
- Les 10 000 images de test ne sont utilisées qu'à la fin pour l'évaluation
  finale.
- La graine aléatoire est fixée à 42.
- Les pixels sont normalisés dans l'intervalle [-1, 1].
- L'entraînement utilise Adam, un taux d'apprentissage de 0,001, une taille de
  lot de 64 et 10 époques.

Cette séparation entraînement/validation/test évite de choisir le modèle à
partir des données de test, ce qui limiterait la validité des résultats.

## 3. Architecture utilisée

Les deux frameworks utilisent le même principe :

```text
Image 28 × 28 × 1
→ Convolution 32 filtres, noyau 3 × 3 + ReLU
→ Max-pooling 2 × 2
→ Convolution 64 filtres, noyau 3 × 3 + ReLU
→ Max-pooling 2 × 2
→ Flatten (3 136 valeurs)
→ Dense 128 + ReLU
→ Dropout 0,5
→ Dense 10 logits
```

Les convolutions détectent progressivement des motifs simples puis plus
complexes. Le max-pooling réduit la taille spatiale et le coût de calcul. Le
dropout désactive aléatoirement une partie des neurones pendant l'entraînement,
ce qui aide à limiter le surapprentissage.

## 4. Résultats expérimentaux réels

| Framework | Accuracy sur le test | Macro-F1 sur le test |
|---|---:|---:|
| PyTorch | 91,62 % | 91,66 % |
| TensorFlow/Keras | 91,82 % | 91,77 % |

TensorFlow est légèrement meilleur : +0,20 point d'accuracy et +0,11 point de
macro-F1. Cet écart est faible ; on peut donc conclure que les deux
implémentations produisent des performances très proches lorsque l'architecture
et les paramètres sont comparables.

Le meilleur checkpoint est sélectionné avec la plus faible `validation loss`,
et non avec les performances du test :

| Framework | Époque sélectionnée | Validation loss |
|---|---:|---:|
| PyTorch | 9 | 0,2148 |
| TensorFlow/Keras | 9 | 0,2072 |

À l'époque 10, la loss de validation augmente légèrement dans les deux cas. Ce
comportement indique que continuer l'entraînement ne garantit pas une meilleure
généralisation ; conserver le meilleur checkpoint est donc justifié.

Les fichiers générés dans `results/` constituent les preuves à insérer ou à
commenter dans le rapport : courbes loss/accuracy, matrices de confusion,
métriques JSON, comparaison des frameworks et exemples de prédictions.

## 5. Justification des métriques

- **Accuracy** : proportion totale d'images bien classées. C'est une mesure
  facile à interpréter pour résumer la performance globale.
- **Precision** : parmi les images prédites dans une classe, proportion qui
  appartient réellement à cette classe.
- **Recall** : parmi les images réellement présentes dans une classe,
  proportion retrouvée par le modèle.
- **F1-score** : moyenne harmonique de precision et recall ; elle pénalise un
  modèle qui serait bon dans une mesure mais mauvais dans l'autre.
- **Macro-F1** : moyenne des F1 des dix classes, en donnant le même poids à
  chaque classe. Elle complète l'accuracy et vérifie qu'une bonne valeur globale
  ne masque pas de mauvaises performances sur une catégorie.
- **Matrice de confusion** : elle permet de visualiser les classes confondues
  (par exemple des vêtements visuellement proches).

## 6. Tests et robustesse

Les tests unitaires ont été exécutés avec succès : **9 tests réussis**.

Ils vérifient notamment :

- la séparation reproductible et sans recouvrement des données ;
- la normalisation dans [-1, 1] ;
- les dix étiquettes de classes ;
- les dimensions de sortie des CNN ;
- la sauvegarde et le rechargement des checkpoints ;
- le calcul des métriques.

## 7. Problèmes rencontrés et solutions

| Problème constaté | Cause | Solution appliquée |
|---|---|---|
| L'environnement `ia.labo2.0` était introuvable. | Il n'existait pas sur la machine. | Création d'un environnement Conda `fashion-cnn` avec Python 3.12 puis installation des dépendances. |
| PyTorch échouait avec `GET was unable to find an engine`. | La Quadro P620 possède une architecture CUDA `sm_61`, non prise en charge par la version PyTorch/CUDA installée. | Exécution sur CPU avec `CUDA_VISIBLE_DEVICES=""`. Les résultats restent valides. |
| TensorFlow affichait des avertissements CUDA. | TensorFlow ne pouvait pas utiliser ce GPU. | Exécution sur CPU ; les avertissements ne bloquent pas l'entraînement ni l'inférence. |
| L'interface Streamlit ne pouvait pas comparer les deux modèles. | PyTorch retentait d'utiliser le GPU incompatible. | Lancer Streamlit en CPU : `CUDA_VISIBLE_DEVICES="" streamlit run app.py`. |
| Une photo de chaussure était prédite comme `Bag` avec une confiance élevée. | Une photo réelle est différente de Fashion-MNIST : fond clair, contraste, cadrage, décor et résolution différents. | Inverser les couleurs si l'objet est sombre sur fond clair ; vérifier l'aperçu 28 × 28, centrer l'objet et utiliser un fond uniforme. |

## 8. Limites à expliquer dans le rapport

1. **Domaine limité à Fashion-MNIST.** Les modèles sont évalués officiellement
   sur les images de test Fashion-MNIST, pas sur des photos Internet ou des
   photos prises avec un téléphone.
2. **Décalage de domaine pour les photos réelles.** Une image externe peut avoir
   une couleur, un arrière-plan, une orientation, une ombre ou un cadrage très
   différent. La prédiction peut alors être erronée, même avec une grande
   confiance.
3. **Résolution très faible.** L'entrée 28 × 28 perd beaucoup de détails d'une
   photographie originale.
4. **Dix classes seulement.** Le modèle est obligé de choisir l'une des dix
   classes, même si l'image ne contient aucun article Fashion-MNIST. Une
   confiance élevée ne signifie donc pas toujours que la prédiction est fiable
   hors du domaine appris.
5. **Comparaison matérielle limitée.** Les deux expériences ont été exécutées
   sur CPU à cause de l'incompatibilité de la Quadro P620 avec la version CUDA
   installée. Les mesures de temps ne permettent donc pas de comparer les
   performances GPU des frameworks.
6. **Architecture volontairement simple.** Deux blocs convolutifs suffisent
   pour l'exercice, mais ne représentent pas nécessairement l'architecture la
   plus performante possible.

## 9. Interface Streamlit : intérêt et limites

L'interface est un bonus utile. Elle permet d'importer une image, de voir la
version réellement envoyée au CNN (28 × 28), d'inverser les couleurs, de
comparer les prédictions PyTorch et TensorFlow, puis d'afficher les
probabilités. Elle affiche aussi des cartes d'activation et Grad-CAM pour
interpréter les zones importantes pour PyTorch.

Pour une démonstration fiable :

- utiliser une image contenant un seul article ;
- centrer l'article et supprimer autant que possible le décor ;
- privilégier un fond noir avec un article clair ;
- si le fond est clair et l'article sombre, activer l'inversion ;
- regarder l'aperçu 28 × 28 avant d'interpréter la prédiction.

La bonne commande de lancement sur cette machine est :

```bash
CUDA_VISIBLE_DEVICES="" streamlit run app.py
```

### Formats de fichiers acceptés par l'interface

L'interface accepte les formats suivants : **PNG** (`.png`), **JPEG/JPG**
(`.jpg`, `.jpeg`), **JFIF** (`.jfif`), **WebP** (`.webp`), **BMP** (`.bmp`),
**GIF** (`.gif`), **TIFF** (`.tif`, `.tiff`), **ICO** (`.ico`), **PPM**
(`.ppm`) et **PGM** (`.pgm`).

Le format de fichier n'améliore pas, à lui seul, une prédiction. Pour les
démonstrations, préférer **PNG** ou **JPG/JPEG** : ils sont courants et
fonctionnent très bien. Un PNG est intéressant si l'image possède un fond déjà
net ; un JPG convient pour une photographie simple.

### Type d'image à rechercher pour une bonne prédiction

Le CNN ne voit pas l'image originale : il reçoit une image en niveaux de gris
de seulement 28 × 28 pixels. Il faut donc rechercher ou préparer des images
visuellement proches de Fashion-MNIST :

- un seul article de mode dans l'image ;
- article vu de face ou de profil simple, sans rotation forte ;
- article grand et **centré** dans le cadre ;
- fond uniforme et peu chargé, idéalement noir ;
- article clair sur fond noir, comme dans Fashion-MNIST ;
- image nette, sans main, personne, texte, logo dominant, ombre marquée ou
  plusieurs objets ;
- pour une chaussure, un sac ou un vêtement : préférer une photo de produit
  isolé (« product photo », « isolated on black background ») plutôt qu'une
  photo portée dans la rue.

Requêtes de recherche utiles :

```text
white sneaker isolated on black background
handbag isolated on black background
black trousers isolated on white background
plain t-shirt product photo black background
```

Dans le troisième exemple, l'article est sombre sur un fond clair : il faudra
activer **« Inverser les couleurs »** dans l'interface. Après chaque import,
il faut contrôler l'image affichée sous **« Entrée du CNN (28 × 28) »**. Si
l'objet est presque invisible, trop petit ou si le fond occupe tout le carré,
la prédiction ne sera pas fiable.

### Solutions pratiques pour les images externes

| Situation observée | Solution à appliquer |
|---|---|
| Article sombre sur fond clair | Activer **« Inverser les couleurs »**. Cela crée un fond sombre et un article clair, plus proche de Fashion-MNIST. |
| Article clair sur fond sombre | Laisser l'inversion désactivée. |
| Article trop petit ou avec beaucoup de décor | Recadrer l'image autour de l'article avant l'import, puis vérifier l'aperçu 28 × 28. |
| Plusieurs vêtements, personne ou main dans la photo | Choisir une image avec un seul article isolé. |
| Prédiction inattendue avec une confiance très élevée | Ne pas interpréter cette confiance comme une preuve : le modèle est obligé de choisir une des dix classes. Essayer une image plus proche du dataset ou une autre orientation. |
| PyTorch affiche `GET was unable to find an engine` dans Streamlit | Relancer l'interface avec `CUDA_VISIBLE_DEVICES="" streamlit run app.py` afin de forcer le CPU. |
| Les deux modèles ne sont pas comparables dans l'interface | Vérifier que `pytorch_best.pt` et `tensorflow_best.keras` sont présents dans `results/`, puis lancer Streamlit avec la commande CPU ci-dessus. |

Pour la défense, il faut préciser que ces manipulations concernent uniquement
le **bonus d'inférence sur photo externe**. Les résultats officiels restent les
métriques obtenues sur le jeu de test Fashion-MNIST, dont le format est déjà
conforme aux données d'entraînement.

## 10. Améliorations possibles

Ces propositions sont des bonus ; elles ne sont pas nécessaires pour satisfaire
le sujet 9.1.

1. Ajouter un **MLP** comme référence afin de quantifier l'apport des
   convolutions par rapport à un réseau dense simple.
2. Essayer un **deuxième CNN** avec batch normalization, davantage de filtres,
   ou un dropout additionnel, puis comparer accuracy, macro-F1 et temps.
3. Ajouter de la **data augmentation** : petites translations, rotations et
   zooms. Cela pourrait améliorer la robustesse aux photos externes.
4. Améliorer le prétraitement de l'interface : détection/cadrage de l'objet,
   suppression du fond, centrage et ajustement automatique du contraste.
5. Utiliser un modèle pré-entraîné adapté à des images plus réalistes, tout en
   précisant que ce serait un bonus différent du CNN personnalisé demandé.
6. Ajouter une estimation d'incertitude ou un seuil de confiance afin de dire
   « image hors domaine » plutôt que de forcer une classe lorsque la photo ne
   ressemble pas au dataset.

## 11. Plan conseillé pour le rapport final (4 pages maximum)

1. **Introduction et données** : objectif, Fashion-MNIST, classes et séparation
   entraînement/validation/test.
2. **Méthode** : architecture, normalisation, hyperparamètres et choix de la
   loss/optimiseur.
3. **Évaluation** : justification des métriques, tableau des deux résultats,
   courbes et une matrice de confusion pertinente.
4. **Discussion et conclusion** : comparaison PyTorch/TensorFlow, limites,
   complexité, interface Streamlit comme bonus et améliorations futures.

Pour la complexité, écrire que le coût des convolutions dépend principalement
de la hauteur, de la largeur, du nombre de canaux d'entrée, du nombre de
filtres et de la taille des noyaux. Le max-pooling réduit les dimensions
spatiales, ce qui diminue le coût des couches suivantes.
