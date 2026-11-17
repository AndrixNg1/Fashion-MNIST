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

L'interface est un bonus utile. Cette variante dédiée aux photos réelles
détecte automatiquement la zone la plus contrastée, conserve son plus grand
objet, la recadre, la centre sur un fond noir et construit l'entrée 28 × 28
envoyée au CNN. Elle compare les prédictions PyTorch et TensorFlow, affiche les
probabilités et répond **« Je ne sais pas »** si l'objet ne peut pas être isolé,
si les modèles ne sont pas d'accord ou si l'un d'eux a une confiance inférieure
à 75 %. Elle affiche aussi des cartes d'activation et Grad-CAM pour interpréter
les zones importantes pour PyTorch.

### Fonctionnement détaillé du prétraitement automatique

Le prétraitement ne modifie pas les CNN entraînés ni leurs résultats officiels
sur Fashion-MNIST. Il transforme uniquement une **photo externe** afin qu'elle
ressemble davantage à l'entrée que les CNN connaissent. Son déroulement est le
suivant :

1. **Passage en niveaux de gris.** La photo couleur est convertie en une seule
   couche de luminosité. C'est cohérent avec Fashion-MNIST, qui ne contient pas
   de couleur.
2. **Réduction de travail.** Pour ne pas ralentir l'interface avec une très
   grande photo, l'image est limitée à 512 × 512 pixels au maximum pendant la
   détection. L'image finale fournie au CNN fera ensuite toujours 28 × 28.
3. **Estimation du fond.** Les pixels présents sur les quatre bords de l'image
   sont considérés comme une estimation du fond. L'interface calcule leur
   luminosité médiane. La médiane est plus robuste qu'une moyenne lorsqu'un bord
   contient quelques pixels anormaux.
4. **Carte de contraste.** Pour chaque pixel, l'interface mesure la différence
   absolue entre sa luminosité et celle du fond estimé. Un objet sombre sur un
   fond clair et un objet clair sur un fond sombre donnent tous deux un contraste
   élevé. Cette étape remplace l'ancienne inversion manuelle des couleurs.
5. **Seuil et région principale.** Les pixels dont le contraste est trop faible
   sont considérés comme du fond. Le seuil est au minimum de 18 niveaux de gris
   et augmente si les bords de l'image sont bruités. Parmi les régions restantes,
   seule la plus grande région connectée est conservée : cela aide à ignorer de
   petits détails isolés, par exemple un logo ou une poussière.
6. **Vérification de qualité.** L'interface refuse l'image avant de lui faire
   confiance si la zone détectée représente moins de 1 % de l'image, si son
   contraste est trop faible, ou si elle représente plus de 70 % de l'image.
   Ces cas correspondent souvent à une image vide, à un article impossible à
   séparer du fond, ou à une scène avec beaucoup de décor.
7. **Recadrage avec marge.** Une boîte englobante est placée autour de l'objet
   détecté, avec une marge d'environ 12 % (au moins 4 pixels). La marge évite de
   couper les contours d'une chaussure, d'un sac ou d'un vêtement.
8. **Création d'un objet clair sur fond noir.** Les valeurs de contraste dans la
   boîte sont mises à l'échelle : les pixels de l'objet deviennent lumineux et
   le fond devient noir. Ainsi, un pantalon noir sur fond blanc et un t-shirt
   blanc sur fond noir sont convertis vers la même convention visuelle que
   Fashion-MNIST.
9. **Centrage et redimensionnement.** L'objet garde ses proportions, est réduit
   pour tenir dans une zone maximale de 22 × 22 pixels, puis est collé au centre
   d'un canevas noir de 28 × 28 pixels. Laisser une bordure noire est utile car
   les articles Fashion-MNIST ne touchent généralement pas les bords.
10. **Normalisation et format pour les modèles.** Les valeurs de 0 à 255 sont
    converties dans l'intervalle [-1, 1], exactement comme pendant
    l'entraînement. TensorFlow reçoit le tableau sous la forme `(1, 28, 28, 1)`
    et PyTorch reçoit le même tableau réorganisé en `(1, 1, 28, 28)`.

L'image affichée dans l'interface sous **« Entrée automatique du CNN (28 × 28) »**
est donc l'image exacte donnée aux deux CNN. C'est elle qu'il faut examiner
avant d'interpréter une prédiction.

### Décision « Je ne sais pas »

Fashion-MNIST est un problème à classes fermées : un CNN standard doit toujours
choisir une des dix classes, même pour une voiture, un visage ou une image vide.
L'interface ajoute donc une règle de rejet prudente. Elle répond **« Je ne sais
pas »** lorsqu'au moins une condition est vraie :

- l'objet principal n'a pas pu être isolé par les contrôles de qualité ;
- PyTorch et TensorFlow prédisent deux classes différentes ;
- l'une des deux confiances maximales est inférieure à 75 % ;
- l'un des deux checkpoints est absent ou indisponible.

Cette règle réduit les réponses arbitraires, mais elle n'est pas un détecteur
d'anomalies parfait : une photo très différente peut encore recevoir une forte
confiance de deux CNN similaires. Une solution plus robuste demanderait des
photos « hors classe » et des photos réelles pendant l'entraînement, ou un
modèle spécialisé de détection hors domaine.

Pour une démonstration fiable :

- utiliser une image contenant un seul article ;
- centrer l'article et supprimer autant que possible le décor ;
- privilégier un fond uniforme, car le prétraitement estime le fond à partir
  des bords de l'image ;
- regarder l'aperçu 28 × 28 avant d'interpréter la prédiction.

La bonne commande de lancement sur cette machine est :

```bash
streamlit run app.py
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

Dans le troisième exemple, le prétraitement automatique transforme le contraste
pour produire un article clair sur fond noir. Après chaque import, il faut
contrôler l'image affichée sous **« Entrée automatique du CNN (28 × 28) »**. Si
l'objet est presque invisible, trop petit ou si le fond occupe tout le carré,
la prédiction ne sera pas fiable.

### Solutions pratiques pour les images externes

| Situation observée | Solution à appliquer |
|---|---|
| Article sombre sur fond clair | Le prétraitement automatique utilise le contraste avec le fond et construit un article clair sur fond noir. |
| Article clair sur fond sombre | Le même prétraitement est appliqué automatiquement. |
| Article trop petit ou avec beaucoup de décor | Recadrer l'image autour de l'article avant l'import, puis vérifier l'aperçu 28 × 28. |
| Plusieurs vêtements, personne ou main dans la photo | Choisir une image avec un seul article isolé. |
| Prédiction inattendue avec une confiance très élevée | Ne pas interpréter cette confiance comme une preuve : le modèle est obligé de choisir une des dix classes. Essayer une image plus proche du dataset ou une autre orientation. |
| PyTorch affichait `GET was unable to find an engine` dans Streamlit | Cette variante force PyTorch à utiliser le CPU, donc elle évite l'incompatibilité CUDA de la Quadro P620. |
| L'interface répond « Je ne sais pas » | Importer une image avec un seul article net et contrasté ; si cela persiste, l'image est probablement trop différente de Fashion-MNIST. |
| Les deux modèles ne sont pas comparables dans l'interface | Vérifier que `pytorch_best.pt` et `tensorflow_best.keras` sont présents dans `results/`. |

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
