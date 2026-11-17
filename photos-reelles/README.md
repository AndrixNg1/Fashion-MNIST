# Fashion-MNIST — CNN personnalisé

Projet d’examen d’Intelligence Artificielle (sujet 9.1) : comparaison reproductible de deux CNN personnalisés, implémentés avec **PyTorch** et **TensorFlow/Keras**, pour classifier Fashion-MNIST. Aucun modèle pré-entraîné n’est utilisé.

## Objectif et méthode

Fashion-MNIST comprend 60 000 images d’entraînement et 10 000 images de test, en niveaux de gris (28 × 28), réparties dans dix classes : T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag et Ankle boot.

La partition est fixe et sans fuite : 54 000 exemples servent à l’apprentissage, 6 000 à la validation et les 10 000 images de test restent réservées à l’évaluation finale. La graine est `42`. Les pixels sont convertis en `float32` puis normalisés dans `[-1, 1]`. PyTorch reçoit le format NCHW et TensorFlow le format NHWC ; la partition et les conditions d’entraînement sont identiques.

L’architecture des deux modèles est volontairement comparable et adaptée au CPU :

```text
28×28×1 → Conv(32, 3×3) + ReLU → MaxPool(2×2)
        → Conv(64, 3×3) + ReLU → MaxPool(2×2)
        → Flatten → Dense(128) + ReLU → Dropout(0.5) → Dense(10 logits)
```

L’optimiseur est Adam (`lr=0.001`), avec cross-entropy sur logits. Le meilleur modèle est choisi uniquement avec la `validation loss`, puis évalué une seule fois sur le test.

## Installation

Le projet cible Python 3.12 et fonctionne sur CPU, avec utilisation automatique de CUDA lorsque PyTorch le détecte.

```bash
conda activate ia.labo2.0
python -m pip install -r requirements.txt
# Alternative locale :
python -m pip install -e .
```

## Utilisation

```bash
# Télécharge le dataset si nécessaire et sauvegarde results/dataset_examples.png
python main.py --mode explore

# Entraîne, sélectionne le meilleur checkpoint de validation et évalue le test
python main.py --framework pytorch --mode train
python main.py --framework tensorflow --mode train

# Lance les deux expériences successivement
python main.py --framework both --mode train --epochs 10 --batch-size 64

# Réévalue un checkpoint existant, sans réentraîner
python main.py --framework pytorch --mode evaluate
python main.py --framework tensorflow --mode evaluate
```

### Interface de reconnaissance

Cette variante destinée aux photos réelles applique automatiquement un
prétraitement : elle isole la région la plus contrastée, la recadre, la centre
sur un fond noir et l'adapte à 28 × 28. L'interface permet ensuite de visualiser
l'entrée réellement envoyée aux CNN et de comparer côte à côte la classe, la
confiance, les dix probabilités et le temps d'inférence de PyTorch et
TensorFlow. Elle répond **« Je ne sais pas »** quand l'article ne peut pas être
isolé, que les modèles ne sont pas d'accord ou que leur confiance est trop
faible. Cette décision est prudente, mais ne remplace pas un vrai détecteur
d'images hors domaine entraîné sur des exemples réels.
Les formats PNG, JPG/JPEG/JFIF, WebP, BMP, GIF, TIFF, ICO, PPM et PGM sont acceptés.
L'interface présente aussi le parcours dimensionnel complet de l'image dans le
CNN, les cartes d'activation des deux convolutions et des deux max poolings,
ainsi qu'une carte Grad-CAM expliquant les zones qui influencent la prédiction
PyTorch.

Après avoir entraîné au moins un modèle (les checkpoints sont lus dans `results/`) :

```bash
streamlit run app.py
```

PyTorch est volontairement exécuté sur CPU dans cette interface : la Quadro
P620 de la machine de développement n'est pas compatible avec la version CUDA
installée. Les deux checkpoints restent nécessaires à la comparaison et à la
décision « Je ne sais pas ».

Les paramètres `--epochs`, `--batch-size`, `--learning-rate`, `--validation-size`, `--data-dir` et `--results-dir` sont modifiables. L’exploration nécessite TensorFlow/Keras, car il est utilisé pour télécharger et lire les données sous forme de tableaux.

## Résultats générés

Chaque entraînement produit dans `results/` un checkpoint (`pytorch_best.pt` ou `tensorflow_best.keras`), les courbes loss/accuracy, une matrice de confusion, des exemples de prédictions correctes et incorrectes, ainsi qu’un fichier JSON de métriques. Les fichiers de résultats sont ignorés par Git afin de ne pas confondre code et résultats expérimentaux.

Les métriques comprennent l’accuracy globale et les precision, recall et F1 macro/par classe. L’accuracy donne une synthèse intuitive, tandis que les métriques macro donnent le même poids à chaque catégorie et révèlent les confusions qui pourraient être masquées par une valeur globale. La matrice de confusion rend ces erreurs visibles. Aucun résultat chiffré n’est fourni avant qu’une expérience réelle soit lancée.

## Tests

```bash
pytest
```

Les tests vérifient la partition reproductible, la normalisation, les labels, les métriques et les dimensions de sortie des CNN. Les tests propres à PyTorch ou TensorFlow sont automatiquement ignorés si le framework concerné n’est pas installé ; dans l’environnement d’examen ils doivent être exécutés avec les deux bibliothèques disponibles.

## Structure

```text
src/fashion_cnn/
├── config.py                 # hyperparamètres centralisés
├── datasets/fashion_mnist.py # chargement, normalisation, séparation, aperçu
├── models/                   # CNN PyTorch et TensorFlow comparables
├── training/trainers.py      # entraînement, checkpoints, prédictions
├── evaluation/metrics.py     # métriques et figures scientifiques
└── utils/reproducibility.py  # graines et déterminisme
tests/                        # tests unitaires rapides
```

## État

L’implémentation, la CLI, les visualisations et les tests sont présents. Les valeurs expérimentales (temps, accuracy et comparaison de frameworks) restent volontairement à produire en exécutant les commandes d’entraînement dans l’environnement `ia.labo2.0` décrit dans le sujet. Elles formeront la base factuelle du rapport de quatre pages.
