# Commandes du projet Fashion-MNIST

Deux interfaces sont disponibles :

- **version originale** : comparaison directe avec une image déjà proche de
  Fashion-MNIST ;
- **version photos réelles** (`photos-reelles/`) : recadrage, centrage et
  prétraitement automatiques, avec une réponse « Je ne sais pas » pour les
  images ambiguës.

Exécuter les commandes suivantes depuis un terminal Linux :

```bash
cd /home/veldra/Bureau/exam-IA/Fashion-MNIST
```

## 1. Activer l'environnement Python

```bash
conda activate fashion-cnn
```

## 2. Installer les dépendances

```bash
python -m pip install -r requirements.txt
```

Pour installer uniquement Streamlit :

```bash
python -m pip install streamlit
```

## 3. Entraîner les deux modèles

Cette commande crée les checkpoints PyTorch et TensorFlow nécessaires à
l'interface dans le dossier `results/` :

```bash
python main.py --framework both --mode train --epochs 10 --batch-size 64
```

Il est également possible de les entraîner séparément :

```bash
python main.py --framework pytorch --mode train
python main.py --framework tensorflow --mode train
```

## 4. Lancer l'interface originale

La carte graphique Quadro P620 n'est pas compatible avec la version PyTorch
installée. Il faut donc désactiver le GPU pour cette version :

```bash
CUDA_VISIBLE_DEVICES="" streamlit run app.py
```

## 5. Lancer l'interface pour photos réelles

Cette version applique automatiquement la détection de l'objet principal, le
recadrage, le centrage et l'adaptation au format Fashion-MNIST. Elle force
PyTorch à utiliser le CPU elle-même ; il ne faut pas ajouter
`CUDA_VISIBLE_DEVICES=""`.

```bash
cd /home/veldra/Bureau/exam-IA/Fashion-MNIST/photos-reelles
```

```bash
streamlit run app.py
```

L'interface est normalement disponible à l'adresse suivante :

```text
http://localhost:8501
```

Elle permet d'importer des images, notamment au format JFIF, de comparer les
prédictions PyTorch et TensorFlow, d'afficher les probabilités des dix classes
et de vérifier si les deux modèles sont d'accord. Elle affiche également les
activations des convolutions et du max pooling ainsi qu'une explication Grad-CAM
de la prédiction PyTorch, sans commande supplémentaire.

Pour arrêter l'une ou l'autre interface, revenir dans le terminal concerné puis
utiliser :

```text
Ctrl+C
```

## 6. Lancer les tests

```bash
python -m pytest
```
