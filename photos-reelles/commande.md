# Commandes du projet Fashion-MNIST

Exécuter les commandes suivantes depuis le dossier du projet :

```powershell
cd C:\Users\Dell\Videos\Fashion-MNIST
```

## 1. Activer l'environnement Python

```powershell
conda activate ia.labo2.0
```

## 2. Installer les dépendances

```powershell
python -m pip install -r requirements.txt
```

Pour installer uniquement Streamlit :

```powershell
python -m pip install streamlit
```

## 3. Entraîner les deux modèles

Cette commande crée les checkpoints PyTorch et TensorFlow nécessaires à
l'interface dans le dossier `results/` :

```powershell
python main.py --framework both --mode train --epochs 10 --batch-size 64
```

Il est également possible de les entraîner séparément :

```powershell
python main.py --framework pytorch --mode train
python main.py --framework tensorflow --mode train
```

## 4. Lancer l'interface graphique

```powershell
python -m streamlit run app.py
```

L'interface est normalement disponible à l'adresse suivante :

```text
http://localhost:8501
```

Elle permet d'importer des images, notamment au format JFIF, de comparer les
prédictions PyTorch et TensorFlow, d'afficher les probabilités des dix classes
et de vérifier si les deux modèles sont d'accord. Elle affiche également les
activations des convolutions et du max pooling ainsi qu'une explication Grad-CAM
de la prédiction PyTorch, sans commande supplémentaire. Dans cette variante,
l'image est aussi recadrée et centrée automatiquement avant sa réduction à
28 × 28. Si l'article n'est pas suffisamment isolé, si les modèles ne sont pas
d'accord ou si la confiance est faible, l'interface répond **« Je ne sais pas »**.
PyTorch utilise volontairement le CPU dans l'interface afin d'éviter
l'incompatibilité de la Quadro P620 avec CUDA.

## 5. Lancer les tests

```powershell
python -m pytest
```

## 6. Arrêter Streamlit

Dans le terminal où Streamlit est lancé, utiliser :

```text
Ctrl+C
```
