# Fashion-MNIST — CNN Image Classification

A deep learning project focused on building, training, and evaluating custom Convolutional Neural Networks (CNNs) for image classification using **PyTorch** and **TensorFlow/Keras**.

This project is developed as part of the **Artificial Intelligence — Project 9.1** practical examination.

## Overview

The objective is to implement a custom CNN capable of classifying grayscale images of clothing items from the Fashion-MNIST dataset.

The project explores the fundamentals of computer vision, convolutional neural networks, model training, evaluation, and performance comparison between two deep learning frameworks.

## Technologies

- **Python 3.12**
- **PyTorch & TorchVision**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**
- **Pytest**

## Dataset

The [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) contains 70,000 grayscale images of fashion products across 10 categories.

| Property | Description |
|---|---|
| Training samples | 60,000 |
| Test samples | 10,000 |
| Image dimensions | 28 × 28 pixels |
| Image channels | 1 (grayscale) |
| Number of classes | 10 |

**Categories:** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot.

## Project Objectives

- Implement custom CNN architectures using PyTorch and TensorFlow.
- Use convolutional, max-pooling, and dropout layers.
- Train and evaluate both implementations on Fashion-MNIST.
- Visualize training and validation loss and accuracy.
- Evaluate model performance using appropriate classification metrics.
- Compare the results obtained with both frameworks.
- Ensure reproducibility using a fixed random seed of `42`.

## Project Structure

```text
Fashion-MNIST/
├── data/
├── src/
│   └── fashion_cnn/
│       ├── datasets/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       └── utils/
├── tests/
├── results/
├── main.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

The structure will evolve as the implementations are developed.

## Getting Started

### Clone the repository

```bash
git clone https://github.com/AndrixNg1/Fashion-MNIST.git
cd Fashion-MNIST
```

### Configure the environment

Create a Python environment or activate an existing compatible environment.

```bash
conda activate ia.labo2.0
```

The environment name above corresponds to the development setup. Other users should create their own environment with the required dependencies.

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Run the project

```bash
python main.py
```

The training and evaluation commands will be documented as the corresponding modules are implemented.

## Development Status

**Current phase: Project initialization**

Planned development stages:

- [x] Initialize project structure
- [x] Configure development environment
- [ ] Load and explore Fashion-MNIST
- [ ] Implement the CNN with PyTorch
- [ ] Implement the CNN with TensorFlow
- [ ] Train and evaluate both models
- [ ] Visualize training metrics
- [ ] Compare experimental results
- [ ] Finalize tests and experimental report

## Academic Context

**Course:** Artificial Intelligence  
**Project:** 9.1 — Custom CNN for Image Classification  
**Academic year:** 2026

## Author

**Andrix Ngoyi** — Software Engineer

[GitHub](https://github.com/AndrixNg1) · [Portfolio](https://andrixng.dev)

## License

No license has been assigned to this project yet.
