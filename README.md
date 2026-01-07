# 🧪 CIFAR-10 Classifier: PyTorch & Optuna

![Status](https://img.shields.io/badge/Status-Active%20Optimization-orange) ![Accuracy](https://img.shields.io/badge/Current%20Best-77%25-green) ![PyTorch](https://img.shields.io/badge/PyTorch-Learning-red)

**Current Status:** Active educational project exploring CNN architecture design and automated hyperparameter optimization.

This repository contains a modular computer vision pipeline built from scratch in PyTorch. It features a custom dynamic CNN architecture and integrates **Optuna** for automated hyperparameter tuning. The project is currently achieving around **~77% Test Accuracy** on the CIFAR-10 dataset, with active work focusing on reducing overfitting and improving validation strategies.

## 🧠 Project Architecture

### The Model: `GeneralCNN`
Unlike a static hard-coded model, I implemented a flexible class `GeneralCNN` (in `src/model.py`) that constructs the architecture based on runtime arguments. This allows for rapid experimentation with depth and width without rewriting code.
* **Dynamic Depth:** Stackable Convolutional Blocks (Conv2d -> ReLU -> MaxPool).
* **Adaptive Pooling:** Uses `AdaptiveAvgPool2d` to handle variable feature map sizes before the dense layers.
* **Regularization:** Integrated Dropout and Batch Normalization (Planned) to combat overfitting.

### The Pipeline
The codebase is structured for reproducibility and experimentation:
* **`main.py`**: Entry point handling CLI arguments for training vs. tuning modes.
* **`src/optuna_tune.py`**: Automated search for optimal Learning Rate, Dropout, and Architecture depth.
* **`src/train.py`**: Modular training and validation loops.

## 📊 Current Results & Configuration

Through automated tuning with Optuna, the current best model achieves around **77% Accuracy** using the following hyperparameters:

```json
{
    "lr": 0.00033,
    "dropout": 0.42,
    "fc1_size": 512,
    "num_blocks": 4,
    "base_filters": 64
}
