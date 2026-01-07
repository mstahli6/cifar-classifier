# 🚧 [WIP] CIFAR-10 Image Classifier (Learning PyTorch)

![Status](https://img.shields.io/badge/Status-In%20Development-orange) ![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-Learning-red)

**Current Status:** Active Development / Educational Project

This repository serves as a learning sandbox for mastering **Convolutional Neural Networks (CNNs)** and the **PyTorch** ecosystem. The goal is to build, train, and optimize an image classifier from scratch on the CIFAR-10 dataset, documenting the learning process along the way.

## 🎯 Learning Objectives

* **Architecture Design:** Experimenting with manual CNN layer construction (Conv2d, MaxPool, BatchNorm) vs. Transfer Learning.
* **PyTorch Internals:** Writing custom training loops instead of using high-level abstractions to understand backpropagation and gradient steps.
* **Optimization:** Comparing different optimizers (SGD vs. Adam) and learning rate schedulers.

## 📝 Project Roadmap & Progress

- [x] **Environment Setup:** PyTorch dependencies and GPU/MPS acceleration checks.
- [x] **Data Pipeline:**
    - [x] Download CIFAR-10 dataset via `torchvision`.
    - [ ] Implement data augmentation (RandomCrop, HorizontalFlip).
    - [ ] Create efficient DataLoaders.
- [ ] **Model Architecture:**
    - [ ] Define baseline CNN (Convolution -> ReLU -> Pool).
    - [ ] Implement fully connected classification head.
- [ ] **Training Loop:**
    - [ ] Implement forward pass and loss calculation (CrossEntropy).
    - [ ] Implement backpropagation and optimizer step.
    - [ ] Add validation step to monitor overfitting.
- [ ] **Evaluation:**
    - [ ] accurate metric calculation.
    - [ ] Confusion matrix visualization.

## 🛠️ Tech Stack

* **Core:** Python, PyTorch
* **Data:** Torchvision, NumPy
* **Visualization:** Matplotlib

## 🚀 Usage (Current State)

*Note: The code is currently in the early implementation phase.*

To verify the setup and download the data:
```bash
python src/data_setup.py
