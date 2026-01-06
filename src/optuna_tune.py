import torch
import torch.nn as nn
import torch.optim as optim
import optuna

# Modular imports from your own project
from src.model import GeneralCNN
from src.utils import get_loaders
from src.train import train_one_epoch, validate

def objective(trial, device, dataset_name):
    # --- 1. HYPERPARAMETER SUGGESTIONS (The Behavior) ---
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    fc1_size = trial.suggest_categorical("fc1_size", [128, 256, 512])
    
    # --- 2. ARCHITECTURE SUGGESTIONS (The Shape) ---
    num_blocks = trial.suggest_int("num_blocks", 4, 5) 
    base_filters = trial.suggest_categorical("base_filters", [32, 64]) 

    # --- 3. SETUP ---
    train_loader, test_loader = get_loaders(dataset_name=dataset_name)
    in_channels = 3 if dataset_name == "CIFAR10" else 1
    
    # Initialize the model with ALL the suggestions
    model = GeneralCNN(
        in_channels=in_channels,
        num_classes=10,
        dropout_rate=dropout,
        num_blocks=num_blocks,
        base_filters=base_filters,
        fc1_size=fc1_size
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # --- 4. TUNING LOOP ---
    # We keep this short (2-3 epochs) so Optuna can try more combinations quickly
    for epoch in range(2):
        train_one_epoch(model, device, train_loader, optimizer, criterion)
        accuracy = validate(model, device, test_loader)
        
        # Pruning: Stops the trial early if it's performing poorly
        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return accuracy