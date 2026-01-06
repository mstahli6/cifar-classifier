import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import optuna
from pathlib import Path

# Import our custom modules
from src.model import GeneralCNN
from src.utils import get_loaders
from src.train import train_one_epoch, validate
from src.optuna_tune import objective

import sys


# This tells Python to look in the current folder for the 'src' package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))



def main():
    parser = argparse.ArgumentParser(description="Generalizable Vision Pipeline")
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'tune'],
                        help="Run 'train' for a final model or 'tune' for hyperparameter search")
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10'],
                        help="Which dataset to use")
    parser.add_argument('--epochs', type=int, default=5, 
                        help="Number of epochs to train")
    
    args = parser.parse_args()

    # 1. Hardware Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"✅ Running {args.mode} on {args.dataset} using {device}")

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Define paths for this specific dataset
    config_path = f"models/best_config_{args.dataset}.json"
    model_path = f"models/{args.dataset.lower()}_model.pth"

    # 2. Execution Logic
    if args.mode == 'tune':
        print(f"🚀 Starting Hyperparameter Tuning for {args.dataset}...")
        study = optuna.create_study(direction="maximize")
        
        # We use a lambda to pass arguments into the objective function
        study.optimize(lambda trial: objective(trial, device, args.dataset), n_trials=20)
        
        # Save the winner
        with open(config_path, "w") as f:
            json.dump(study.best_params, f, indent=4)
        
        print(f"\n🏆 Tuning Complete!")
        print(f"Best Accuracy: {study.best_value:.4f}")
        print(f"Best Params saved to {config_path}")

    else:
        print(f"🏗️ Starting Final Training for {args.dataset}...")
        
        # 1. Load best params with expanded defaults
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                best_params = json.load(f)
            print(f"📂 Loaded tuned params from {config_path}")
        else:
            # Updated defaults to match the new architecture
            best_params = {
                "lr": 0.001, 
                "dropout": 0.3, 
                "fc1_size": 128,
                "num_blocks": 3, 
                "base_filters": 32
            }
            print("⚠️ No tuned config found. Using default hyperparameters.")

        # 2. Data & Model Setup
        train_loader, test_loader = get_loaders(dataset_name=args.dataset, batch_size=64)
        in_channels = 3 if args.dataset == "CIFAR10" else 1
        
        # Pass ALL the new parameters into the model
        model = GeneralCNN(
            in_channels=in_channels, 
            num_classes=10, 
            dropout_rate=best_params.get('dropout', 0.3),
            num_blocks=best_params.get('num_blocks', 3),
            base_filters=best_params.get('base_filters', 32),
            fc1_size=best_params.get('fc1_size', 128)
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=best_params.get('lr', 0.001))
        criterion = nn.CrossEntropyLoss()

        # 3. Training Loop
        for epoch in range(1, args.epochs + 1):
            train_one_epoch(model, device, train_loader, optimizer, criterion)
            acc = validate(model, device, test_loader)
            print(f"Epoch {epoch}/{args.epochs} | Accuracy: {acc*100:.2f}%")
        
        # Save Model Weights
        torch.save(model.state_dict(), model_path)
        print(f"💾 Model weights saved to {model_path}")

if __name__ == "__main__":
    main()