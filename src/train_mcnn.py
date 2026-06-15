"""
Script to train the Multi-Scale CNN using pre-extracted DNABERT-2 embeddings.
"""

import argparse
import os
import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Fix import path for direct script execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mcnn_model import MultiScaleCNN
from src.train import DNAEmbeddingDataset, evaluate_model, plot_curves, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Multi-Scale CNN on top of saved DNABERT-2 embeddings")
    parser.add_argument("--embeddings_path", type=str, default="data/processed/dnabert_embeddings.npy", help="Path to embeddings npy file")
    parser.add_argument("--labels_path", type=str, default="data/processed/dnabert_labels.npy", help="Path to labels npy file")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save the trained model")
    parser.add_argument("--figures_dir", type=str, default="figures", help="Directory to save evaluation plots")
    args = parser.parse_args()
    
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        
    print(f"Using device: {device}")
    
    # 1. Load pre-extracted tensors
    print(f"Loading embeddings from {args.embeddings_path}...")
    X = np.load(args.embeddings_path)
    print(f"Loading labels from {args.labels_path}...")
    y = np.load(args.labels_path)
    print(f"Data shape: X = {X.shape}, y = {y.shape}")
    
    # 2. Stratified train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Split sizes: Train = {X_train.shape[0]}, Val = {X_val.shape[0]}")
    
    # 3. Create datasets and dataloaders (uses memory-sharing torch.from_numpy)
    train_dataset = DNAEmbeddingDataset(X_train, y_train)
    val_dataset = DNAEmbeddingDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 4. Initialize Multi-Scale CNN
    print("\nInitializing Multi-Scale CNN classifier head...")
    model = MultiScaleCNN(
        embedding_dim=768, 
        branch_channels=128, 
        num_classes=4, 
        dropout_rate=args.dropout
    )
    
    # 5. Train the model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        output_dir=args.output_dir
    )
    
    # 6. Evaluate and save plots
    print("\nRunning model evaluation and saving figures...")
    plot_curves(history, save_dir=args.figures_dir)
    
    best_model_path = os.path.join(args.output_dir, "best_mcnn_model.pt")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Loaded best weights from {best_model_path} for final evaluation.")
    
    class_names = ['SP1', 'SP2', 'SP4', 'Negative']
    evaluate_model(
        model=model,
        val_loader=val_loader,
        class_names=class_names,
        device=device,
        save_dir=args.figures_dir
    )
    
    print("\nTraining and evaluation workflow complete.")

if __name__ == "__main__":
    main()
