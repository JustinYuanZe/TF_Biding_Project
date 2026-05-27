import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

class DNAEmbeddingDataset(Dataset):
    """
    Custom PyTorch Dataset for loading pre-extracted DNABERT-2 embeddings and labels.
    Uses memory-sharing (torch.from_numpy) to prevent duplicating large arrays in RAM.
    """
    def __init__(self, embeddings, labels):
        if isinstance(embeddings, np.ndarray):
            self.embeddings = torch.from_numpy(embeddings)
        else:
            self.embeddings = embeddings

        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels)
        else:
            self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert to float32/long dynamically during batch collation
        return self.embeddings[idx].to(torch.float32), self.labels[idx].to(torch.long)


def train_model(model, train_loader, val_loader, epochs=15, lr=0.001, device='cuda', output_dir='models'):
    """
    Train the mCNN model and track loss/accuracy curves.
    """
    os.makedirs(output_dir, exist_ok=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_loss = float('inf')

    print("\n=== Start Training mCNN ===")
    for epoch in range(epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            if isinstance(inputs, dict):
                inputs = {k: v.to(device) for k, v in inputs.items()}
            else:
                inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            if isinstance(inputs, dict):
                outputs = model(**inputs)
            else:
                outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                if isinstance(inputs, dict):
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                else:
                    inputs = inputs.to(device)
                targets = targets.to(device)
                
                if isinstance(inputs, dict):
                    outputs = model(**inputs)
                else:
                    outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        # Learning rate scheduling
        scheduler.step(epoch_val_loss)
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.4f}")
        
        # Save best checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_mcnn_model.pt"))
            
    print(f"Training completed. Best Val Loss: {best_val_loss:.4f}")
    return history


def plot_curves(history, save_dir='figures'):
    """
    Plot and save training/validation Loss and Accuracy curves.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = len(history['train_loss'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(range(1, epochs + 1), history['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    ax1.plot(range(1, epochs + 1), history['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2)
    ax1.set_title('Loss Convergence')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Accuracy plot
    ax2.plot(range(1, epochs + 1), history['train_acc'], label='Train Acc', color='#2ca02c', linewidth=2)
    ax2.plot(range(1, epochs + 1), history['val_acc'], label='Val Acc', color='#d62728', linewidth=2)
    ax2.set_title('Accuracy Performance')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mcnn_training_curves.png"), dpi=150)
    plt.close()
    print(f"Training curves saved to: {save_dir}/mcnn_training_curves.png")


def evaluate_model(model, val_loader, class_names, device='cuda', save_dir='figures'):
    """
    Generate ROC curves, Precision-Recall curves, and Confusion Matrix.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            if isinstance(inputs, dict):
                inputs = {k: v.to(device) for k, v in inputs.items()}
            else:
                inputs = inputs.to(device)
            
            if isinstance(inputs, dict):
                outputs = model(**inputs)
            else:
                outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    num_classes = len(class_names)
    
    # 1. Print classification report
    print("\n=== Classification Report ===")
    print(classification_report(all_targets, all_preds, target_names=class_names))
    
    # 2. Confusion Matrix Plot
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=14, pad=15)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Annotate counts in cells
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
                 
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"Confusion Matrix saved to: {save_dir}/confusion_matrix.png")
    
    # 3. Multi-Class ROC Curves
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        # Binary target vector for class i
        bin_targets = (all_targets == i).astype(int)
        fpr, tpr, _ = roc_curve(bin_targets, all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.4f})')
        
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curves.png"), dpi=150)
    plt.close()
    print(f"ROC Curves saved to: {save_dir}/roc_curves.png")

    # 4. Multi-Class Precision-Recall Curves
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        bin_targets = (all_targets == i).astype(int)
        precision, recall, _ = precision_recall_curve(bin_targets, all_probs[:, i])
        ap = average_precision_score(bin_targets, all_probs[:, i])
        plt.plot(recall, precision, label=f'{class_names[i]} (AP = {ap:.4f})')
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves', fontsize=14)
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "precision_recall_curves.png"), dpi=150)
    plt.close()
    print(f"Precision-Recall Curves saved to: {save_dir}/precision_recall_curves.png")
