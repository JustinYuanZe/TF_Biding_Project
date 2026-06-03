#!/usr/bin/env python3
"""
Train an Improved One-Hot Multi-Scale CNN (mCNN) on the 4-class TF binding task.
Features:
- Integer mapped DNA sequences (A=0, C=1, G=2, T=3, N=4)
- Learnable Embedding Layer (maps bases to 32-dim space)
- Parallel Conv1D branches of multiple kernel sizes [3, 5, 7, 9]
- GroupShuffleSplit (revcomp-aware) to prevent data leakage
- Accuracy-based checkpointing and early stopping
"""

import os
import sys
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize

# Model definition (matches src/mcnn_model.py ImprovedOneHotCNN)
class ImprovedOneHotCNN(nn.Module):
    def __init__(self, seq_len=101, num_classes=4, embedding_dim=32, branch_channels=64, kernel_sizes=[3, 5, 7, 9], dropout_rate=0.6):
        super().__init__()
        # Categories: A=0, C=1, G=2, T=3, N/Padding=4
        self.embedding = nn.Embedding(5, embedding_dim, padding_idx=4)
        
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=branch_channels,
                    kernel_size=k,
                    padding=k // 2
                ),
                nn.BatchNorm1d(branch_channels),
                nn.ReLU(),
                nn.Dropout(p=0.2)
            ) for k in kernel_sizes
        ])
        
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        concatenated_dim = len(kernel_sizes) * branch_channels * 2
        
        self.fc_head = nn.Sequential(
            nn.Linear(concatenated_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)  # Shape: (batch_size, seq_len, embedding_dim)
        x = x.transpose(1, 2)  # Shape: (batch_size, embedding_dim, seq_len)
        
        branch_feats = []
        for branch in self.branches:
            feat = branch(x)
            max_feat = self.max_pool(feat).squeeze(-1)
            avg_feat = self.avg_pool(feat).squeeze(-1)
            branch_feats.extend([max_feat, avg_feat])
            
        feat = torch.cat(branch_feats, dim=1)
        out = self.fc_head(feat)
        return out

def find_file(filename, fallback_dir="data/processed"):
    """Search for target_file in absolute paths, Kaggle input, fallback dirs, or current directory."""
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    
    # 1. Prioritize recursive search in /kaggle/input
    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if filename in files:
                fpath = os.path.join(root, filename)
                print(f"  [Auto-detect] Found {filename} at {fpath}")
                return fpath
                
    # 2. Check fallback dir and its subdirectory 'fixed_negative'
    if fallback_dir and os.path.exists(fallback_dir):
        p1 = os.path.join(fallback_dir, filename)
        if os.path.exists(p1):
            return p1
        p2 = os.path.join(fallback_dir, "fixed_negative", filename)
        if os.path.exists(p2):
            return p2
            
    # 3. Check current working directory
    if os.path.exists(filename):
        return filename
        
    return None


def auto_detect_dir(target_file, fallback="data/processed"):
    """Search for target_file in local path or Kaggle directory."""
    resolved_path = find_file(target_file, fallback)
    if resolved_path:
        return os.path.dirname(resolved_path)
    return fallback

def load_fasta(filepath):
    """Load sequences and headers from a FASTA file."""
    sequences = []
    headers = []
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing file: {filepath}")
    with open(filepath, "r") as f:
        seq_lines = []
        current_header = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_lines:
                    sequences.append("".join(seq_lines).upper())
                    headers.append(current_header)
                    seq_lines = []
                current_header = line[1:]
            else:
                seq_lines.append(line)
        if seq_lines:
            sequences.append("".join(seq_lines).upper())
            headers.append(current_header)
    return sequences, headers

def seqs_to_indices(sequences):
    """Map nucleotides to integer indices (A=0, C=1, G=2, T=3, N=4)."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    n = len(sequences)
    seq_len = len(sequences[0])
    indices = np.zeros((n, seq_len), dtype=np.int64)
    for i, seq in enumerate(sequences):
        for j, nuc in enumerate(seq):
            indices[i, j] = mapping.get(nuc, 4)
    return indices

def main():
    # Seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths config
    fasta_dir = auto_detect_dir("sp1_positive_final.fasta", "data/processed")
    output_dir = "outputs_onehot_mcnn"
    fig_dir = os.path.join(output_dir, "figures")
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Load data
    print("\n" + "=" * 60)
    print("LOADING DATASETS (One-Hot + mCNN)")
    print("=" * 60)
    
    neg_fasta_path = None
    fasta_candidates = ["negative_genomic_matched.fasta", "negative_promoter_cpg.fasta", "negative_final.fasta"]
    for cand in fasta_candidates:
        path = find_file(cand, fasta_dir)
        if path:
            neg_fasta_path = path
            break
            
    if not neg_fasta_path:
        raise FileNotFoundError("Could not find any negative FASTA file among candidates.")
            
    print(f"  [Auto-detect] Using negative FASTA file: {neg_fasta_path}")
    
    fasta_files = {
        "SP1": find_file("sp1_positive_final.fasta", fasta_dir),
        "SP2": find_file("sp2_positive_final.fasta", fasta_dir),
        "SP4": find_file("sp4_positive_final.fasta", fasta_dir),
        "Negative": neg_fasta_path
    }
    
    # Verify all are resolved
    for cls_name, fpath in fasta_files.items():
        if not fpath:
            raise FileNotFoundError(f"Missing FASTA file for {cls_name}")
            
    all_sequences = []
    all_headers = []
    all_labels = []
    all_groups = []
    group_id = 0
    
    for cls_idx, (cls_name, fpath) in enumerate(fasta_files.items()):
        seqs, hdrs = load_fasta(fpath)
        print(f"  {cls_name}: {len(seqs)} sequences ({os.path.basename(fpath)})")
        all_sequences.extend(seqs)
        all_headers.extend(hdrs)
        all_labels.extend([cls_idx] * len(seqs))
        
        if cls_name != "Negative":
            for i in range(0, len(seqs), 2):
                all_groups.extend([group_id, group_id])
                group_id += 1
        else:
            for _ in seqs:
                all_groups.append(group_id)
                group_id += 1
                
    all_labels = np.array(all_labels)
    all_groups = np.array(all_groups)
    
    # Map sequences to integers
    X = seqs_to_indices(all_sequences)
    
    # 2. Split data (group-aware)
    print("\n" + "=" * 60)
    print("SPLITTING DATA (GroupShuffleSplit — no revcomp leakage)")
    print("=" * 60)
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, all_labels, all_groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = all_labels[train_idx], all_labels[test_idx]
    headers_test = [all_headers[i] for i in test_idx]
    
    print(f"  Train: {X_train.shape[0]} sequences")
    print(f"  Test:  {X_test.shape[0]} sequences")
    print(f"  Train class dist: {np.bincount(y_train)}")
    print(f"  Test  class dist: {np.bincount(y_test)}")
    
    # 3. DataLoaders
    train_ds = TensorDataset(torch.from_numpy(X_train).long(), torch.from_numpy(y_train).long())
    test_ds = TensorDataset(torch.from_numpy(X_test).long(), torch.from_numpy(y_test).long())
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    
    # 4. Initialize model
    model = ImprovedOneHotCNN(
        seq_len=101,
        num_classes=4,
        embedding_dim=32,
        branch_channels=64,
        kernel_sizes=[3, 5, 7, 9],
        dropout_rate=0.6
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel initialized with {total_params:,} parameters.")
    
    # 5. Training config
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    
    class_counts = np.bincount(y_train)
    total_count = len(y_train)
    class_weights = total_count / (4 * class_counts.astype(np.float32))
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    epochs = 25
    patience = 10
    patience_counter = 0
    best_val_acc = 0.0
    best_val_loss = float('inf')
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    print("\n" + "=" * 60)
    print("=== TRAINING ONE-HOT + mCNN ===")
    print("=" * 60)
    
    for epoch in range(epochs):
        t0 = time.time()
        
        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Eval
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                running_val_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        val_loss = running_val_loss / val_total
        val_acc = val_correct / val_total
        elapsed = time.time() - t0
        
        # Log
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_acc)
        
        print(
            f"Epoch {epoch+1:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.1e} | {elapsed:.1f}s"
        )
        
        # Checkpoint based on val_acc
        if val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            ckpt_path = os.path.join(model_dir, "best_onehot_mcnn.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → Saved best model (val_loss={val_loss:.4f}, val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  ⏹ Early stopping at epoch {epoch+1} (patience={patience})")
                break
                
    print(f"\nTraining complete. Best val_acc: {best_val_acc:.4f}")
    
    # 6. Evaluation on Best Checkpoint
    best_path = os.path.join(model_dir, "best_onehot_mcnn.pt")
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs.float(), dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Print report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    class_names = ["SP1", "SP2", "SP4", "Negative"]
    report_str = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
    print(report_str)
    
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write("CLASSIFICATION REPORT — One-Hot + mCNN Baseline\n")
        f.write("=" * 60 + "\n")
        f.write(report_str)
        
    # Generate Curves
    # A. Training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(history["train_loss"])+1), history["train_loss"], label="Train Loss")
    plt.plot(range(1, len(history["val_loss"])+1), history["val_loss"], label="Val Loss")
    plt.title("Loss Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(history["train_acc"])+1), history["train_acc"], label="Train Acc")
    plt.plot(range(1, len(history["val_acc"])+1), history["val_acc"], label="Val Acc")
    plt.axhline(y=0.25, color="gray", linestyle="--", alpha=0.5, label="Random Guess")
    plt.title("Accuracy curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "onehot_mcnn_training_curves.png"), dpi=150)
    plt.close()
    
    # B. Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.title("Confusion Matrix (Counts)")
    plt.xticks(range(4), class_names)
    plt.yticks(range(4), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar(im)
    
    thresh = cm.max() / 2.
    for i in range(4):
        for j in range(4):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "onehot_mcnn_confusion_matrix.png"), dpi=150)
    plt.close()
    
    print(f"\nAll figures saved to: {fig_dir}")
    
    # 7. Post-processing: Export BED files for IGV Analysis
    def parse_header_to_bed(header):
        try:
            clean_hdr = header.split("_")[0]
            chrom, coords = clean_hdr.split(":")
            start, end = coords.split("-")
            return chrom, start, end
        except Exception:
            return None

    true_sp1_coords = []
    true_sp4_coords = []
    confused_sp4_as_sp1_coords = []
    
    for idx, (pred, target) in enumerate(zip(all_preds, all_targets)):
        header = headers_test[idx]
        bed_fields = parse_header_to_bed(header)
        if not bed_fields:
            continue
        chrom, start, end = bed_fields
        bed_line = f"{chrom}\t{start}\t{end}\n"
        
        if target == 0 and pred == 0:
            true_sp1_coords.append(bed_line)
        elif target == 2 and pred == 2:
            true_sp4_coords.append(bed_line)
        elif target == 2 and pred == 0:
            confused_sp4_as_sp1_coords.append(bed_line)
            
    with open(os.path.join(output_dir, "True_SP1.bed"), "w") as f:
        f.writelines(true_sp1_coords)
    with open(os.path.join(output_dir, "True_SP4.bed"), "w") as f:
        f.writelines(true_sp4_coords)
    with open(os.path.join(output_dir, "Confused_SP4_as_SP1.bed"), "w") as f:
        f.writelines(confused_sp4_as_sp1_coords)
        
    print(f"Exported BED files to {output_dir} for visual validation.")

if __name__ == "__main__":
    main()
