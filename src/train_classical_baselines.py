#!/usr/bin/env python3
"""
Train 9 classical ML baseline classifiers for the 4-class TF binding prediction task.
Models:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Extra Trees
5. Gradient Boosting (using fast HistGradientBoosting)
6. SVM (Support Vector Classifier with probability=True)
7. KNN (n=5)
8. Naive Bayes (BernoulliNB)
9. MLP (traditional Multilayer Perceptron)

Features:
- DNA k-mer TF-IDF representation (ngram_range=(3, 5))

Saves performance figures to figures/ directory.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
)

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier

# Config
DATA_DIR = "data/processed"
FIG_DIR = "figures"
CLASS_NAMES = ["SP1", "SP2", "SP4", "Negative"]

def auto_detect_dir(target_file, fallback="data/processed"):
    """Search for the directory containing target_file in Kaggle input or local path."""
    if os.path.exists(fallback) and os.path.exists(os.path.join(fallback, target_file)):
        return fallback
    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if target_file in files:
                print(f"  [Auto-detect] Found {target_file} at {root}")
                return root
    return fallback

def load_fasta(filepath):
    """Load DNA sequences from a FASTA file and convert to uppercase."""
    sequences = []
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing dataset file: {filepath}")
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith(">"):
                sequences.append(line.upper())
    return sequences

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("=== TRAINING CLASSICAL ML BASELINES ===")

    # 1. Load Data
    fasta_dir = auto_detect_dir("sp1_positive_final.fasta", DATA_DIR)
    
    # Detect negative FASTA file
    neg_fasta = "negative_final.fasta"
    fasta_candidates = ["negative_genomic_matched.fasta", "negative_promoter_cpg.fasta", "negative_final.fasta"]
    for cand in fasta_candidates:
        if os.path.exists(os.path.join(fasta_dir, cand)):
            neg_fasta = cand
            break
        elif os.path.exists(os.path.join(fasta_dir, "fixed_negative", cand)):
            neg_fasta = os.path.join("fixed_negative", cand)
            break

    print(f"  [Auto-detect] Using negative FASTA file: {neg_fasta}")

    sp1_path = os.path.join(fasta_dir, "sp1_positive_final.fasta")
    sp2_path = os.path.join(fasta_dir, "sp2_positive_final.fasta")
    sp4_path = os.path.join(fasta_dir, "sp4_positive_final.fasta")
    neg_path = os.path.join(fasta_dir, neg_fasta)

    try:
        seqs_sp1 = load_fasta(sp1_path)
        seqs_sp2 = load_fasta(sp2_path)
        seqs_sp4 = load_fasta(sp4_path)
        seqs_neg = load_fasta(neg_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the dataset is properly mounted or prepared.")
        sys.exit(1)

    print(f"Loaded datasets:")
    print(f"  SP1 Positive: {len(seqs_sp1)} sequences")
    print(f"  SP2 Positive: {len(seqs_sp2)} sequences")
    print(f"  SP4 Positive: {len(seqs_sp4)} sequences")
    print(f"  Negative:     {len(seqs_neg)} sequences")

    sequences = seqs_sp1 + seqs_sp2 + seqs_sp4 + seqs_neg
    y = np.concatenate([
        np.zeros(len(seqs_sp1)),
        np.ones(len(seqs_sp2)),
        np.full(len(seqs_sp4), 2),
        np.full(len(seqs_neg), 3)
    ], axis=0)

    # 2. Build group IDs to prevent reverse-complement data leakage.
    # In positive FASTA files, sequences alternate: [orig_0, rc_0, orig_1, rc_1, ...]
    # A sequence and its reverse complement MUST stay in the same split,
    # otherwise the model has effectively "seen" the test sample in a different form.
    # Negative sequences are independent (each is its own group).
    groups = []
    group_id = 0
    for class_seqs in [seqs_sp1, seqs_sp2, seqs_sp4]:
        for i in range(0, len(class_seqs), 2):
            groups.extend([group_id, group_id])
            group_id += 1
    for _ in seqs_neg:
        groups.append(group_id)
        group_id += 1
    groups = np.array(groups)

    # 3. Split raw sequences FIRST, THEN extract features (prevents TF-IDF leakage).
    # Use GroupShuffleSplit to keep orig/revcomp pairs in the same split.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(sequences, y, groups))

    seq_train = [sequences[i] for i in train_idx]
    seq_test = [sequences[i] for i in test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    print(f"\nTrain/Test split (group-aware): {len(seq_train)} train, {len(seq_test)} test")
    print(f"  Train class distribution: {np.bincount(y_train.astype(int))}")
    print(f"  Test  class distribution: {np.bincount(y_test.astype(int))}")

    # 4. Feature Extraction (k-mer TF-IDF) — fit ONLY on training data
    print("\nExtracting k-mer TF-IDF features (k=3 to 5)...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
    X_train = vectorizer.fit_transform(seq_train)   # fit + transform on train
    X_test = vectorizer.transform(seq_test)          # transform only on test
    print(f"Feature matrix: Train {X_train.shape}, Test {X_test.shape} (extracted in {time.time() - t0:.2f}s)")

    # Convert to dense for classifiers that don't support sparse input
    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()

    # 5. Define 9 Classifiers
    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1), False),
        "Decision Tree": (DecisionTreeClassifier(max_depth=10, random_state=42), False),
        "Random Forest": (RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42), False),
        "Extra Trees": (ExtraTreesClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42), False),
        "Gradient Boosting": (HistGradientBoostingClassifier(max_iter=100, random_state=42), True), # needs dense X
        "SVM": (SVC(probability=True, max_iter=1000, random_state=42), False),
        "KNN (k=5)": (KNeighborsClassifier(n_neighbors=5, n_jobs=-1), False),
        "Naive Bayes": (BernoulliNB(), False),
        "MLP": (MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, early_stopping=True, random_state=42), False),
    }

    # 6. Training Loop and Evaluation
    results = []
    confusion_matrices = {}
    roc_curves_data = {}

    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])

    print("\nTraining and evaluating models...")
    for name, (model, needs_dense) in models.items():
        print(f"  Training {name}...")
        t_start = time.time()
        
        # Select dense or sparse input
        X_tr = X_train_dense if needs_dense else X_train
        X_te = X_test_dense if needs_dense else X_test
        
        model.fit(X_tr, y_train)
        elapsed = time.time() - t_start
        
        # Predict
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Train Time (s)": elapsed
        })
        
        print(f"    -> Acc: {acc:.4f} | F1 (macro): {f1:.4f} | Time: {elapsed:.2f}s")
        
        # Save confusion matrix
        confusion_matrices[name] = confusion_matrix(y_test, y_pred)
        
        # Compute Macro-Average ROC
        fpr_dict = {}
        tpr_dict = {}
        for i in range(4):
            fpr_dict[i], tpr_dict[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            
        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(4)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(4):
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
        mean_tpr /= 4
        
        roc_curves_data[name] = (all_fpr, mean_tpr, auc(all_fpr, mean_tpr))

    # Print summary DataFrame
    df_results = pd.DataFrame(results)
    print("\n=== PERFORMANCE SUMMARY ===")
    print(df_results.to_string(index=False))

    # 7. Generate comparative graphs
    print("\nGenerating performance comparison graphs...")
    
    # Plot 1: Performance comparison bar plot
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(df_results))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, df_results[metric], width, label=metric)
        
    ax.axhline(y=0.25, color="red", linestyle="--", alpha=0.7, label="Random Guess (0.25)")
    ax.set_xticks(x)
    ax.set_xticklabels(df_results["Model"], rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Comparative Performance Metrics of 9 Classical ML Models")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig1_path = os.path.join(FIG_DIR, "classical_ml_metrics_comparison.png")
    plt.savefig(fig1_path, dpi=150)
    plt.close()
    print(f"  -> Saved performance metrics comparison to: {fig1_path}")

    # Plot 2: Confusion Matrices (3x3 grid)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    for idx, (name, cm) in enumerate(confusion_matrices.items()):
        ax = axes[idx]
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(name, fontsize=12, pad=10)
        
        # Show labels
        ax.set_xticks(range(4))
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticks(range(4))
        ax.set_yticklabels(CLASS_NAMES)
        
        # Annotate numbers
        thresh = cm.max() / 2.
        for i in range(4):
            for j in range(4):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
                
        ax.set_ylabel("True label" if idx % 3 == 0 else "")
        ax.set_xlabel("Predicted label" if idx >= 6 else "")
        
    plt.suptitle("Confusion Matrices of 9 Baseline Classifiers (4-class Task)", fontsize=16, y=0.98)
    plt.tight_layout()
    fig2_path = os.path.join(FIG_DIR, "classical_ml_confusion_matrices.png")
    plt.savefig(fig2_path, dpi=150)
    plt.close()
    print(f"  -> Saved confusion matrices to: {fig2_path}")

    # Plot 3: Macro-Average ROC Curves
    plt.figure(figsize=(10, 8))
    for name, (fpr, tpr, roc_auc) in roc_curves_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Macro-Average ROC Curves Comparison", fontsize=14, pad=15)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig3_path = os.path.join(FIG_DIR, "classical_ml_roc_curves.png")
    plt.savefig(fig3_path, dpi=150)
    plt.close()
    print(f"  -> Saved ROC curves to: {fig3_path}")
    
    print("\nAll models trained and evaluated. Figures saved successfully!")

if __name__ == "__main__":
    main()
