#!/usr/bin/env python3
"""
🧬 Script 27: Classical ML Baselines for Binary Transcription Factor Binding Prediction
========================================================================================
Task: Binary classification of SP-family TF binding (Positive: SP1+SP2+SP4) vs. Genomic Negative control.
Features: 
  1. One-hot encoding (flattened to 404 features)
  2. k-mer TF-IDF representation (character n-grams of DNA, length 4 to 6)
Models:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. Extra Trees
  5. Gradient Boosting
  6. AdaBoost
  7. SVM (Linear)
  8. KNN (k=5)
  9. Naive Bayes
  10. MLP (Multi-Layer Perceptron)
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
class Config:
    DATA_DIRS = ["data/processed", "/kaggle/input/datasets/lehotrongtin/datas1"]
    OUTPUT_DIR = "outputs_binary_classical_ml"
    FIG_DIR = "figures/classical_ml_binary"
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    K_MIN = 4
    K_MAX = 6

cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
os.makedirs(cfg.FIG_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# UTILITIES & DATA LOADING
# ═══════════════════════════════════════════════════════════════════════
def find_file(filename, dirs):
    """Dynamically search directories for a file."""
    for d in dirs:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    # Try recursive search in current directory
    for root, _, files in os.walk("."):
        if filename in files:
            return os.path.join(root, filename)
    return None

def load_fasta_seqs(fasta_path):
    """Load and return capitalized DNA sequences from a FASTA file."""
    sequences = []
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('>'):
                sequences.append(line.upper())
    return sequences

def seqs_to_onehot(sequences):
    """Convert DNA sequences of length 101 to flattened one-hot representation."""
    n_samples = len(sequences)
    seq_array = np.array([list(seq) for seq in sequences], dtype='U1')
    onehot = np.zeros((n_samples, 101, 4), dtype=np.int8)
    onehot[seq_array == 'A', 0] = 1
    onehot[seq_array == 'C', 1] = 1
    onehot[seq_array == 'G', 2] = 1
    onehot[seq_array == 'T', 3] = 1
    return onehot.reshape(n_samples, -1)

def extract_kmer_features(sequences, ngram_range=(4, 6)):
    """Extract character n-gram TF-IDF k-mer features."""
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=ngram_range)
    return vectorizer.fit_transform(sequences)

# ═══════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. LOADING FASTA DATASETS")
print("=" * 60)

sp1_path = find_file("sp1_positive_final.fasta", cfg.DATA_DIRS)
sp2_path = find_file("sp2_positive_final.fasta", cfg.DATA_DIRS)
sp4_path = find_file("sp4_positive_final.fasta", cfg.DATA_DIRS)

# Try matched negative first, fallback to original negative
neg_path = find_file("negative_genomic_matched.fasta", cfg.DATA_DIRS)
if not neg_path:
    neg_path = find_file("negative_final.fasta", cfg.DATA_DIRS)

if not all([sp1_path, sp2_path, sp4_path, neg_path]):
    raise FileNotFoundError(f"Missing FASTA files: SP1={sp1_path}, SP2={sp2_path}, SP4={sp4_path}, Negative={neg_path}")

print(f"  [Positive SP1] : {sp1_path}")
print(f"  [Positive SP2] : {sp2_path}")
print(f"  [Positive SP4] : {sp4_path}")
print(f"  [Negative Ctrl]: {neg_path}")

seqs_sp1 = load_fasta_seqs(sp1_path)
seqs_sp2 = load_fasta_seqs(sp2_path)
seqs_sp4 = load_fasta_seqs(sp4_path)
seqs_neg = load_fasta_seqs(neg_path)

print(f"\n  SP1 Positive : {len(seqs_sp1)} sequences")
print(f"  SP2 Positive : {len(seqs_sp2)} sequences")
print(f"  SP4 Positive : {len(seqs_sp4)} sequences")
print(f"  Genomic Neg  : {len(seqs_neg)} sequences")

# Merge Positive (SP1 + SP2 + SP4) -> Class 1, and Negative -> Class 0
pos_sequences = seqs_sp1 + seqs_sp2 + seqs_sp4
neg_sequences = seqs_neg

all_sequences = pos_sequences + neg_sequences
all_labels = np.concatenate([np.ones(len(pos_sequences)), np.zeros(len(neg_sequences))], axis=0)

print(f"\n  Total dataset size: {len(all_sequences)} sequences")
print(f"  Class distribution: Negative={len(neg_sequences)} (0), Positive={len(pos_sequences)} (1)")

# ═══════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION & SPLITTING
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. EXTRACTING REPRESENTATIONS")
print("=" * 60)

print("  -> Extracting One-Hot features...")
X_onehot = seqs_to_onehot(all_sequences)
print(f"     One-hot matrix shape: {X_onehot.shape}")

print("  -> Extracting k-mer TF-IDF features...")
X_tfidf = extract_kmer_features(all_sequences, ngram_range=(cfg.K_MIN, cfg.K_MAX))
print(f"     TF-IDF matrix shape: {X_tfidf.shape}")

# ═══════════════════════════════════════════════════════════════════════
# EVALUATION ROUTINE
# ═══════════════════════════════════════════════════════════════════════
def train_and_eval_baselines(X, y, representation_name):
    print(f"\nTraining on {representation_name} features...")
    
    # Train/Test Split (Stratified to maintain class ratios)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, stratify=y, random_state=cfg.RANDOM_SEED
    )
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=cfg.RANDOM_SEED, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=cfg.RANDOM_SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=cfg.RANDOM_SEED),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=cfg.RANDOM_SEED),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=cfg.RANDOM_SEED),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=cfg.RANDOM_SEED),
        "SVM (Linear)": SVC(kernel='linear', probability=True, random_state=cfg.RANDOM_SEED),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Naive Bayes": BernoulliNB(),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=cfg.RANDOM_SEED),
    }
    
    results = {}
    roc_curves = {}
    pr_curves = {}
    confusion_matrices = {}
    best_acc = 0.0
    best_model_name = ""
    best_y_pred = None
    best_y_prob = None
    
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        
        # Predictions
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            # Fallback for models without predict_proba (like SVM if probability=False, though we set it True)
            y_prob = model.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-y_prob))  # sigmoid
            
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="binary")
        prec = precision_score(y_test, y_pred, average="binary")
        rec = recall_score(y_test, y_pred, average="binary")
        
        # ROC / PR metrics
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = auc(fpr, tpr)
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        
        results[name] = {
            "Accuracy": acc,
            "F1-Score": f1,
            "Precision": prec,
            "Recall": rec,
            "ROC-AUC": auc_val,
            "AvgPrecision": ap,
            "Time (s)": elapsed
        }
        
        roc_curves[name] = (fpr, tpr, auc_val)
        pr_curves[name] = (recall_vals, precision_vals, ap)
        confusion_matrices[name] = confusion_matrix(y_test, y_pred)
        
        print(f"  {name:<20} | Acc: {acc:.4%}, AUC: {auc_val:.4f}, F1: {f1:.4f} | {elapsed:.2f}s")
        
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_y_pred = y_pred
            best_y_prob = y_prob
            
    df_res = pd.DataFrame(results).T.sort_values(by="Accuracy", ascending=False)
    
    # Save text report
    report_path = os.path.join(cfg.OUTPUT_DIR, f"{representation_name.lower()}_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"=== Classical ML Baseline Report ({representation_name} Features) ===\n\n")
        f.write(df_res.to_string())
        f.write(f"\n\nBest performing model: {best_model_name} (Accuracy: {best_acc:.4f})\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, best_y_pred, target_names=["Negative", "SP_Positive"], digits=4))
        
    print(f"  -> Report saved: {report_path}")
    
    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df_res.index, y=df_res["Accuracy"], palette="Blues_r")
    random_baseline = np.sum(y == 1) / len(y)
    plt.axhline(y=random_baseline, color="r", linestyle="--", label=f"Random Guess ({random_baseline:.2f})")
    plt.title(f"Classical ML Accuracy - {representation_name} Features", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right")
    plt.grid(axis='y', linestyle="--", alpha=0.3)
    plt.tight_layout()
    acc_chart_path = os.path.join(cfg.FIG_DIR, f"classical_{representation_name.lower()}_accuracy.png")
    plt.savefig(acc_chart_path, dpi=150)
    plt.close()
    
    # Plot ROC & PR Curves on side-by-side subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i, (name, (fpr, tpr, auc_val)) in enumerate(roc_curves.items()):
        ax1.plot(fpr, tpr, color=colors[i], linewidth=1.8, label=f"{name} (AUC = {auc_val:.4f})")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax1.set_xlabel("False Positive Rate", fontsize=11)
    ax1.set_ylabel("True Positive Rate", fontsize=11)
    ax1.set_title("ROC Curves Comparison", fontsize=13, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.3)
    
    for i, (name, (rec_vals, prec_vals, ap)) in enumerate(pr_curves.items()):
        ax2.plot(rec_vals, prec_vals, color=colors[i], linewidth=1.8, label=f"{name} (AP = {ap:.4f})")
    ax2.axhline(y=random_baseline, color="r", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.set_title("Precision-Recall Curves Comparison", fontsize=13, fontweight="bold")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.3)
    
    plt.suptitle(f"Performance Comparison - {representation_name} Features", fontsize=15, fontweight="bold")
    plt.tight_layout()
    curves_path = os.path.join(cfg.FIG_DIR, f"classical_{representation_name.lower()}_curves.png")
    plt.savefig(curves_path, dpi=150)
    plt.close()
    
    # Plot Confusion Matrix of the best model
    cm = confusion_matrices[best_model_name]
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    class_names = ["Negative", "SP_Positive"]
    
    for ax, data, title, fmt in [
        (ax1, cm, f"Confusion Matrix - Counts ({best_model_name})", "d"),
        (ax2, cm_norm, f"Confusion Matrix - Normalized ({best_model_name})", ".2%"),
    ]:
        im = ax.imshow(data, cmap="Blues", interpolation="nearest")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(class_names)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        thresh = data.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if data[i, j] > thresh else "black", fontsize=12, fontweight="bold")
                
    plt.suptitle(f"Best Model Performance: {best_model_name} ({representation_name} Features)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(cfg.FIG_DIR, f"classical_{representation_name.lower()}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    
    print(f"  -> Curves saved: {curves_path}")
    print(f"  -> Confusion Matrix saved: {cm_path}")
    return df_res

# Run evaluations for both feature sets
df_onehot_res = train_and_eval_baselines(X_onehot, all_labels, "OneHot")
df_tfidf_res = train_and_eval_baselines(X_tfidf, all_labels, "TFIDF")

# ═══════════════════════════════════════════════════════════════════════
# CROSS-REPRESENTATION SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. CROSS-REPRESENTATION ANALYSIS SUMMARY")
print("=" * 60)
best_oh_model = df_onehot_res.index[0]
best_oh_acc = df_onehot_res.iloc[0]["Accuracy"]
best_tf_model = df_tfidf_res.index[0]
best_tf_acc = df_tfidf_res.iloc[0]["Accuracy"]

print(f"  [Best One-Hot Model] : {best_oh_model:<22} | Accuracy: {best_oh_acc:.4%}")
print(f"  [Best TF-IDF Model]  : {best_tf_model:<22} | Accuracy: {best_tf_acc:.4%}")

# Zip outputs for easy download
import shutil
zip_filename = "outputs_binary_classical_ml"
shutil.make_archive(zip_filename, 'zip', cfg.OUTPUT_DIR)
print(f"\nAll outputs zipped into: {zip_filename}.zip")
