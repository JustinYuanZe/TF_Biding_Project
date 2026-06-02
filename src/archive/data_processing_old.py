import argparse
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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
from sklearn.metrics import accuracy_score

def load_fasta_seqs(fasta_path):
    """Load sequences from a FASTA file."""
    sequences = []
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"Dataset file not found: {fasta_path}")
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('>'):
                sequences.append(line.upper())
    return sequences

def seqs_to_onehot(sequences):
    """Convert DNA sequences to flattened one-hot representation."""
    n_samples = len(sequences)
    seq_array = np.array([list(seq) for seq in sequences], dtype='U1')
    onehot = np.zeros((n_samples, 101, 4), dtype=np.int8)
    onehot[seq_array == 'A', 0] = 1
    onehot[seq_array == 'C', 1] = 1
    onehot[seq_array == 'G', 2] = 1
    onehot[seq_array == 'T', 3] = 1
    return onehot.reshape(n_samples, -1)

def extract_kmer_features(sequences, ngram_range=(4, 6), vectorizer_type='tfidf'):
    """Extract k-mer features using CountVectorizer or TfidfVectorizer."""
    if vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=ngram_range)
    else:
        vectorizer = CountVectorizer(analyzer='char', ngram_range=ngram_range)
    return vectorizer.fit_transform(sequences)

def train_and_evaluate(X, y):
    """Train and evaluate 10 classical ML models."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "SVM (Linear)": SVC(kernel='linear', random_state=42),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Naive Bayes": BernoulliNB(),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    }

    print("Training 10 models...")
    for name, model in models.items():
        # Clean warning for jobs
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        elapsed = time.time() - start_time
        print(f"{name:<20} | Accuracy: {acc:.4f} | Time: {elapsed:.1f}s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SP1/SP2/SP4 classical ML baselines")
    parser.add_argument(
        '--task',
        type=str,
        default='binary',
        choices=['binary', '4class'],
        help="Classification task: 'binary' (SP1 vs Negative) or '4class' (SP1, SP2, SP4, Neg)"
    )
    parser.add_argument(
        '--features',
        type=str,
        default='tfidf',
        choices=['onehot', 'count', 'tfidf'],
        help="Feature representation: 'onehot' (flattened one-hot), 'count' (k-mer count), or 'tfidf' (k-mer TF-IDF)"
    )
    parser.add_argument(
        '--k_min',
        type=int,
        default=4,
        help="Minimum size of k-mers (default: 4)"
    )
    parser.add_argument(
        '--k_max',
        type=int,
        default=6,
        help="Maximum size of k-mers (default: 6)"
    )
    args = parser.parse_args()

    # Define paths relative to the project root
    sp1_path = os.path.join("data", "processed", "sp1_positive_final.fasta")
    sp2_path = os.path.join("data", "processed", "sp2_positive_final.fasta")
    sp4_path = os.path.join("data", "processed", "sp4_positive_final.fasta")
    neg_path = os.path.join("data", "processed", "negative_final.fasta")

    print(f"Loading data for task: {args.task}...")
    if args.task == 'binary':
        # SP1 Positive vs Negative (balanced 4830 vs 4830)
        seqs_pos = load_fasta_seqs(sp1_path)
        seqs_neg = load_fasta_seqs(neg_path)
        sequences = seqs_pos + seqs_neg
        y = np.concatenate([np.ones(len(seqs_pos)), np.zeros(len(seqs_neg))], axis=0)
    else:
        # 4-class classification
        seqs_sp1 = load_fasta_seqs(sp1_path)
        seqs_sp2 = load_fasta_seqs(sp2_path)
        seqs_sp4 = load_fasta_seqs(sp4_path)
        seqs_neg = load_fasta_seqs(neg_path)
        sequences = seqs_sp1 + seqs_sp2 + seqs_sp4 + seqs_neg
        y = np.concatenate([
            np.zeros(len(seqs_sp1)),
            np.ones(len(seqs_sp2)),
            np.full(len(seqs_sp4), 2),
            np.full(len(seqs_neg), 3)
        ], axis=0)

    print(f"Extracting features: {args.features}...")
    if args.features == 'onehot':
        X = seqs_to_onehot(sequences)
    else:
        X = extract_kmer_features(
            sequences,
            ngram_range=(args.k_min, args.k_max),
            vectorizer_type=args.features
        )

    print(f"Data shape: {X.shape}")
    train_and_evaluate(X, y)
