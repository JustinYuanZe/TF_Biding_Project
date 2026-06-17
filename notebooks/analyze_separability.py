"""
analyze_separability.py
=======================

REAL CPU analysis quantifying how separable the positive and negative classes
are for the SP1/SP2/SP4 TF-binding project, and whether the engineered bio
features carry any UNIVARIATE signal.

This is the evidence backing the paper's honest Discussion:
  - The dinucleotide-shuffled negatives preserve mono/dinucleotide composition
    exactly, so composition-based bio features (GC, CpG O/E) should be
    NON-discriminative (univariate ROC-AUC ~ 0.5).
  - Real negatives (genomic-matched, CpG-promoter) may differ, so we report
    them separately when present.
  - A k-mer (char n-gram 4-6) TF-IDF + Logistic Regression classifier
    quantifies the SEQUENCE-LEVEL separability that a model could exploit for
    each negative type.

Outputs (under outputs_separability/):
  - separability_report.txt   (full numeric report)
  - bio_feature_auc.png       (bar chart of the 3 bio-feature univariate AUCs)

CPU only. No GPU / DNABERT training here.
"""

import os
import re
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
OUT_DIR = os.path.join(ROOT, "outputs_separability")
os.makedirs(OUT_DIR, exist_ok=True)

POS_FILES = [
    os.path.join(PROC, "sp1_positive_final.fasta"),
    os.path.join(PROC, "sp2_positive_final.fasta"),
    os.path.join(PROC, "sp4_positive_final.fasta"),
]
DINUC_NEG = os.path.join(PROC, "negative_final.fasta")

# Auto-detect real negatives across the known candidate locations.
GENOMIC_CANDIDATES = [
    os.path.join(PROC, "negative_genomic_matched.fasta"),
    os.path.join(PROC, "FINAL", "datas1", "negative_genomic_matched.fasta"),
    os.path.join(PROC, "fixed_negative", "negative_genomic_matched.fasta"),
]
CPG_CANDIDATES = [
    os.path.join(PROC, "negative_promoter_cpg.fasta"),
    os.path.join(PROC, "fixed_negative", "negative_promoter_cpg.fasta"),
    os.path.join(PROC, "FINAL", "datas1", "negative_promoter_cpg.fasta"),
]


def first_existing(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


# --------------------------------------------------------------------------- #
# FASTA loading
# --------------------------------------------------------------------------- #
def load_fasta(path):
    """Return list of uppercase sequences from a FASTA file."""
    seqs = []
    cur = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur).upper())
                    cur = []
            else:
                cur.append(line)
    if cur:
        seqs.append("".join(cur).upper())
    return seqs


# --------------------------------------------------------------------------- #
# Bio features
# --------------------------------------------------------------------------- #
G4_RE = re.compile(r"(G{3,}[ACGTN]{1,7}){3,}G{3,}")


def cpg_oe(seq):
    """CpG observed/expected = (N_CG * L) / (N_C * N_G)."""
    L = len(seq)
    n_c = seq.count("C")
    n_g = seq.count("G")
    n_cg = seq.count("CG")
    if n_c == 0 or n_g == 0:
        return 0.0
    return (n_cg * L) / (n_c * n_g)


def gc_content(seq):
    L = len(seq)
    if L == 0:
        return 0.0
    return (seq.count("C") + seq.count("G")) / L


def g4_flag(seq):
    return 1.0 if G4_RE.search(seq) else 0.0


def compute_bio(seqs):
    cpg = np.array([cpg_oe(s) for s in seqs], dtype=np.float64)
    gc = np.array([gc_content(s) for s in seqs], dtype=np.float64)
    g4 = np.array([g4_flag(s) for s in seqs], dtype=np.float64)
    return {"CpG O/E": cpg, "GC content": gc, "G4 presence": g4}


# --------------------------------------------------------------------------- #
# Univariate ROC-AUC for a single feature (raw feature as score).
# roc_auc_score is symmetric: AUC and 1-AUC are equivalent strength; we report
# the directed AUC plus its distance from 0.5 for interpretability.
# --------------------------------------------------------------------------- #
def univariate_auc(pos_vals, neg_vals):
    y = np.concatenate([np.ones(len(pos_vals)), np.zeros(len(neg_vals))])
    score = np.concatenate([pos_vals, neg_vals])
    if np.all(score == score[0]):  # degenerate constant feature
        return 0.5
    return roc_auc_score(y, score)


# --------------------------------------------------------------------------- #
# k-mer TF-IDF + Logistic Regression separability
# --------------------------------------------------------------------------- #
def kmer_logreg(pos_seqs, neg_seqs, seed=42):
    X = list(pos_seqs) + list(neg_seqs)
    y = np.concatenate([np.ones(len(pos_seqs)), np.zeros(len(neg_seqs))])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char", ngram_range=(4, 6), lowercase=False, min_df=2
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000, C=1.0)),
        ]
    )
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_te, pred)
    auc = roc_auc_score(y_te, proba)
    return acc, auc, len(X_tr), len(X_te)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 78)
    emit("SEPARABILITY & UNIVARIATE BIO-FEATURE SIGNAL ANALYSIS")
    emit("SP1/SP2/SP4 TF-binding project")
    emit("=" * 78)

    # ---- Load positives -------------------------------------------------- #
    pos_seqs = []
    for f in POS_FILES:
        s = load_fasta(f)
        pos_seqs.extend(s)
        emit(f"[load] {os.path.basename(f):32s} {len(s):6d} records")
    emit(f"[load] POSITIVE total                    {len(pos_seqs):6d} records")
    emit("")

    # ---- Load negatives -------------------------------------------------- #
    neg_sets = {}  # name -> list[str]
    dinuc = load_fasta(DINUC_NEG)
    neg_sets["dinuc"] = dinuc
    emit(f"[load] dinuc negatives (negative_final)   {len(dinuc):6d} records")

    gpath = first_existing(GENOMIC_CANDIDATES)
    if gpath:
        g = load_fasta(gpath)
        neg_sets["genomic"] = g
        emit(f"[load] genomic-matched negatives          {len(g):6d} records  "
             f"({os.path.relpath(gpath, ROOT)})")
    else:
        emit("[load] genomic-matched negatives          NOT FOUND (skipped)")

    cpath = first_existing(CPG_CANDIDATES)
    if cpath:
        c = load_fasta(cpath)
        neg_sets["cpg_promoter"] = c
        emit(f"[load] CpG-promoter negatives             {len(c):6d} records  "
             f"({os.path.relpath(cpath, ROOT)})")
    else:
        emit("[load] CpG-promoter negatives             NOT FOUND (skipped)")
    emit("")

    # ---- Bio features ---------------------------------------------------- #
    emit("-" * 78)
    emit("1) BIO-FEATURE DESCRIPTIVE STATS (mean +/- std)")
    emit("-" * 78)
    pos_bio = compute_bio(pos_seqs)
    neg_bio = {name: compute_bio(seqs) for name, seqs in neg_sets.items()}

    feat_names = ["CpG O/E", "GC content", "G4 presence"]
    hdr = f"{'feature':14s} | {'POSITIVE':>18s}"
    for name in neg_sets:
        hdr += f" | {('NEG:' + name):>18s}"
    emit(hdr)
    emit("-" * len(hdr))
    for fn in feat_names:
        row = f"{fn:14s} | {pos_bio[fn].mean():8.4f} +/-{pos_bio[fn].std():7.4f}"
        for name in neg_sets:
            v = neg_bio[name][fn]
            row += f" | {v.mean():8.4f} +/-{v.std():7.4f}"
        emit(row)
    # G4 present fraction explicitly
    emit("")
    g4line = f"{'G4 % present':14s} | {100*pos_bio['G4 presence'].mean():16.2f}%"
    for name in neg_sets:
        g4line += f" | {100*neg_bio[name]['G4 presence'].mean():16.2f}%"
    emit(g4line)
    emit("")

    # ---- Univariate ROC-AUC --------------------------------------------- #
    emit("-" * 78)
    emit("2) UNIVARIATE ROC-AUC  (positive vs negative; 0.50 = non-discriminative)")
    emit("-" * 78)
    auc_table = {}  # neg_name -> {feat: auc}
    for neg_name in neg_sets:
        auc_table[neg_name] = {}
    hdr = f"{'feature':14s}"
    for name in neg_sets:
        hdr += f" | {('vs ' + name):>16s}"
    emit(hdr)
    emit("-" * len(hdr))
    for fn in feat_names:
        row = f"{fn:14s}"
        for name in neg_sets:
            auc = univariate_auc(pos_bio[fn], neg_bio[name][fn])
            auc_table[name][fn] = auc
            dist = abs(auc - 0.5)
            row += f" | {auc:8.4f} (d{dist:5.3f})"
        emit(row)
    emit("")
    emit("Interpretation: AUC near 0.50 -> feature carries NO univariate signal")
    emit("for that negative type. d = |AUC - 0.50| (effect size, direction-free).")
    emit("")

    # ---- k-mer separability --------------------------------------------- #
    emit("-" * 78)
    emit("3) SEQUENCE-LEVEL SEPARABILITY: k-mer TF-IDF (char 4-6) + LogisticReg")
    emit("    stratified 80/20 split, seed=42")
    emit("-" * 78)
    kmer_results = {}
    for neg_name, neg_seqs in neg_sets.items():
        acc, auc, n_tr, n_te = kmer_logreg(pos_seqs, neg_seqs)
        kmer_results[neg_name] = (acc, auc)
        emit(f"  pos vs {neg_name:14s}: test acc = {acc:.4f}   "
             f"test ROC-AUC = {auc:.4f}   (train={n_tr}, test={n_te})")
    emit("")
    emit("Interpretation: higher acc/AUC => the negative type is MORE separable")
    emit("from positives at the raw-sequence level (k-mer composition).")
    emit("")

    # ---- Summary --------------------------------------------------------- #
    emit("=" * 78)
    emit("SUMMARY")
    emit("=" * 78)
    emit("Bio-feature univariate AUC (vs dinuc-shuffled negatives):")
    for fn in feat_names:
        emit(f"   {fn:14s}: {auc_table['dinuc'][fn]:.4f}")
    if "genomic" in auc_table:
        emit("Bio-feature univariate AUC (vs genomic-matched negatives):")
        for fn in feat_names:
            emit(f"   {fn:14s}: {auc_table['genomic'][fn]:.4f}")
    if "cpg_promoter" in auc_table:
        emit("Bio-feature univariate AUC (vs CpG-promoter negatives):")
        for fn in feat_names:
            emit(f"   {fn:14s}: {auc_table['cpg_promoter'][fn]:.4f}")
    emit("")
    emit("k-mer LogReg separability (test acc / ROC-AUC):")
    for neg_name, (acc, auc) in kmer_results.items():
        emit(f"   pos vs {neg_name:14s}: acc={acc:.4f}  AUC={auc:.4f}")
    emit("")

    # ---- Save report ----------------------------------------------------- #
    report_path = os.path.join(OUT_DIR, "separability_report.txt")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    emit(f"[saved] report -> {report_path}")

    # ---- Bar chart of bio-feature AUCs (vs dinuc) ----------------------- #
    fig, ax = plt.subplots(figsize=(7, 4.5))
    aucs_dinuc = [auc_table["dinuc"][fn] for fn in feat_names]
    x = np.arange(len(feat_names))
    width = 0.8 / max(1, len(neg_sets))
    colors = {"dinuc": "#4C72B0", "genomic": "#C44E52", "cpg_promoter": "#55A868"}
    for i, name in enumerate(neg_sets):
        vals = [auc_table[name][fn] for fn in feat_names]
        ax.bar(x + i * width, vals, width, label=f"vs {name}",
               color=colors.get(name, None))
    ax.axhline(0.5, ls="--", color="k", lw=1, label="0.50 (no signal)")
    ax.set_xticks(x + width * (len(neg_sets) - 1) / 2)
    ax.set_xticklabels(feat_names)
    ax.set_ylabel("Univariate ROC-AUC")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Univariate bio-feature discriminative power\n(positive vs negative)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    chart_path = os.path.join(OUT_DIR, "bio_feature_auc.png")
    fig.savefig(chart_path, dpi=130)
    plt.close(fig)
    emit(f"[saved] chart  -> {chart_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
