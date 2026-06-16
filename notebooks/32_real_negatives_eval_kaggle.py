#!/usr/bin/env python3
"""
Script 32: REAL-Negatives Evaluation of the Binary Tri-Branch G-CMAB msCNN
Designed for: Kaggle GPU (1xT4 or 2xT4 via HuggingFace Accelerate)
Task: Binary SP_Positive / Negative TF-binding classification

PURPOSE (the question this script answers)
------------------------------------------
The proposed Tri-Branch model reaches ~96.35% accuracy against
DINUCLEOTIDE-SHUFFLED negatives. Dinucleotide shuffling PRESERVES GC content
and CpG O/E *exactly*, so a model that keys on those bio-features could be
inflated by an easy, artificial contrast. This script re-runs the SAME proven
architecture / training / evaluation / SHAP, but swaps the negatives for REAL
genomic sequences:

  * negative_genomic_matched.fasta  (GC/length-matched real genomic loci), or
  * negative_promoter_cpg.fasta     (real CpG-island / promoter loci).

If accuracy and especially NEGATIVE-RECALL hold near the 96.35% dinuc figure,
the gain is real. A large drop (e.g. Neg-recall collapsing) would indicate the
dinuc-shuffle figure is partly an artifact. The script prints an explicit
"ARTIFACT CHECK" verdict comparing to the 96.35% baseline.

This file is a faithful copy of notebooks/28_binary_tribranch_shap_kaggle.py.
The ONLY substantive change is the negative-data SOURCE selection (Config.NEG_KIND
with auto-detection) plus an output dir of outputs_real_negatives/ and the
artifact-check summary. Architecture, training loop, evaluation, and the inline
tri-modal SHAP are unchanged.

ARCHITECTURE (unchanged, proven) — G-CMAB Tri-Branch:
  1. Sequence : DNABERT-2 (last 6 layers) + ELMo scalar-mix -> proj 128
  2. Shape    : 1x1 Conv proj (5 -> 128) + GroupNorm
  3. Bio      : MLP (3 -> 16 -> 32)  [CpG O/E, GC content, G4 motif]
  Seq<->Shape coupled by bidirectional Cross-Modal Attention, then multi-scale
  depthwise-separable CNN (msCNN) + 1-Max pooling; fused with Bio -> sigmoid head.
"""

# ═══════════════════════════════════════════════════════════════════════
# CELL 0: Install Dependencies
# ═══════════════════════════════════════════════════════════════════════

import subprocess
import sys

def install_packages():
    """Install required packages that are not pre-installed on Kaggle."""
    packages = [
        "transformers>=4.37.0",
        "einops>=0.7.0",
        "datasets>=2.16.0",
        "accelerate>=0.25.0",
        "safetensors>=0.4.0",
        "shap>=0.44.0",
        "seaborn>=0.12.0",
    ]
    for pkg in packages:
        try:
            pkg_name = pkg.split(">=")[0].split("==")[0]
            __import__(pkg_name)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install_packages()

# ═══════════════════════════════════════════════════════════════════════
# CELL 1: Imports
# ═══════════════════════════════════════════════════════════════════════

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import math
import time
import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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
    recall_score,
)

# Custom import hook: force DNABERT-2 to fall back to pure-PyTorch attention
# (Triton flash-attn is unavailable / unstable on Kaggle T4).
import builtins

orig_import = builtins.__import__

def custom_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'triton' or name.startswith('triton.'):
        try:
            frame = sys._getframe(1)
            while frame:
                filename = frame.f_code.co_filename
                if 'flash_attn_triton' in filename or 'bert_layers' in filename:
                    raise ImportError("Forced fallback for DNABERT-2 custom Triton attention")
                frame = frame.f_back
        except ImportError:
            raise
        except Exception:
            pass
    return orig_import(name, globals, locals, fromlist, level)

builtins.__import__ = custom_import

from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ═══════════════════════════════════════════════════════════════════════
# CELL 2: Configuration
# ═══════════════════════════════════════════════════════════════════════

def find_file(filename: str, fallback_dir: str = "data/processed"):
    """Search for filename in absolute paths, Kaggle input, fallback dirs, or CWD.

    Searches a few well-known sub-locations used in this project so that the
    real-negative files (which live under FINAL/ or fixed_negative/) are found
    regardless of which top-level dir is passed in.
    """
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if filename in files:
                fpath = os.path.join(root, filename)
                print(f"  [Auto-detect] Found {filename} at {fpath}")
                return fpath
    if fallback_dir and os.path.exists(fallback_dir):
        # Direct hit first
        p1 = os.path.join(fallback_dir, filename)
        if os.path.exists(p1):
            return p1
        # Known project sub-locations for the alternative real negatives
        for sub in ["fixed_negative",
                    os.path.join("FINAL", "datas1"),
                    os.path.join("FINAL", "datashape"),
                    "FINAL"]:
            p = os.path.join(fallback_dir, sub, filename)
            if os.path.exists(p):
                return p
        # Last resort: recursive walk
        for root, _, files in os.walk(fallback_dir):
            if filename in files:
                return os.path.join(root, filename)
    if os.path.exists(filename):
        return filename
    return None


def auto_detect_dir(target_file: str, fallback: str = "data/processed") -> str:
    resolved_path = find_file(target_file, fallback)
    if resolved_path:
        return os.path.dirname(resolved_path)
    return fallback


# ── REAL-NEGATIVE AUTO-DETECTION ────────────────────────────────────────
# Priority order: genomic-matched (most stringent control) > promoter/CpG >
# dinucleotide-shuffled (the artifact-prone baseline, used only as a fallback
# so the script still runs if no real negatives are present).
NEG_SOURCES = [
    # (kind, fasta_filename, shape_filename, human description)
    ("genomic", "negative_genomic_matched.fasta", "dnashape_negative_genomic.npy",
     "GC/length-matched REAL genomic loci"),
    ("cpg", "negative_promoter_cpg.fasta", "dnashape_negative_cpg.npy",
     "REAL CpG-island / promoter loci"),
    ("dinuc", "negative_final.fasta", "dnashape_negative.npy",
     "dinucleotide-shuffled (ARTIFACT-PRONE fallback)"),
]


def detect_negative_source(fasta_search_dir: str, shape_search_dir: str):
    """Return (kind, fasta_path, shape_path, desc) for the highest-priority
    negative source whose BOTH files (FASTA + DNAshape) exist. None if none."""
    for kind, fa, sh, desc in NEG_SOURCES:
        fa_path = find_file(fa, fasta_search_dir)
        sh_path = find_file(sh, shape_search_dir)
        if fa_path and sh_path:
            return kind, fa_path, sh_path, desc
    return None


class Config:
    """Script 32: Real-Negatives eval of the Binary Tri-Branch G-CMAB config."""

    # ── Paths ──
    FASTA_DIR = auto_detect_dir("sp1_positive_final.fasta", "data/processed")
    SHAPE_DIR = auto_detect_dir("dnashape_sp1.npy", "data/processed")
    OUTPUT_DIR = "outputs_real_negatives"
    FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

    # ── Negative source (auto-detected below; one of 'genomic'|'cpg'|'dinuc') ──
    # Set to a fixed kind to force a specific source; leave as "auto" to let the
    # priority detector choose genomic > cpg > dinuc based on file availability.
    NEG_KIND = "auto"
    # Baseline figure to compare against (proposed model vs dinuc-shuffle).
    DINUC_BASELINE_ACC = 0.9635

    # ── DNABERT-2 ──
    DNABERT_MODEL = "zhihan1996/DNABERT-2-117M"
    EMBEDDING_DIM = 768

    # ── Sequence length (auto from data) ──
    AUTO_MAX_LENGTH = True
    MAX_TOKEN_LENGTH = 48
    MAX_LENGTH_CAP = 96
    MAX_LENGTH_FLOOR = 32

    # ── Fine-Tuning ──
    UNFREEZE_LAST_N_LAYERS = 6
    BACKBONE_LR = 2e-5

    # ── Cross-Modal Attention ──
    USE_CROSS_ATTN = True
    CROSS_ATTN_D_MODEL = 128
    CROSS_ATTN_NHEAD = 4
    CROSS_ATTN_LAYERS = 1
    CROSS_ATTN_DROPOUT = 0.1
    CROSS_ATTN_LR = 2e-4
    SEQ_PROJ_LR = 2e-4

    # ── DNAshape branch ──
    SHAPE_CHANNELS = 5
    SHAPE_SEQ_LEN = 101
    SHAPE_LR = 3e-4

    # ── msCNN ──
    MSCNN_OUT_CHANNELS = 256
    SEQ_MSCNN_KERNELS = [7, 9, 11, 15]
    SHAPE_MSCNN_KERNELS = [4, 8, 12, 16]

    # ── Safe flags ──
    USE_GROUPNORM = True
    GROUPNORM_GROUPS = 16
    USE_LAYER_ATTN = True
    LAYER_ATTN_N = 6
    LAYER_ATTN_LR = 1e-3

    # ── Classifier head ──
    HIDDEN_DIM = 256
    FUSION_DROPOUT = 0.7
    HEAD_LR = 2e-4

    # ── Loss / regularization ──
    NUM_CLASSES = 2          # binary -> single logit head
    WEIGHT_DECAY = 0.1

    # ── Training ──
    BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = 1
    EPOCHS = 30
    PATIENCE = 12
    MAX_OVERFITTING_GAP = 30.0
    WARMUP_RATIO = 0.15
    MAX_GRAD_NORM = 0.5

    # ── Split ──
    TEST_SIZE = 0.2
    RANDOM_SEED = 42

    CLASS_NAMES = ["Negative", "SP_Positive"]

    SHAPE_FILES = {
        "SP1": "dnashape_sp1.npy",
        "SP2": "dnashape_sp2.npy",
        "SP4": "dnashape_sp4.npy",
        "Negative": "dnashape_negative.npy",   # placeholder; overwritten by detected source
    }

    # ── SHAP ──
    SHAP_BG_SIZE = 50
    SHAP_NUM_EXPLAIN = 20
    SHAP_WINDOW = 15          # half-window (bp) around GC-box center

cfg = Config()

for d in [cfg.OUTPUT_DIR, cfg.FIG_DIR, cfg.MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

random.seed(cfg.RANDOM_SEED)
np.random.seed(cfg.RANDOM_SEED)
torch.manual_seed(cfg.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.RANDOM_SEED)

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=cfg.GRAD_ACCUM_STEPS,
    kwargs_handlers=[ddp_kwargs],
)
DEVICE = accelerator.device

if not accelerator.is_main_process:
    builtins.print = lambda *args, **kwargs: None

# ── Resolve which negatives to use (genomic > cpg > dinuc) ──────────────
_detected = detect_negative_source(cfg.FASTA_DIR, cfg.SHAPE_DIR)

if _detected is None:
    # No negative source at all (not even dinuc). Cannot proceed.
    print("=" * 70)
    print("REAL-NEGATIVE EVALUATION — NO NEGATIVE DATASET FOUND")
    print("=" * 70)
    print("Could not locate ANY negative FASTA + DNAshape pair among:")
    for kind, fa, sh, desc in NEG_SOURCES:
        print(f"    [{kind:7s}] {fa}  +  {sh}   ({desc})")
    print("\nReal genomic negatives must be generated first, e.g.:")
    print("    python src/generate_negatives_v2.py")
    print("This produces negative_genomic_matched.fasta / negative_promoter_cpg.fasta")
    print("and the matching DNAshape arrays. Re-run this script afterwards.")
    print("Exiting gracefully (no crash).")
    sys.exit(0)

NEG_KIND, NEG_FASTA_PATH, NEG_SHAPE_PATH, NEG_DESC = _detected

# Honor an explicit, non-"auto" NEG_KIND override if that source is available.
if cfg.NEG_KIND != "auto":
    forced = next((s for s in NEG_SOURCES if s[0] == cfg.NEG_KIND), None)
    if forced is not None:
        fa_path = find_file(forced[1], cfg.FASTA_DIR)
        sh_path = find_file(forced[2], cfg.SHAPE_DIR)
        if fa_path and sh_path:
            NEG_KIND, NEG_FASTA_PATH, NEG_SHAPE_PATH, NEG_DESC = (
                forced[0], fa_path, sh_path, forced[3])
        else:
            print(f"  [WARN] Forced NEG_KIND='{cfg.NEG_KIND}' files missing; "
                  f"falling back to auto-detected '{NEG_KIND}'.")

cfg.NEG_KIND = NEG_KIND
# Point the shape-loader at the detected negative DNAshape file by name.
cfg.SHAPE_FILES["Negative"] = os.path.basename(NEG_SHAPE_PATH)

IS_REAL_NEG = NEG_KIND in ("genomic", "cpg")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    n_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {n_gpus}")
print(f"\nUsing device: {DEVICE}")
print(f"Num processes: {accelerator.num_processes}")
print("Architecture: Binary Tri-Branch G-CMAB msCNN (Seq + Shape + Bio) + SHAP")
print(f"  Cross-Attention: {cfg.USE_CROSS_ATTN} | GroupNorm: {cfg.USE_GROUPNORM} | LayerAttn: {cfg.USE_LAYER_ATTN}")
print(f"  Seq msCNN kernels: {cfg.SEQ_MSCNN_KERNELS} | Shape msCNN kernels: {cfg.SHAPE_MSCNN_KERNELS}")
print("-" * 60)
print("NEGATIVE SOURCE FOR THIS RUN (artifact check):")
print(f"  NEG_KIND : {NEG_KIND}  ({NEG_DESC})")
print(f"  FASTA    : {NEG_FASTA_PATH}")
print(f"  DNAshape : {NEG_SHAPE_PATH}")
if IS_REAL_NEG:
    print("  -> Using REAL genomic negatives. Comparing to the "
          f"{cfg.DINUC_BASELINE_ACC:.2%} dinuc-shuffle figure.")
else:
    print("  -> WARNING: only dinuc-shuffled negatives were found; this run "
          "CANNOT test for shuffle artifact.")
    print("     Generate real negatives via: python src/generate_negatives_v2.py")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════════════
# CELL 3: Data Loading (Sequences + DNAshape)
# ═══════════════════════════════════════════════════════════════════════

def load_fasta(filepath):
    sequences, headers = [], []
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing dataset: {filepath}")
    with open(filepath, "r") as f:
        seq_lines, current_header = [], None
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


def load_shape_features(data_dir, shape_files, neg_shape_path):
    """Load DNAshape for SP1/SP2/SP4 plus the EXPLICITLY chosen negative file.

    Unlike script 28 (which auto-picks the first negative shape file it finds),
    we pin the negative shape to the detected real-negative source so the
    sequences and shapes stay consistent for the artifact check.
    """
    all_shapes = []
    if accelerator.is_main_process:
        print(f"  [Negatives] Using DNAshape file: {neg_shape_path}")

    resolved_paths = {}
    for cls_name, fname in shape_files.items():
        if cls_name == "Negative":
            resolved_paths[cls_name] = neg_shape_path
        else:
            path = find_file(fname, data_dir)
            if not path:
                raise FileNotFoundError(f"Missing DNAshape file for {cls_name}: {fname}")
            resolved_paths[cls_name] = path

    for cls_name, fpath in resolved_paths.items():
        shape_data = np.load(fpath)
        if accelerator.is_main_process:
            print(f"  {cls_name} shape: {shape_data.shape} ({os.path.basename(fpath)})")
        all_shapes.append(shape_data)
    return np.concatenate(all_shapes, axis=0)


def load_all_data(fasta_dir, shape_dir, shape_files, neg_fasta_path, neg_shape_path):
    if accelerator.is_main_process:
        print("=" * 60)
        print("LOADING DATASETS (Sequences + DNAshape)")
        print("=" * 60)
        print(f"  [Negatives] Using FASTA file: {neg_fasta_path}")

    fasta_files = {
        "SP1": find_file("sp1_positive_final.fasta", fasta_dir),
        "SP2": find_file("sp2_positive_final.fasta", fasta_dir),
        "SP4": find_file("sp4_positive_final.fasta", fasta_dir),
        "Negative": neg_fasta_path,
    }
    for cls_name, fpath in fasta_files.items():
        if not fpath:
            raise FileNotFoundError(f"Missing FASTA file for {cls_name}")

    all_sequences, all_headers, all_labels, all_groups = [], [], [], []
    group_id = 0
    for cls_idx, (cls_name, fpath) in enumerate(fasta_files.items()):
        seqs, hdrs = load_fasta(fpath)
        if accelerator.is_main_process:
            print(f"  {cls_name}: {len(seqs)} sequences ({os.path.basename(fpath)})")
        all_sequences.extend(seqs)
        all_headers.extend(hdrs)
        all_labels.extend([cls_idx] * len(seqs))
        # Positives are stored as (original, revcomp) pairs -> same group; negatives singletons
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

    if accelerator.is_main_process:
        print("\n  Loading DNAshape features...")
    all_shapes = load_shape_features(shape_dir, shape_files, neg_shape_path)

    assert len(all_sequences) == all_shapes.shape[0], (
        f"Sequence count ({len(all_sequences)}) != shape count ({all_shapes.shape[0]})"
    )
    if accelerator.is_main_process:
        print(f"\n  Total: {len(all_sequences)} sequences, {group_id} groups")
        print(f"  Shape features: {all_shapes.shape}")
        print(f"  Class distribution (SP1,SP2,SP4,Neg): {np.bincount(all_labels)}")
    return all_sequences, all_labels, all_groups, all_shapes, all_headers


def split_data(sequences, labels, groups, shapes, headers, test_size=0.2, seed=42):
    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("SPLITTING DATA (GroupShuffleSplit — no revcomp leakage)")
        print("=" * 60)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(sequences, labels, groups))
    seq_train = [sequences[i] for i in train_idx]
    seq_test = [sequences[i] for i in test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    shape_train, shape_test = shapes[train_idx], shapes[test_idx]
    headers_train = [headers[i] for i in train_idx]
    headers_test = [headers[i] for i in test_idx]
    if accelerator.is_main_process:
        print(f"  Train: {len(seq_train)} | Test: {len(seq_test)}")
    return seq_train, seq_test, y_train, y_test, shape_train, shape_test, headers_train, headers_test


all_sequences, all_labels, all_groups, all_shapes, all_headers = load_all_data(
    cfg.FASTA_DIR, cfg.SHAPE_DIR, cfg.SHAPE_FILES, NEG_FASTA_PATH, NEG_SHAPE_PATH
)
seq_train, seq_test, y_train, y_test, shape_train, shape_test, headers_train, headers_test = split_data(
    all_sequences, all_labels, all_groups, all_shapes, all_headers,
    test_size=cfg.TEST_SIZE, seed=cfg.RANDOM_SEED,
)
del all_sequences, all_labels, all_groups, all_shapes, all_headers
gc.collect()

# ═══════════════════════════════════════════════════════════════════════
# CELL 4: DNAshape Normalization — Robust Scaler (P1-P99), stats from TRAIN only
# ═══════════════════════════════════════════════════════════════════════

def robust_normalize_shapes(shape_train, shape_test):
    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("NORMALIZING DNAshape (Robust Scaler P1-P99)")
        print("=" * 60)
    n_channels = shape_train.shape[1]
    channel_names = ["MGW", "ProT", "Roll", "HelT", "EP"]
    shape_train_norm = np.copy(shape_train).astype(np.float32)
    shape_test_norm = np.copy(shape_test).astype(np.float32)
    for ch in range(n_channels):
        train_vals = shape_train[:, ch, :].flatten()
        valid_vals = train_vals[~np.isnan(train_vals)]
        median_val = np.median(valid_vals)
        p1_val, p99_val = np.percentile(valid_vals, 1), np.percentile(valid_vals, 99)
        scale = max(p99_val - p1_val, 1e-9)
        shape_train_norm[:, ch, :] = (shape_train_norm[:, ch, :] - median_val) / scale
        shape_test_norm[:, ch, :] = (shape_test_norm[:, ch, :] - median_val) / scale
        if accelerator.is_main_process:
            print(f"  {channel_names[ch]:>5s}: median={median_val:>8.4f}, P1={p1_val:>8.4f}, P99={p99_val:>8.4f}")
    nan_train = np.isnan(shape_train_norm).sum()
    nan_test = np.isnan(shape_test_norm).sum()
    shape_train_norm = np.nan_to_num(shape_train_norm, nan=0.0)
    shape_test_norm = np.nan_to_num(shape_test_norm, nan=0.0)
    if accelerator.is_main_process:
        print(f"\n  NaN filled with 0: train={nan_train}, test={nan_test}")
    return shape_train_norm, shape_test_norm


shape_train_norm, shape_test_norm = robust_normalize_shapes(shape_train, shape_test)
del shape_train, shape_test
gc.collect()

# ═══════════════════════════════════════════════════════════════════════
# CELL 5: DNABERT-2 Flash-Attention Patch (pure PyTorch, no Triton)
# ═══════════════════════════════════════════════════════════════════════

def _pytorch_flash_attn_qkvpacked(qkv, bias=None, causal=False, softmax_scale=None):
    q, k, v = qkv.unbind(dim=2)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    scale = softmax_scale if softmax_scale is not None else (q.shape[-1] ** -0.5)
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    if bias is not None:
        attn = attn + bias
    if causal:
        S = q.shape[2]
        attn = attn.masked_fill(torch.triu(torch.ones(S, S, device=q.device, dtype=torch.bool), 1), float("-inf"))
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(attn, v).transpose(1, 2).contiguous()


def _pytorch_flash_attn_kvpacked(q, kv, bias=None, causal=False, softmax_scale=None):
    k, v = kv.unbind(dim=2)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    scale = softmax_scale if softmax_scale is not None else (q.shape[-1] ** -0.5)
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    if bias is not None:
        attn = attn + bias
    if causal:
        Sq, Sk = q.shape[2], k.shape[2]
        attn = attn.masked_fill(torch.triu(torch.ones(Sq, Sk, device=q.device, dtype=torch.bool), 1), float("-inf"))
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(attn, v).transpose(1, 2).contiguous()


def _pytorch_flash_attn_func(q, k, v, bias=None, causal=False, softmax_scale=None):
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    scale = softmax_scale if softmax_scale is not None else (q.shape[-1] ** -0.5)
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    if bias is not None:
        attn = attn + bias
    if causal:
        Sq, Sk = q.shape[2], k.shape[2]
        attn = attn.masked_fill(torch.triu(torch.ones(Sq, Sk, device=q.device, dtype=torch.bool), 1), float("-inf"))
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(attn, v).transpose(1, 2).contiguous()


def patch_flash_attention():
    patched = 0
    targets = {
        "flash_attn_qkvpacked_func": _pytorch_flash_attn_qkvpacked,
        "flash_attn_kvpacked_func":  _pytorch_flash_attn_kvpacked,
        "flash_attn_func":           _pytorch_flash_attn_func,
    }
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if "flash_attn_triton" in mod_name or "bert_layers" in mod_name:
            for attr_name, replacement in targets.items():
                if hasattr(mod, attr_name):
                    setattr(mod, attr_name, replacement)
                    patched += 1
    if accelerator.is_main_process:
        print(f"  Patched {patched} flash-attention refs -> pure PyTorch." if patched
              else "  No flash-attn refs to patch.")

# ═══════════════════════════════════════════════════════════════════════
# CELL 6: Load DNABERT-2 Backbone (selective unfreezing, 3 loading strategies)
# ═══════════════════════════════════════════════════════════════════════

def load_dnabert2(model_name, unfreeze_last_n=6):
    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("LOADING DNABERT-2 BACKBONE")
        print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 3
    if cfg.USE_LAYER_ATTN:
        config.output_hidden_states = True

    model = None
    # Strategy 1: direct
    try:
        model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True, low_cpu_mem_usage=False)
        for name, param in model.named_parameters():
            if param.device == torch.device("meta"):
                raise RuntimeError(f"Parameter {name} on meta device")
        for name, buf in model.named_buffers():
            if buf.device == torch.device("meta"):
                raise RuntimeError(f"Buffer {name} on meta device")
        if accelerator.is_main_process:
            print("  Strategy 1 (direct load) OK")
    except Exception as e:
        if accelerator.is_main_process:
            print(f"  Strategy 1 failed: {e}")
        model = None
    # Strategy 2: empty init + manual state_dict
    if model is None:
        try:
            from huggingface_hub import hf_hub_download
            with torch.no_grad():
                model = AutoModel.from_config(config, trust_remote_code=True)
            try:
                wf = hf_hub_download(repo_id=model_name, filename="model.safetensors")
                from safetensors.torch import load_file
                sd = load_file(wf, device="cpu")
            except Exception:
                wf = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
                sd = torch.load(wf, map_location="cpu", weights_only=False)
            clean_sd = {(k[5:] if k.startswith("bert.") else k): v for k, v in sd.items()}
            result = model.load_state_dict(clean_sd, strict=False)
            for ck in ["embeddings.word_embeddings.weight", "encoder.layer.0.attention.self.Wqkv.weight"]:
                if ck in set(result.missing_keys):
                    raise RuntimeError(f"Core weight '{ck}' missing!")
            model = model.to("cpu")
            if accelerator.is_main_process:
                print("  Strategy 2 OK")
        except Exception as e:
            if accelerator.is_main_process:
                print(f"  Strategy 2 failed: {e}")
            model = None
    # Strategy 3: monkey-patch torch.empty meta->cpu (ALiBi safe)
    if model is None:
        try:
            _orig = torch.empty
            def _patched(*a, **kw):
                if kw.get("device") == torch.device("meta") or str(kw.get("device", "")) == "meta":
                    kw["device"] = "cpu"
                return _orig(*a, **kw)
            torch.empty = _patched
            try:
                model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True, low_cpu_mem_usage=False)
            finally:
                torch.empty = _orig
            if accelerator.is_main_process:
                print("  Strategy 3 OK")
        except Exception as e:
            raise RuntimeError(f"All loading strategies failed: {e}") from e

    patch_flash_attention()

    for param in model.parameters():
        param.requires_grad = False
    total_layers = len(model.encoder.layer)
    unfreeze_from = total_layers - unfreeze_last_n
    for i, layer in enumerate(model.encoder.layer):
        if i >= unfreeze_from:
            for param in layer.parameters():
                param.requires_grad = True
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if accelerator.is_main_process:
        print(f"  Encoder layers: {total_layers} | Unfrozen last {unfreeze_last_n}")
        print(f"  Total: {total_params:,} | Trainable: {trainable:,} ({100*trainable/total_params:.1f}%)")
    return model, tokenizer


dnabert_model, tokenizer = load_dnabert2(cfg.DNABERT_MODEL, cfg.UNFREEZE_LAST_N_LAYERS)


def compute_max_token_length(sequences, tok, sample_size=2000, percentile=99, floor=32, cap=96):
    if len(sequences) > sample_size:
        idx = random.sample(range(len(sequences)), sample_size)
        sample = [sequences[i] for i in idx]
    else:
        sample = sequences
    lengths = [len(tok(s, add_special_tokens=True)["input_ids"]) for s in sample]
    p_val = int(np.percentile(lengths, percentile))
    chosen = int(max(floor, min(cap, p_val + 2)))
    if accelerator.is_main_process:
        print(f"\n  AUTO MAX TOKEN LENGTH: p{percentile}={p_val} -> chosen={chosen}")
    return chosen


MAX_LENGTH = (compute_max_token_length(seq_train, tokenizer, floor=cfg.MAX_LENGTH_FLOOR, cap=cfg.MAX_LENGTH_CAP)
              if cfg.AUTO_MAX_LENGTH else cfg.MAX_TOKEN_LENGTH)

# ═══════════════════════════════════════════════════════════════════════
# CELL 7: Architecture — Tri-Branch G-CMAB msCNN
# ═══════════════════════════════════════════════════════════════════════

class LayerAttention(nn.Module):
    """ELMo-style scalar-mix over BERT hidden states (N+1 learnable params)."""
    def __init__(self, n_layers):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, hidden_states_list):
        weights = F.softmax(self.layer_weights, dim=0)
        mixed = torch.zeros_like(hidden_states_list[0])
        for w, h in zip(weights, hidden_states_list):
            mixed = mixed + w * h
        return self.gamma * mixed


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise (per-channel) conv + pointwise (1x1) conv, GroupNorm, GELU."""
    def __init__(self, in_channels, out_channels, kernel_size, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
        self.norm = nn.GroupNorm(num_groups=min(16, out_channels), num_channels=out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.pointwise(self.depthwise(x))))


class MSCNNBranchStack(nn.Module):
    """Parallel depthwise-separable convs at multiple kernel sizes + 1-Max pooling."""
    def __init__(self, in_channels, out_channels, kernels):
        super().__init__()
        self.branches = nn.ModuleList([
            DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size=k) for k in kernels
        ])

    def forward(self, x):
        pooled_outputs = []
        for branch in self.branches:
            conv_out = branch(x)
            pooled = conv_out.max(dim=2)[0]
            pooled_outputs.append(pooled)
        return torch.cat(pooled_outputs, dim=1)


class FeedForward(nn.Module):
    def __init__(self, d_model, expansion=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class CrossModalAttentionLayer(nn.Module):
    """Bidirectional cross-modal attention: seq<->shape."""
    def __init__(self, d_model=128, nhead=4, dropout=0.1):
        super().__init__()
        self.cross_attn_seq2shape = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_seq = nn.LayerNorm(d_model)
        self.ffn_seq = FeedForward(d_model, expansion=4, dropout=dropout)
        self.cross_attn_shape2seq = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_shape = nn.LayerNorm(d_model)
        self.ffn_shape = FeedForward(d_model, expansion=4, dropout=dropout)

    def forward(self, seq_features, shape_features, seq_key_padding_mask=None):
        attended_seq, _ = self.cross_attn_seq2shape(query=seq_features, key=shape_features, value=shape_features)
        seq_out = self.ffn_seq(self.norm_seq(seq_features + attended_seq))
        attended_shape, _ = self.cross_attn_shape2seq(query=shape_features, key=seq_features, value=seq_features,
                                                      key_padding_mask=seq_key_padding_mask)
        shape_out = self.ffn_shape(self.norm_shape(shape_features + attended_shape))
        return seq_out, shape_out


class TriBranchClassifier(nn.Module):
    """Tri-Branch G-CMAB: DNABERT-2 (seq) + DNAshape (shape) + Bio MLP, binary head."""
    def __init__(self, backbone, cfg):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        d_model = cfg.CROSS_ATTN_D_MODEL
        self.use_cross_attn = getattr(cfg, "USE_CROSS_ATTN", True)

        # Branch 3: Bio-Features (CpG O/E, GC, G4)
        self.bio_branch = nn.Sequential(
            nn.Linear(3, 16), nn.BatchNorm1d(16), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(16, 32), nn.GELU(),
        )

        self.use_layer_attn = cfg.USE_LAYER_ATTN
        if self.use_layer_attn:
            self.layer_attention = LayerAttention(n_layers=cfg.LAYER_ATTN_N)
            self._layer_attn_fallback = False
        else:
            self.layer_attention = None

        # Branch 1: sequence projection 768 -> d_model
        self.seq_projection = nn.Sequential(
            nn.Linear(cfg.EMBEDDING_DIM, d_model), nn.LayerNorm(d_model), nn.GELU(),
        )

        if self.use_cross_attn:
            # Branch 2: shape projection 5 -> d_model preserving length 101
            self.shape_projection = nn.Sequential(
                nn.Conv1d(cfg.SHAPE_CHANNELS, d_model, kernel_size=1),
                nn.GroupNorm(min(16, d_model), d_model), nn.GELU(),
            )
            self.cross_attention_layers = nn.ModuleList([
                CrossModalAttentionLayer(d_model, cfg.CROSS_ATTN_NHEAD, cfg.CROSS_ATTN_DROPOUT)
                for _ in range(cfg.CROSS_ATTN_LAYERS)
            ])
            self.seq_mscnn = MSCNNBranchStack(d_model, cfg.MSCNN_OUT_CHANNELS, cfg.SEQ_MSCNN_KERNELS)
            self.shape_mscnn = MSCNNBranchStack(d_model, cfg.MSCNN_OUT_CHANNELS, cfg.SHAPE_MSCNN_KERNELS)
        else:
            self.seq_mscnn = MSCNNBranchStack(d_model, cfg.MSCNN_OUT_CHANNELS, cfg.SEQ_MSCNN_KERNELS)
            self.shape_mscnn = MSCNNBranchStack(cfg.SHAPE_CHANNELS, cfg.MSCNN_OUT_CHANNELS, cfg.SHAPE_MSCNN_KERNELS)

        in_features = (len(cfg.SEQ_MSCNN_KERNELS) + len(cfg.SHAPE_MSCNN_KERNELS)) * cfg.MSCNN_OUT_CHANNELS + 32
        # Binary head -> single logit (BCEWithLogitsLoss)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, cfg.HIDDEN_DIM), nn.GELU(),
            nn.Dropout(cfg.FUSION_DROPOUT), nn.Linear(cfg.HIDDEN_DIM, 1),
        )

    def _get_bert_features(self, input_ids, attention_mask):
        if self.use_layer_attn and not getattr(self, '_layer_attn_fallback', False):
            try:
                outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                                        output_hidden_states=True)
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    all_hidden = outputs.hidden_states
                    n = min(self.cfg.LAYER_ATTN_N, len(all_hidden) - 1)
                    return self.layer_attention(list(all_hidden[-n:]))
                elif isinstance(outputs, tuple) and len(outputs) > 2 and isinstance(outputs[2], (tuple, list)):
                    all_hidden = outputs[2]
                    n = min(self.cfg.LAYER_ATTN_N, len(all_hidden) - 1)
                    return self.layer_attention(list(all_hidden[-n:]))
                else:
                    self._layer_attn_fallback = True
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"  [LayerAttn] output_hidden_states failed ({e}); using hook fallback.")
                self._layer_attn_fallback = True

        if self.use_layer_attn and getattr(self, '_layer_attn_fallback', False):
            hidden_states_collected, hooks = [], []
            n = min(self.cfg.LAYER_ATTN_N, len(self.backbone.encoder.layer))
            start_layer = len(self.backbone.encoder.layer) - n

            def make_hook(storage):
                def hook_fn(module, inp, output):
                    storage.append(output[0] if isinstance(output, tuple) else output)
                return hook_fn

            for i in range(start_layer, len(self.backbone.encoder.layer)):
                hooks.append(self.backbone.encoder.layer[i].register_forward_hook(make_hook(hidden_states_collected)))
            _ = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            for h in hooks:
                h.remove()
            mixed = self.layer_attention(hidden_states_collected)
            if mixed.dim() == 2:
                B, T, D = attention_mask.size(0), attention_mask.size(1), mixed.size(-1)
                padded = torch.zeros(B, T, D, dtype=mixed.dtype, device=mixed.device)
                padded[attention_mask.bool()] = mixed
                mixed = padded
            return mixed

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        elif hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs

    def _fuse_from_embeddings(self, seq_embeddings, shape_features, bio_features, attention_mask):
        """Shared trunk from precomputed seq embeddings -> logits. Reused by SHAP."""
        seq_features = self.seq_projection(seq_embeddings)
        if self.use_cross_attn:
            shape_proj = self.shape_projection(shape_features)
            shape_feats = shape_proj.transpose(1, 2)
            seq_key_padding_mask = (attention_mask == 0)
            for cross_layer in self.cross_attention_layers:
                seq_features, shape_feats = cross_layer(seq_features, shape_feats,
                                                        seq_key_padding_mask=seq_key_padding_mask)
            seq_in = seq_features.transpose(1, 2)
            shape_in = shape_feats.transpose(1, 2)
        else:
            seq_in = seq_features.transpose(1, 2)
            shape_in = shape_features
        seq_pooled = self.seq_mscnn(seq_in)
        shape_pooled = self.shape_mscnn(shape_in)
        bio_out = self.bio_branch(bio_features)
        fused = torch.cat([seq_pooled, shape_pooled, bio_out], dim=1)
        return self.classifier(fused)

    def forward(self, input_ids, attention_mask, shape_features, bio_features):
        hidden_states = self._get_bert_features(input_ids, attention_mask)
        logits = self._fuse_from_embeddings(hidden_states, shape_features, bio_features, attention_mask)
        return logits.squeeze(-1)

# ═══════════════════════════════════════════════════════════════════════
# CELL 8: Dataset & DataLoaders
# ═══════════════════════════════════════════════════════════════════════

import re as regex

class TriBranchDataset(Dataset):
    """Tokenized sequence + DNAshape + sequence-derived bio-features (CpG O/E, GC, G4)."""
    def __init__(self, sequences, labels, shape_features, tokenizer, max_length=48):
        self.sequences = sequences
        self.labels = labels
        self.shape_features = shape_features
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.g4_pattern = regex.compile(r'(G{3,}[ACGTN]{1,7}){3,}G{3,}', regex.IGNORECASE)
        self.bio_features = self._precompute_bio_features()

    def _precompute_bio_features(self):
        features = []
        for seq in self.sequences:
            L = len(seq)
            if L == 0:
                features.append([0.0, 0.0, 0.0]); continue
            c = seq.count('C'); g = seq.count('G'); cg = seq.count('CG')
            cpg_oe = (cg * L) / (c * g) if (c * g) > 0 else 0.0       # O/E = (N_CG * L) / (N_C * N_G)
            gc_content = (c + g) / L
            g4 = 1.0 if self.g4_pattern.search(seq) else 0.0
            features.append([cpg_oe, gc_content, g4])
        return np.array(features, dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.sequences[idx], padding="max_length", truncation=True,
                             max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "shape_features": torch.tensor(self.shape_features[idx], dtype=torch.float32),
            "bio_features": torch.tensor(self.bio_features[idx], dtype=torch.float32),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# Binary mapping: SP1(0)/SP2(1)/SP4(2) -> Positive(1); Negative(3) -> Negative(0)
y_train = np.array([1 if l < 3 else 0 for l in y_train])
y_test = np.array([1 if l < 3 else 0 for l in y_test])

train_dataset = TriBranchDataset(seq_train, y_train, shape_train_norm, tokenizer, MAX_LENGTH)
test_dataset = TriBranchDataset(seq_test, y_test, shape_test_norm, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE * 2, shuffle=False,
                         num_workers=2, pin_memory=True)

if accelerator.is_main_process:
    print(f"\nDataLoaders: {len(train_loader)} train / {len(test_loader)} test batches | max_length={MAX_LENGTH}")
    print(f"  Train binary dist (neg,pos): {np.bincount(y_train)} | Test: {np.bincount(y_test)}")

# Quick bio-feature contrast print (real negatives should NOT preserve GC/CpG
# exactly the way dinuc-shuffle does — this is the heart of the artifact check).
if accelerator.is_main_process:
    def _bio_summary(ds, ys):
        bf = ds.bio_features
        pos = bf[ys == 1]; neg = bf[ys == 0]
        return pos.mean(axis=0), neg.mean(axis=0)
    pmean, nmean = _bio_summary(test_dataset, y_test)
    print(f"  [Bio contrast | test] CpG O/E  pos={pmean[0]:.3f} neg={nmean[0]:.3f}")
    print(f"  [Bio contrast | test] GC       pos={pmean[1]:.3f} neg={nmean[1]:.3f}")
    print(f"  [Bio contrast | test] G4 frac  pos={pmean[2]:.3f} neg={nmean[2]:.3f}")

# ═══════════════════════════════════════════════════════════════════════
# CELL 9: Build Model, Optimizer, Loss
# ═══════════════════════════════════════════════════════════════════════

model = TriBranchClassifier(backbone=dnabert_model, cfg=cfg)

param_groups = []
backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
if backbone_params:
    param_groups.append({"params": backbone_params, "lr": cfg.BACKBONE_LR, "weight_decay": cfg.WEIGHT_DECAY})
param_groups.append({"params": list(model.seq_projection.parameters()), "lr": cfg.SEQ_PROJ_LR, "weight_decay": cfg.WEIGHT_DECAY})
if hasattr(model, "shape_projection"):
    param_groups.append({"params": list(model.shape_projection.parameters()), "lr": cfg.SHAPE_LR, "weight_decay": cfg.WEIGHT_DECAY})
if hasattr(model, "cross_attention_layers"):
    param_groups.append({"params": list(model.cross_attention_layers.parameters()), "lr": cfg.CROSS_ATTN_LR, "weight_decay": cfg.WEIGHT_DECAY})
param_groups.append({"params": list(model.seq_mscnn.parameters()), "lr": cfg.SHAPE_LR, "weight_decay": cfg.WEIGHT_DECAY})
param_groups.append({"params": list(model.shape_mscnn.parameters()), "lr": cfg.SHAPE_LR, "weight_decay": cfg.WEIGHT_DECAY})
param_groups.append({"params": list(model.bio_branch.parameters()), "lr": cfg.SHAPE_LR, "weight_decay": cfg.WEIGHT_DECAY})
param_groups.append({"params": list(model.classifier.parameters()), "lr": cfg.HEAD_LR, "weight_decay": cfg.WEIGHT_DECAY})
if cfg.USE_LAYER_ATTN and model.layer_attention is not None:
    param_groups.append({"params": list(model.layer_attention.parameters()), "lr": cfg.LAYER_ATTN_LR, "weight_decay": 0.0})

optimizer = optim.AdamW(param_groups)

total_steps = (len(train_loader) // max(cfg.GRAD_ACCUM_STEPS, 1)) * cfg.EPOCHS
warmup_steps = int(total_steps * cfg.WARMUP_RATIO)

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

pos_count = int(np.sum(y_train == 1))
neg_count = int(np.sum(y_train == 0))
pos_weight_val = float(neg_count) / float(pos_count) if pos_count > 0 else 1.0
pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=accelerator.device)
if accelerator.is_main_process:
    print(f"  BCE pos_weight: {pos_weight_val:.4f} (neg={neg_count}, pos={pos_count})")
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, test_loader, scheduler
)

# ═══════════════════════════════════════════════════════════════════════
# CELL 10: Training Loop (early stopping + overfitting guard)
# ═══════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, scheduler, criterion, acc, grad_accum, max_grad_norm, epoch_num):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    grad_norms = []
    pbar = tqdm(loader, desc="  Training", leave=False, disable=not acc.is_main_process)
    for batch in pbar:
        with acc.accumulate(model):
            with acc.autocast():
                logits = model(batch["input_ids"], batch["attention_mask"],
                               batch["shape_features"], batch["bio_features"])
                loss = criterion(logits, batch["labels"].float())
            acc.backward(loss)
            if acc.sync_gradients:
                tn = acc.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                grad_norms.append(tn.item() if isinstance(tn, torch.Tensor) else float(tn))
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
        running_loss += loss.item() * batch["labels"].size(0)
        predicted = (logits > 0).long()
        total += batch["labels"].size(0)
        correct += predicted.eq(batch["labels"]).sum().item()
        if acc.is_main_process:
            pbar.set_postfix({"loss": f"{running_loss/total:.4f}", "acc": f"{correct/total:.4f}"})
    if epoch_num < 5 and grad_norms and acc.is_main_process:
        print(f"  [Grad] avg={np.mean(grad_norms):.4f}, max={np.max(grad_norms):.4f}")
    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, acc):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        with acc.autocast():
            logits = model(batch["input_ids"], batch["attention_mask"],
                           batch["shape_features"], batch["bio_features"])
            loss = criterion(logits, batch["labels"].float())
        preds = (logits > 0).long()
        preds, labels_g = acc.gather_for_metrics((preds, batch["labels"]))
        loss_g = acc.gather_for_metrics(loss.repeat(batch["labels"].size(0)))
        running_loss += loss_g.sum().item()
        total += labels_g.size(0)
        correct += preds.eq(labels_g).sum().item()
    return running_loss / max(total, 1), correct / max(total, 1)


def train_model(model, train_loader, test_loader, optimizer, scheduler, criterion, acc, cfg):
    if acc.is_main_process:
        print("\n" + "=" * 60)
        print(f"TRAINING -- Binary Tri-Branch G-CMAB msCNN (Script 32, NEG={cfg.NEG_KIND})")
        print("=" * 60)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, patience_counter = 0.0, 0
    for epoch in range(cfg.EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, acc,
                                          cfg.GRAD_ACCUM_STEPS, cfg.MAX_GRAD_NORM, epoch)
        val_loss, val_acc = evaluate(model, test_loader, criterion, acc)
        elapsed = time.time() - t0

        tr_loss = acc.gather(torch.tensor(tr_loss, device=acc.device)).mean().item()
        tr_acc = acc.gather(torch.tensor(tr_acc, device=acc.device)).mean().item()
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)
        acc.wait_for_everyone()

        gap = (tr_acc - val_acc) * 100
        print(f"Epoch {epoch+1:02d}/{cfg.EPOCHS} | Train {tr_loss:.4f}/{tr_acc:.4f} | "
              f"Val {val_loss:.4f}/{val_acc:.4f} | Gap {gap:+.2f}% | "
              f"LR {optimizer.param_groups[0]['lr']:.1e} | {elapsed:.0f}s")

        if gap >= cfg.MAX_OVERFITTING_GAP:
            print(f"\n  Early stop (overfitting gap {gap:+.2f}% >= {cfg.MAX_OVERFITTING_GAP}%)")
            break

        if val_acc > best_val_acc:
            best_val_acc, patience_counter = val_acc, 0
            if acc.is_main_process:
                unwrapped = acc.unwrap_model(model)
                torch.save({"epoch": epoch + 1, "model_state_dict": unwrapped.state_dict(),
                            "val_loss": val_loss, "val_acc": val_acc},
                           os.path.join(cfg.MODEL_DIR, "best_real_negatives.pt"))
                print(f"  -> Saved best (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                print(f"\n  Early stop (patience={cfg.PATIENCE})")
                break
        acc.wait_for_everyone()
    if acc.is_main_process:
        print(f"\nTraining complete. Best val_acc: {best_val_acc:.4f}")
    return history


history = train_model(model, train_loader, test_loader, optimizer, scheduler, criterion, accelerator, cfg)

# ═══════════════════════════════════════════════════════════════════════
# CELL 11: Load Best Model + Full Evaluation
# ═══════════════════════════════════════════════════════════════════════

best_path = os.path.join(cfg.MODEL_DIR, "best_real_negatives.pt")
checkpoint = torch.load(best_path, map_location=DEVICE)
accelerator.unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
print(f"Loaded best model from epoch {checkpoint['epoch']} (val_acc={checkpoint['val_acc']:.4f})")


def roc_auc_safe(targets, probs):
    try:
        fpr, tpr, _ = roc_curve(targets, probs)
        return auc(fpr, tpr)
    except Exception:
        return float("nan")


@torch.no_grad()
def full_evaluation(model, test_loader, class_names, acc):
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    for batch in tqdm(test_loader, desc="  Evaluating", disable=not acc.is_main_process):
        with acc.autocast():
            logits = model(batch["input_ids"], batch["attention_mask"],
                           batch["shape_features"], batch["bio_features"])
        probs = torch.sigmoid(logits.float())
        preds = (logits > 0).long()
        preds_g, labels_g, probs_g = acc.gather_for_metrics((preds, batch["labels"], probs))
        all_preds.extend(preds_g.cpu().numpy())
        all_targets.extend(labels_g.cpu().numpy())
        all_probs.extend(probs_g.cpu().numpy())
    all_preds, all_targets, all_probs = np.array(all_preds), np.array(all_targets), np.array(all_probs)

    if acc.is_main_process:
        print("\n" + "=" * 60); print("CLASSIFICATION REPORT"); print("=" * 60)
        report_str = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
        print(report_str)
        acc_val = accuracy_score(all_targets, all_preds)
        roc_auc = roc_auc_safe(all_targets, all_probs)
        pr_auc = average_precision_score(all_targets, all_probs)
        f1b = f1_score(all_targets, all_preds, average="binary")
        # Negative recall is the key artifact-sensitive metric.
        neg_recall = recall_score(all_targets, all_preds, pos_label=0, zero_division=0)
        pos_recall = recall_score(all_targets, all_preds, pos_label=1, zero_division=0)
        with open(os.path.join(cfg.OUTPUT_DIR, "classification_report.txt"), "w") as f:
            f.write("CLASSIFICATION REPORT -- Binary Tri-Branch G-CMAB msCNN (Script 32, REAL negatives)\n")
            f.write(f"Negative source: NEG_KIND={cfg.NEG_KIND}  ({NEG_DESC})\n")
            f.write(f"  FASTA:    {NEG_FASTA_PATH}\n")
            f.write(f"  DNAshape: {NEG_SHAPE_PATH}\n")
            f.write("Sequence (DNABERT-2) + DNAshape + Bio-features\n")
            f.write("=" * 60 + "\n")
            f.write(report_str + "\n")
            f.write(f"\nOverall Accuracy: {acc_val:.4f}\nBinary F1: {f1b:.4f}\n")
            f.write(f"ROC-AUC: {roc_auc:.4f}\nPR-AUC (Average Precision): {pr_auc:.4f}\n")
            f.write(f"Negative-recall: {neg_recall:.4f}\nPositive-recall: {pos_recall:.4f}\n")
        print(f"Overall Accuracy: {acc_val:.4f} | Binary F1: {f1b:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
        print(f"Negative-recall: {neg_recall:.4f} | Positive-recall: {pos_recall:.4f}")

        # ── ARTIFACT CHECK verdict (vs the dinuc-shuffle baseline) ──
        print("\n" + "=" * 60)
        print("ARTIFACT CHECK (REAL negatives vs dinuc-shuffle baseline)")
        print("=" * 60)
        drop = cfg.DINUC_BASELINE_ACC - acc_val
        verdict_lines = []
        verdict_lines.append(f"NEG_KIND               : {cfg.NEG_KIND}  ({NEG_DESC})")
        verdict_lines.append(f"Dinuc-shuffle baseline : {cfg.DINUC_BASELINE_ACC:.4f}")
        verdict_lines.append(f"Real-negative accuracy : {acc_val:.4f}")
        verdict_lines.append(f"Accuracy drop          : {drop:+.4f} ({drop*100:+.2f} pp)")
        verdict_lines.append(f"Negative-recall        : {neg_recall:.4f}")
        if not IS_REAL_NEG:
            verdict_lines.append("VERDICT: INCONCLUSIVE — only dinuc-shuffled negatives were available; "
                                 "this run cannot test for a shuffle artifact.")
        elif drop <= 0.02 and neg_recall >= 0.85:
            verdict_lines.append("VERDICT: HOLDS — performance is preserved on real genomic negatives; "
                                 "the gain is NOT a shuffle artifact.")
        elif drop <= 0.05 and neg_recall >= 0.70:
            verdict_lines.append("VERDICT: MOSTLY HOLDS — modest drop on real negatives; "
                                 "some shuffle dependence but the signal is largely real.")
        else:
            verdict_lines.append("VERDICT: LIKELY ARTIFACT — large drop / collapsed Neg-recall on real "
                                 "negatives; the dinuc-shuffle figure is partly an artifact.")
        for ln in verdict_lines:
            print("  " + ln)
        with open(os.path.join(cfg.OUTPUT_DIR, "artifact_check.txt"), "w") as f:
            f.write("ARTIFACT CHECK -- Real genomic negatives vs dinuc-shuffle baseline (Script 32)\n")
            f.write("=" * 70 + "\n")
            for ln in verdict_lines:
                f.write(ln + "\n")
        print("  -> artifact_check.txt")
    return all_preds, all_targets, all_probs


all_preds, all_targets, all_probs = full_evaluation(model, test_loader, cfg.CLASS_NAMES, accelerator)

# ═══════════════════════════════════════════════════════════════════════
# CELL 11.5: BED export for IGV — BINARY (TP / FN / FP)
# ═══════════════════════════════════════════════════════════════════════

def parse_header_to_bed(header):
    """Positive headers look like 'chr5:10375416-10375517[_revcomp]'. Real
    genomic negatives also carry coords ('chr11:74454433-74454534'); dinuc
    negatives ('neg_sp1_...|chrN:...') are handled by stripping the prefix."""
    try:
        clean = header.split("_revcomp")[0]
        if "|" in clean:                  # e.g. 'neg_sp1_dinuc_shuffle|chr9:...'
            clean = clean.split("|")[-1]
        chrom, coords = clean.split(":")
        start, end = coords.split("-")
        if not chrom.startswith("chr"):
            return None
        int(start); int(end)
        return chrom, start, end
    except Exception:
        return None


def export_igv_bed_files(headers_test, preds, targets, output_dir):
    beds = {"True_Positive": [], "False_Negative": [], "False_Positive": []}
    for idx, (pred, target) in enumerate(zip(preds, targets)):
        fields = parse_header_to_bed(headers_test[idx])
        if not fields:
            continue
        line = f"{fields[0]}\t{fields[1]}\t{fields[2]}\n"
        if target == 1 and pred == 1:
            beds["True_Positive"].append(line)
        elif target == 1 and pred == 0:
            beds["False_Negative"].append(line)
        elif target == 0 and pred == 1:
            beds["False_Positive"].append(line)
    for name, lines in beds.items():
        with open(os.path.join(output_dir, f"{name}.bed"), "w") as f:
            f.writelines(lines)
        print(f"    {name}.bed: {len(lines)} lines")


if accelerator.is_main_process:
    print("\n  BED files for IGV (binary TP/FN/FP):")
    export_igv_bed_files(headers_test, all_preds, all_targets, cfg.OUTPUT_DIR)

# ═══════════════════════════════════════════════════════════════════════
# CELL 12: Performance Figures
# ═══════════════════════════════════════════════════════════════════════

TITLE_PREFIX = f"Binary Tri-Branch G-CMAB (Seq + Shape + Bio) -- REAL negatives [{cfg.NEG_KIND}]"

def plot_training_curves(history, save_dir):
    epochs = len(history["train_loss"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(range(1, epochs+1), history["train_loss"], label="Train Loss", color="#2196F3", linewidth=2)
    ax1.plot(range(1, epochs+1), history["val_loss"], label="Val Loss", color="#FF5722", linewidth=2)
    ax1.set_title("Loss Convergence", fontweight="bold"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, linestyle="--", alpha=0.4)
    ax2.plot(range(1, epochs+1), history["train_acc"], label="Train Acc", color="#4CAF50", linewidth=2)
    ax2.plot(range(1, epochs+1), history["val_acc"], label="Val Acc", color="#E91E63", linewidth=2)
    ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance (0.50)")
    ax2.set_title("Accuracy", fontweight="bold"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.legend(); ax2.grid(True, linestyle="--", alpha=0.4)
    plt.suptitle(f"{TITLE_PREFIX} -- Training Progress", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "tribranch_training_curves.png"), dpi=150, bbox_inches="tight"); plt.close()


def plot_confusion_matrix(targets, preds, class_names, save_dir):
    cm = confusion_matrix(targets, preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, data, title, fmt in [(ax1, cm, "Counts", "d"), (ax2, cm_norm, "Normalized", ".2%")]:
        im = ax.imshow(data, cmap="Blues")
        ax.set_title(f"Confusion Matrix ({title})", fontweight="bold")
        ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        thresh = data.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if data[i, j] > thresh else "black", fontweight="bold")
    plt.suptitle(TITLE_PREFIX, fontweight="bold", y=1.02); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "tribranch_confusion_matrix.png"), dpi=150, bbox_inches="tight"); plt.close()


def plot_roc_curve(targets, probs, save_dir):
    plt.figure(figsize=(9, 7))
    fpr, tpr, _ = roc_curve(targets, probs)
    plt.plot(fpr, tpr, color="#2196F3", linewidth=2.5, label=f"SP_Positive vs Negative (AUC = {auc(fpr, tpr):.4f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (0.50)")
    plt.xlim([0, 1]); plt.ylim([0, 1.05]); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC -- {TITLE_PREFIX}", fontweight="bold"); plt.legend(loc="lower right"); plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, "tribranch_roc_curve.png"), dpi=150, bbox_inches="tight"); plt.close()


def plot_pr_curve(targets, probs, save_dir):
    plt.figure(figsize=(9, 7))
    precision, recall, _ = precision_recall_curve(targets, probs)
    ap = average_precision_score(targets, probs)
    plt.plot(recall, precision, color="#2196F3", linewidth=2.5, label=f"SP_Positive (AP = {ap:.4f})")
    pos_ratio = np.mean(targets == 1)
    plt.axhline(y=pos_ratio, color="gray", linestyle="--", alpha=0.5, label=f"Baseline ({pos_ratio:.2f})")
    plt.xlim([0, 1]); plt.ylim([0, 1.05]); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision-Recall -- {TITLE_PREFIX}", fontweight="bold"); plt.legend(loc="lower left"); plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, "tribranch_pr_curve.png"), dpi=150, bbox_inches="tight"); plt.close()


def plot_per_class_metrics(targets, preds, class_names, save_dir):
    prec, rec, f1, support = precision_recall_fscore_support(targets, preds, average=None)
    x = np.arange(len(class_names)); width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    for offset, vals, lbl, col in [(-width, prec, "Precision", "#2196F3"), (0, rec, "Recall", "#4CAF50"), (width, f1, "F1", "#FF9800")]:
        bars = ax.bar(x + offset, vals, width, label=lbl, color=col, alpha=0.85)
        for bar in bars:
            ax.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([f"{cn}\n(n={s})" for cn, s in zip(class_names, support)])
    ax.set_ylabel("Score"); ax.set_title(f"Per-Class Performance -- {TITLE_PREFIX}", fontweight="bold")
    ax.set_ylim(0, 1.15); ax.legend(); ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, "tribranch_per_class_metrics.png"), dpi=150, bbox_inches="tight"); plt.close()


if accelerator.is_main_process:
    print("\n" + "=" * 60); print("GENERATING PERFORMANCE FIGURES"); print("=" * 60)
    plot_training_curves(history, cfg.FIG_DIR)
    plot_confusion_matrix(all_targets, all_preds, cfg.CLASS_NAMES, cfg.FIG_DIR)
    plot_roc_curve(all_targets, all_probs, cfg.FIG_DIR)
    plot_pr_curve(all_targets, all_probs, cfg.FIG_DIR)
    plot_per_class_metrics(all_targets, all_preds, cfg.CLASS_NAMES, cfg.FIG_DIR)
    print("  Figures saved to:", cfg.FIG_DIR)

# ═══════════════════════════════════════════════════════════════════════
# CELL 13: Tri-Modal SHAP Analysis (Sequence + DNAshape + Bio)
#   GradientExplainer over the 3 differentiable branches (backbone bypassed via
#   precomputed embeddings). Subset A = correctly-called SP_Positive (TP),
#   Subset B = correctly-called Negative (TN) — always populated, and directly
#   answers "what features drive a binding vs non-binding decision".
# ═══════════════════════════════════════════════════════════════════════

if accelerator.is_main_process:
    print("\n" + "=" * 60)
    print("TRI-MODAL SHAP INTERPRETABILITY (Seq + Shape + Bio)")
    print("=" * 60)
    try:
        import shap
        import seaborn as sns
        import re

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.eval()
        device = accelerator.device

        # 1) Predictions to bucket samples
        sh_preds, sh_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                logits = unwrapped_model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                                         batch["shape_features"].to(device), batch["bio_features"].to(device))
                sh_preds.extend((logits > 0).long().cpu().numpy())
                sh_targets.extend(batch["labels"].cpu().numpy())
        sh_preds, sh_targets = np.array(sh_preds), np.array(sh_targets)

        idx_TP = [i for i in range(len(sh_targets)) if sh_targets[i] == 1 and sh_preds[i] == 1]
        idx_TN = [i for i in range(len(sh_targets)) if sh_targets[i] == 0 and sh_preds[i] == 0]
        idx_FN = [i for i in range(len(sh_targets)) if sh_targets[i] == 1 and sh_preds[i] == 0]
        idx_FP = [i for i in range(len(sh_targets)) if sh_targets[i] == 0 and sh_preds[i] == 1]
        print(f"  Buckets: TP={len(idx_TP)}, TN={len(idx_TN)}, FN={len(idx_FN)}, FP={len(idx_FP)}")

        subset_A_indices = idx_TP                         # correctly called SP_Positive
        subset_B_indices = idx_TN if len(idx_TN) >= 3 else idx_FN
        label_A, label_B = "Correct SP_Positive", ("Correct Negative" if len(idx_TN) >= 3 else "False Negative")

        if len(subset_A_indices) >= 3 and len(subset_B_indices) >= 3:
            n_explain = cfg.SHAP_NUM_EXPLAIN
            explain_A = subset_A_indices[:n_explain]
            explain_B = subset_B_indices[:n_explain]

            def stack_field(indices, field):
                return torch.stack([test_dataset[i][field] for i in indices]).to(device)

            # Background (precompute embeddings through the frozen-at-inference backbone)
            np.random.seed(cfg.RANDOM_SEED)
            bg_indices = np.random.choice(len(test_dataset), min(cfg.SHAP_BG_SIZE, len(test_dataset)), replace=False)
            bg_input_ids = stack_field(bg_indices, "input_ids")
            bg_attn = stack_field(bg_indices, "attention_mask")
            bg_shapes = stack_field(bg_indices, "shape_features").float()
            bg_bio = stack_field(bg_indices, "bio_features").float()
            with torch.no_grad():
                bg_emb = unwrapped_model._get_bert_features(bg_input_ids, bg_attn).float()

            # Differentiable wrapper over (seq_embeddings, shape, bio) -> logit
            class TriBranchEmbeddingWrapper(nn.Module):
                def __init__(self, m):
                    super().__init__(); self.m = m
                def forward(self, seq_emb, shape_features, bio_features):
                    B, T, _ = seq_emb.shape
                    attn = torch.ones(B, T, dtype=torch.long, device=seq_emb.device)
                    return self.m._fuse_from_embeddings(seq_emb, shape_features, bio_features, attn)

            wrapper = TriBranchEmbeddingWrapper(unwrapped_model).to(device).eval()

            def embed(indices):
                ii, am = stack_field(indices, "input_ids"), stack_field(indices, "attention_mask")
                with torch.no_grad():
                    return unwrapped_model._get_bert_features(ii, am).float()

            emb_A, shp_A, bio_A = embed(explain_A), stack_field(explain_A, "shape_features").float(), stack_field(explain_A, "bio_features").float()
            emb_B, shp_B, bio_B = embed(explain_B), stack_field(explain_B, "shape_features").float(), stack_field(explain_B, "bio_features").float()

            print("  Running GradientExplainer over [sequence, shape, bio]...")
            explainer = shap.GradientExplainer(wrapper, [bg_emb, bg_shapes, bg_bio])
            sv_A = explainer.shap_values([emb_A, shp_A, bio_A])
            sv_B = explainer.shap_values([emb_B, shp_B, bio_B])

            def unpack(sv):
                # single-logit output -> list of 3 arrays; squeeze any trailing singleton class axis
                seq_s, shape_s, bio_s = sv[0], sv[1], sv[2]
                def sq(a, nd):
                    a = np.array(a)
                    while a.ndim > nd and a.shape[-1] == 1:
                        a = a[..., 0]
                    return a
                return sq(seq_s, 3), sq(shape_s, 3), sq(bio_s, 2)

            seq_shap_A, shape_shap_A, bio_shap_A = unpack(sv_A)
            seq_shap_B, shape_shap_B, bio_shap_B = unpack(sv_B)

            # ── PLOT 1: DNAshape feature importance (5 params) ──
            imp_shape_A = np.abs(shape_shap_A).sum(axis=2).mean(axis=0)
            imp_shape_B = np.abs(shape_shap_B).sum(axis=2).mean(axis=0)
            feats = ["MGW", "ProT", "Roll", "HelT", "EP"]
            x = np.arange(len(feats)); w = 0.35
            plt.figure(figsize=(8, 5))
            plt.bar(x - w/2, imp_shape_A, w, label=f"Subset A ({label_A})", color="#4CAF50")
            plt.bar(x + w/2, imp_shape_B, w, label=f"Subset B ({label_B})", color="#F44336")
            plt.xticks(x, feats); plt.ylabel("Mean |SHAP| (summed over positions)")
            plt.title("DNAshape Feature Importance"); plt.legend(); plt.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout(); plt.savefig(os.path.join(cfg.FIG_DIR, "tribranch_shap_dnashape_bar.png"), dpi=150); plt.close()
            print("  -> tribranch_shap_dnashape_bar.png")

            # ── PLOT 2: Bio-feature importance (CpG O/E, GC, G4) ── [tri-branch specific]
            imp_bio_A = np.abs(bio_shap_A).mean(axis=0)
            imp_bio_B = np.abs(bio_shap_B).mean(axis=0)
            biofeats = ["CpG O/E", "GC content", "G4 motif"]
            xb = np.arange(len(biofeats))
            plt.figure(figsize=(7, 5))
            plt.bar(xb - w/2, imp_bio_A, w, label=f"Subset A ({label_A})", color="#4CAF50")
            plt.bar(xb + w/2, imp_bio_B, w, label=f"Subset B ({label_B})", color="#F44336")
            plt.xticks(xb, biofeats); plt.ylabel("Mean |SHAP|")
            plt.title("Bio-Feature Importance"); plt.legend(); plt.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout(); plt.savefig(os.path.join(cfg.FIG_DIR, "tribranch_shap_bio_bar.png"), dpi=150); plt.close()
            print("  -> tribranch_shap_bio_bar.png")

            # ── PLOT 3: Modality-level total contribution (Seq vs Shape vs Bio) ──
            def modality_total(seq_s, shape_s, bio_s):
                s = np.abs(seq_s).reshape(seq_s.shape[0], -1).sum(axis=1).mean()
                sh = np.abs(shape_s).reshape(shape_s.shape[0], -1).sum(axis=1).mean()
                b = np.abs(bio_s).reshape(bio_s.shape[0], -1).sum(axis=1).mean()
                return np.array([s, sh, b])
            mod_A = modality_total(seq_shap_A, shape_shap_A, bio_shap_A)
            mod_B = modality_total(seq_shap_B, shape_shap_B, bio_shap_B)
            mods = ["Sequence", "DNAshape", "Bio"]
            xm = np.arange(len(mods))
            plt.figure(figsize=(7, 5))
            plt.bar(xm - w/2, mod_A, w, label=f"Subset A ({label_A})", color="#4CAF50")
            plt.bar(xm + w/2, mod_B, w, label=f"Subset B ({label_B})", color="#F44336")
            plt.xticks(xm, mods); plt.ylabel("Total |SHAP| per sample (Σ over features)")
            plt.title("Modality-Level Contribution (Tri-Branch)"); plt.legend(); plt.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout(); plt.savefig(os.path.join(cfg.FIG_DIR, "tribranch_shap_modality_bar.png"), dpi=150); plt.close()
            print("  -> tribranch_shap_modality_bar.png")

            # ── PLOT 4+5: Sequence SHAP aligned to GC-box center ──
            def get_char_shap(explain_indices, seq_shap):
                char_shaps = []
                for j, original_idx in enumerate(explain_indices):
                    seq = seq_test[original_idx]
                    tokens = tokenizer.convert_ids_to_tokens(test_dataset[original_idx]["input_ids"])
                    token_imp = np.linalg.norm(seq_shap[j], axis=1)
                    char_shap = np.zeros(len(seq)); char_counts = np.zeros(len(seq))
                    cur = 0
                    for t_idx, token in enumerate(tokens):
                        if token in ["[CLS]", "[SEP]", "[PAD]", "<pad>", "<s>", "</s>", "<unk>"]:
                            continue
                        ct = token.replace("##", "").replace("Ġ", "").replace(" ", "")
                        if not ct:
                            continue
                        pos = seq.find(ct, cur)
                        if pos != -1:
                            char_shap[pos:pos+len(ct)] += token_imp[t_idx]
                            char_counts[pos:pos+len(ct)] += 1
                            cur = pos + len(ct)
                    char_shaps.append(np.where(char_counts > 0, char_shap / char_counts, 0.0))
                return char_shaps

            def align_avg(seq_list, char_shaps, window):
                aligned = []
                for seq, cs in zip(seq_list, char_shaps):
                    m = re.search(r'GGGCGG|CCGCCC', seq, re.IGNORECASE)
                    if not m:
                        continue
                    center = (m.span()[0] + m.span()[1]) // 2
                    row = np.zeros(2 * window + 1)
                    for i, off in enumerate(range(-window, window + 1)):
                        p = center + off
                        if 0 <= p < len(cs):
                            row[i] = cs[p]
                    aligned.append(row)
                return np.mean(aligned, axis=0) if aligned else None

            cs_A = get_char_shap(explain_A, seq_shap_A)
            cs_B = get_char_shap(explain_B, seq_shap_B)
            W = cfg.SHAP_WINDOW
            avg_A = align_avg([seq_test[i] for i in explain_A], cs_A, W)
            avg_B = align_avg([seq_test[i] for i in explain_B], cs_B, W)

            if avg_A is not None:
                rows, ylabels = [avg_A], [f"A ({label_A})"]
                if avg_B is not None:
                    rows.append(avg_B); ylabels.append(f"B ({label_B})")
                plt.figure(figsize=(12, 2.5 + 0.6 * len(rows)))
                sns.heatmap(np.vstack(rows), cmap="viridis", yticklabels=ylabels,
                            xticklabels=[str(o) for o in range(-W, W + 1)])
                plt.xlabel("Position relative to GC-box center (bp)")
                plt.title("Sequence SHAP importance around GC-box")
                plt.tight_layout(); plt.savefig(os.path.join(cfg.FIG_DIR, "tribranch_shap_sequence_heatmap.png"), dpi=150); plt.close()
                print("  -> tribranch_shap_sequence_heatmap.png")

                plt.figure(figsize=(10, 4))
                offsets = np.arange(-W, W + 1)
                plt.plot(offsets, avg_A, label=f"A ({label_A})", color="#4CAF50", linewidth=2)
                if avg_B is not None:
                    plt.plot(offsets, avg_B, label=f"B ({label_B})", color="#F44336", linewidth=2)
                plt.axvline(x=0, color="gray", linestyle="--", alpha=0.7, label="GC-box center")
                plt.xlabel("Position relative to GC-box center (bp)"); plt.ylabel("Avg SHAP importance (L2 norm)")
                plt.title("Sequence SHAP importance (GC-box flanks)"); plt.legend(); plt.grid(True, linestyle="--", alpha=0.3)
                plt.tight_layout(); plt.savefig(os.path.join(cfg.FIG_DIR, "tribranch_shap_sequence_line.png"), dpi=150); plt.close()
                print("  -> tribranch_shap_sequence_line.png")
            else:
                print("  WARNING: no GC-box consensus found in Subset A; skipped sequence alignment plots.")

            # Save a compact textual summary of SHAP findings
            with open(os.path.join(cfg.OUTPUT_DIR, "shap_summary.txt"), "w") as f:
                f.write("TRI-MODAL SHAP SUMMARY (Script 32, REAL negatives)\n" + "=" * 50 + "\n")
                f.write(f"Negative source: NEG_KIND={cfg.NEG_KIND}  ({NEG_DESC})\n")
                f.write(f"Subset A = {label_A} (n={len(explain_A)}); Subset B = {label_B} (n={len(explain_B)})\n\n")
                f.write("Modality total |SHAP| per sample [Sequence, DNAshape, Bio]:\n")
                f.write(f"  A: {np.round(mod_A, 4).tolist()}\n  B: {np.round(mod_B, 4).tolist()}\n\n")
                f.write("DNAshape mean |SHAP| [MGW, ProT, Roll, HelT, EP]:\n")
                f.write(f"  A: {np.round(imp_shape_A, 4).tolist()}\n  B: {np.round(imp_shape_B, 4).tolist()}\n\n")
                f.write("Bio mean |SHAP| [CpG O/E, GC, G4]:\n")
                f.write(f"  A: {np.round(imp_bio_A, 4).tolist()}\n  B: {np.round(imp_bio_B, 4).tolist()}\n")
            print("  -> shap_summary.txt")
        else:
            print("  WARNING: not enough samples in Subset A/B for SHAP.")
    except Exception as shap_err:
        print(f"  Error during SHAP analysis: {shap_err}")
        import traceback
        traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════════
# CELL 14: Summary + Zip
# ═══════════════════════════════════════════════════════════════════════

if accelerator.is_main_process:
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY -- Real-Negatives Eval, Tri-Branch G-CMAB msCNN + SHAP (Script 32)")
    print("=" * 60)
    print(f"  Negative source: NEG_KIND={cfg.NEG_KIND} ({NEG_DESC})")
    print(f"  Branches:        Sequence (DNABERT-2, last {cfg.UNFREEZE_LAST_N_LAYERS}) + DNAshape + Bio")
    print(f"  Cross-Attention: {cfg.USE_CROSS_ATTN} | GroupNorm: {cfg.USE_GROUPNORM} | LayerAttn: {cfg.USE_LAYER_ATTN}")
    print(f"  Best val acc:    {max(history['val_acc']):.4f}")
    print(f"  Outputs:         {cfg.OUTPUT_DIR}")

    import shutil
    try:
        from IPython.display import FileLink, display as ipy_display
    except ImportError:
        FileLink = ipy_display = None
    zip_filename = "outputs_real_negatives"
    shutil.make_archive(zip_filename, 'zip', cfg.OUTPUT_DIR)
    print(f"\nAll outputs zipped into: {zip_filename}.zip")
    if FileLink and ipy_display:
        try:
            ipy_display(FileLink(f"{zip_filename}.zip"))
        except Exception:
            print(f"Download {zip_filename}.zip from the Kaggle output panel.")
    else:
        print(f"Download {zip_filename}.zip from the Kaggle output panel.")
