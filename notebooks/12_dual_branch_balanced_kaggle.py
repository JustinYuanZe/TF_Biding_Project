#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  Script 12: Balanced Dual-Branch DNABERT-2 + DNAshape CNN          ║
║  Designed for: Kaggle GPU (T4/P100) or Google Colab                ║
║  Task: 4-class SP1/SP2/SP4/Negative TF-binding classification     ║
╚══════════════════════════════════════════════════════════════════════╝

IMPROVEMENTS OVER Script 11 (Val Acc ~47%):
  A. Dimensionality Bottleneck:
     BERT 1536-dim → Projection → 256-dim (eliminates 12:1 imbalance)
  B. Fusion LayerNorm:
     LayerNorm applied AFTER concatenation to equalize variance
  C. Asymmetric Dropout:
     Heavy dropout (0.5) on BERT branch, light (0.1) on Shape branch
     Forces classifier to attend to structural features
  D. Tuned Differential Learning Rates:
     Backbone: 2e-5 | Shape: 5e-4 | Projection+Head: 2e-4
     Slower head prevents premature convergence on BERT features

ARCHITECTURE:
  Branch 1 — DNABERT-2:
    [CLS]+MeanPool → 1536-dim → Projection(256) → Dropout(0.5)

  Branch 2 — DNAshape CNN:
    5×101 → Conv1D(5→32→64→128) → GlobalAvgPool → 128-dim → Dropout(0.1)

  Fusion:
    Concat(256 + 128) = 384-dim → LayerNorm → FC(256) → GELU
    → Dropout(0.3) → FC(4)

NORMALIZATION — Robust Scaler (P1-P99):
  Same as Script 11 — statistics from TRAINING SET ONLY.

Required packages (auto-installed):
  pip install transformers einops datasets accelerate safetensors
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
import gc
import math
import time
import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

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
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ═══════════════════════════════════════════════════════════════════════
# CELL 2: Configuration
# ═══════════════════════════════════════════════════════════════════════

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
    """Search for the directory containing target_file in Kaggle input or local path."""
    resolved_path = find_file(target_file, fallback)
    if resolved_path:
        return os.path.dirname(resolved_path)
    return fallback

class Config:
    """Central configuration for the Balanced Dual-Branch pipeline."""

    # ── Paths ──
    FASTA_DIR = auto_detect_dir("sp1_positive_final.fasta", "data/processed")
    SHAPE_DIR = auto_detect_dir("dnashape_sp1.npy", "data/processed")
    OUTPUT_DIR = "outputs_balanced_dual"
    FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

    # ── DNABERT-2 ──
    DNABERT_MODEL = "zhihan1996/DNABERT-2-117M"
    EMBEDDING_DIM = 768
    MAX_TOKEN_LENGTH = 512

    # ── Fine-Tuning Strategy ──
    UNFREEZE_LAST_N_LAYERS = 6    # Unfreeze last 6 of 12 encoder layers
    BACKBONE_LR = 1.0e-5          # Lower backbone LR for stability with more unfrozen layers
    SHAPE_LR = 3.0e-4             # Increased to 3e-4 to allow ShapeCNN to train from scratch faster
    HEAD_LR = 5e-5                # Reduced from 2e-4 to 5e-5 (slower head learning)
    WEIGHT_DECAY = 0.1            # Increased from 0.05 to 0.1 (stronger regularization)

    # ── DNAshape Branch ──
    SHAPE_CHANNELS = 5            # MGW, ProT, Roll, HelT, EP
    SHAPE_SEQ_LEN = 101           # 101bp input sequences
    SHAPE_CONV_CHANNELS = [32, 64, 128]  # Conv1D channel progression
    SHAPE_OUTPUT_DIM = 128        # Final shape embedding dimension

    # ── Projection Layer (NEW — Technique A) ──
    BERT_PROJ_DIM = 256           # ★ NEW: Compress BERT 1536 → 256
    BERT_DROPOUT = 0.5            # ★ NEW: Heavy dropout on BERT branch (Technique C)
    SHAPE_DROPOUT = 0.1           # ★ NEW: Light dropout on Shape branch (Technique C)

    # ── Fusion Head ──
    FUSION_DIM = 256 + 128        # ★ CHANGED: 256 (projected BERT) + 128 (Shape) = 384
    HIDDEN_DIM = 256
    DROPOUT_RATE = 0.3            # Fusion head dropout
    LABEL_SMOOTHING = 0.1
    NUM_CLASSES = 4

    # ── Training ──
    BATCH_SIZE = 16               # Per-GPU batch size (T4-friendly)
    GRAD_ACCUM_STEPS = 4          # Effective batch = 16 × 4 = 64
    EPOCHS = 25                   # Increased epochs to 25
    PATIENCE = 10                 # Increased patience to 10
    WARMUP_RATIO = 0.1            # 10% of total steps for warmup

    # ── Data Split ──
    TEST_SIZE = 0.2
    RANDOM_SEED = 42

    # ── Class Names ──
    CLASS_NAMES = ["SP1", "SP2", "SP4", "Negative"]

    # ── DNAshape NPY files ──
    SHAPE_FILES = {
        "SP1": "dnashape_sp1.npy",
        "SP2": "dnashape_sp2.npy",
        "SP4": "dnashape_sp4.npy",
        "Negative": "dnashape_negative.npy",
    }

cfg = Config()

# Create output directories
for d in [cfg.OUTPUT_DIR, cfg.FIG_DIR, cfg.MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# Set seeds for reproducibility
random.seed(cfg.RANDOM_SEED)
np.random.seed(cfg.RANDOM_SEED)
torch.manual_seed(cfg.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {DEVICE}")
print(f"Architecture: Balanced Dual-Branch (DNABERT-2 + DNAshape CNN)")
print(f"  ★ BERT Projection: 1536 → {cfg.BERT_PROJ_DIM}")
print(f"  ★ Asymmetric Dropout: BERT={cfg.BERT_DROPOUT}, Shape={cfg.SHAPE_DROPOUT}")
print(f"  ★ Fusion dim: {cfg.FUSION_DIM} (ratio {cfg.BERT_PROJ_DIM}:{cfg.SHAPE_OUTPUT_DIM} ≈ {cfg.BERT_PROJ_DIM/cfg.SHAPE_OUTPUT_DIM:.1f}:1)")
print(f"  ★ Head LR: {cfg.HEAD_LR} (5× slower than Script 11)")
print(f"Fine-tuning: Unfreeze last {cfg.UNFREEZE_LAST_N_LAYERS} layers")
print(f"Effective batch size: {cfg.BATCH_SIZE} × {cfg.GRAD_ACCUM_STEPS} = {cfg.BATCH_SIZE * cfg.GRAD_ACCUM_STEPS}")

# ═══════════════════════════════════════════════════════════════════════
# CELL 3: Data Loading (Sequences + DNAshape Features)
# ═══════════════════════════════════════════════════════════════════════

def load_fasta(filepath):
    """Load DNA sequences and their headers from a FASTA file."""
    sequences = []
    headers = []
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing dataset: {filepath}")
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


def load_shape_features(data_dir, shape_files):
    """
    Load pre-computed DNAshape feature matrices (.npy).
    Each file has shape [N_class, 5, 101].
    Returns concatenated array in class order: SP1, SP2, SP4, Negative.
    """
    all_shapes = []
    
    # Dynamically find the negative shape file if it exists under another name
    neg_shape_path = None
    neg_candidates = ["dnashape_negative_genomic.npy", "dnashape_negative_cpg.npy", "dnashape_negative.npy"]
    for cand in neg_candidates:
        path = find_file(cand, data_dir)
        if path:
            neg_shape_path = path
            break
            
    if not neg_shape_path:
        raise FileNotFoundError("Could not find any negative DNAshape file among candidates.")
            
    print(f"  [Auto-detect] Using negative shape file: {neg_shape_path}")

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
        shape_data = np.load(fpath)  # [N, 5, 101]
        print(f"  {cls_name} shape: {shape_data.shape} ({os.path.basename(fpath)})")
        all_shapes.append(shape_data)
        
    return np.concatenate(all_shapes, axis=0)  # [N_total, 5, 101]


def load_all_data(fasta_dir, shape_dir, shape_files):
    """Load all 4 classes: sequences + shape features + group-aware labels."""
    print("=" * 60)
    print("LOADING DATASETS (Sequences + DNAshape Features)")
    print("=" * 60)

    # Detect negative FASTA file
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

    # Build file paths for all classes
    fasta_files = {
        "SP1": find_file("sp1_positive_final.fasta", fasta_dir),
        "SP2": find_file("sp2_positive_final.fasta", fasta_dir),
        "SP4": find_file("sp4_positive_final.fasta", fasta_dir),
        "Negative": neg_fasta_path,
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

        # Build group IDs:
        # Positive classes: sequences are in orig/revcomp PAIRS
        # Negative class: each sequence is independent
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

    # Load shape features in the same class order
    print("\n  Loading DNAshape features...")
    all_shapes = load_shape_features(shape_dir, shape_files)

    assert len(all_sequences) == all_shapes.shape[0], (
        f"Sequence count ({len(all_sequences)}) != shape count ({all_shapes.shape[0]})"
    )

    print(f"\n  Total: {len(all_sequences)} sequences, {group_id} groups")
    print(f"  Shape features: {all_shapes.shape}")
    print(f"  Class distribution: {np.bincount(all_labels)}")

    return all_sequences, all_labels, all_groups, all_shapes, all_headers


def split_data(sequences, labels, groups, shapes, headers, test_size=0.2, seed=42):
    """
    Group-aware train/test split.
    Ensures orig/revcomp pairs stay in the SAME split → no data leakage.
    """
    print("\n" + "=" * 60)
    print("SPLITTING DATA (GroupShuffleSplit — no revcomp leakage)")
    print("=" * 60)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(sequences, labels, groups))

    seq_train = [sequences[i] for i in train_idx]
    seq_test = [sequences[i] for i in test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]
    shape_train = shapes[train_idx]  # [N_train, 5, 101]
    shape_test = shapes[test_idx]    # [N_test, 5, 101]
    headers_train = [headers[i] for i in train_idx]
    headers_test = [headers[i] for i in test_idx]

    print(f"  Train: {len(seq_train)} sequences")
    print(f"  Test:  {len(seq_test)} sequences")
    print(f"  Train class dist: {np.bincount(y_train)}")
    print(f"  Test  class dist: {np.bincount(y_test)}")

    return seq_train, seq_test, y_train, y_test, shape_train, shape_test, headers_train, headers_test


# Load and split
all_sequences, all_labels, all_groups, all_shapes, all_headers = load_all_data(
    cfg.FASTA_DIR, cfg.SHAPE_DIR, cfg.SHAPE_FILES
)
seq_train, seq_test, y_train, y_test, shape_train, shape_test, headers_train, headers_test = split_data(
    all_sequences, all_labels, all_groups, all_shapes, all_headers,
    test_size=cfg.TEST_SIZE, seed=cfg.RANDOM_SEED,
)

# Free memory
del all_sequences, all_labels, all_groups, all_shapes, all_headers
gc.collect()

# ═══════════════════════════════════════════════════════════════════════
# CELL 4: DNAshape Normalization — Robust Scaler (P1-P99)
# ═══════════════════════════════════════════════════════════════════════

def robust_normalize_shapes(shape_train, shape_test):
    """
    Apply Robust Scaler normalization per channel:
      x_norm = (x - median) / (P99 - P1)

    Statistics are computed from TRAINING SET ONLY to prevent data leakage.
    NaN values at sequence boundaries are filled with 0 (= median post-norm).

    Args:
        shape_train: np.ndarray [N_train, 5, 101]
        shape_test:  np.ndarray [N_test, 5, 101]

    Returns:
        shape_train_norm, shape_test_norm: normalized arrays (same shapes)
    """
    print("\n" + "=" * 60)
    print("NORMALIZING DNAshape FEATURES (Robust Scaler P1-P99)")
    print("=" * 60)

    n_channels = shape_train.shape[1]
    channel_names = ["MGW", "ProT", "Roll", "HelT", "EP"]

    shape_train_norm = np.copy(shape_train).astype(np.float32)
    shape_test_norm = np.copy(shape_test).astype(np.float32)

    for ch in range(n_channels):
        # Flatten channel across all train samples, ignoring NaN
        train_vals = shape_train[:, ch, :].flatten()
        valid_vals = train_vals[~np.isnan(train_vals)]

        median_val = np.median(valid_vals)
        p1_val = np.percentile(valid_vals, 1)
        p99_val = np.percentile(valid_vals, 99)
        scale = p99_val - p1_val

        if scale < 1e-9:
            scale = 1.0  # Safety: avoid division by zero

        # Normalize
        shape_train_norm[:, ch, :] = (shape_train_norm[:, ch, :] - median_val) / scale
        shape_test_norm[:, ch, :] = (shape_test_norm[:, ch, :] - median_val) / scale

        print(f"  {channel_names[ch]:>5s}: median={median_val:>8.4f}, "
              f"P1={p1_val:>8.4f}, P99={p99_val:>8.4f}, scale={scale:>8.4f}")

    # Fill NaN with 0 (median maps to 0 after robust scaling)
    nan_count_train = np.isnan(shape_train_norm).sum()
    nan_count_test = np.isnan(shape_test_norm).sum()
    shape_train_norm = np.nan_to_num(shape_train_norm, nan=0.0)
    shape_test_norm = np.nan_to_num(shape_test_norm, nan=0.0)

    print(f"\n  NaN filled with 0: train={nan_count_train}, test={nan_count_test}")
    print(f"  Train shape range: [{shape_train_norm.min():.4f}, {shape_train_norm.max():.4f}]")
    print(f"  Test  shape range: [{shape_test_norm.min():.4f}, {shape_test_norm.max():.4f}]")

    return shape_train_norm, shape_test_norm


shape_train_norm, shape_test_norm = robust_normalize_shapes(shape_train, shape_test)
del shape_train, shape_test
gc.collect()

# ═══════════════════════════════════════════════════════════════════════
# CELL 5: DNABERT-2 Flash Attention Patch (Pure PyTorch, No Triton)
# ═══════════════════════════════════════════════════════════════════════

def _pytorch_flash_attn_qkvpacked(qkv, bias=None, causal=False, softmax_scale=None):
    """Drop-in replacement for FlashAttnQKVPackedFunc (no Triton needed)."""
    q, k, v = qkv.unbind(dim=2)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    d = q.shape[-1]
    scale = softmax_scale if softmax_scale is not None else (d ** -0.5)
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    if bias is not None:
        attn = attn + bias
    if causal:
        S = q.shape[2]
        mask = torch.triu(torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float("-inf"))
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(attn, v)
    return out.transpose(1, 2).contiguous()


def _pytorch_flash_attn_kvpacked(q, kv, bias=None, causal=False, softmax_scale=None):
    """Drop-in replacement for FlashAttnKVPackedFunc."""
    k, v = kv.unbind(dim=2)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    d = q.shape[-1]
    scale = softmax_scale if softmax_scale is not None else (d ** -0.5)
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    if bias is not None:
        attn = attn + bias
    if causal:
        Sq, Sk = q.shape[2], k.shape[2]
        mask = torch.triu(torch.ones(Sq, Sk, device=q.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float("-inf"))
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(attn, v)
    return out.transpose(1, 2).contiguous()


def _pytorch_flash_attn_func(q, k, v, bias=None, causal=False, softmax_scale=None):
    """Drop-in replacement for FlashAttnFunc."""
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    d = q.shape[-1]
    scale = softmax_scale if softmax_scale is not None else (d ** -0.5)
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    if bias is not None:
        attn = attn + bias
    if causal:
        Sq, Sk = q.shape[2], k.shape[2]
        mask = torch.triu(torch.ones(Sq, Sk, device=q.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float("-inf"))
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(attn, v)
    return out.transpose(1, 2).contiguous()


def patch_flash_attention():
    """
    Monkey-patch cached HuggingFace modules referencing DNABERT-2's
    flash_attn_triton → use pure-PyTorch replacements.
    """
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
    if patched:
        print(f"  ✅ Patched {patched} flash-attention references → pure PyTorch.")
    else:
        print("  ⚠️  No flash-attention references found to patch (may already be clean).")

# ═══════════════════════════════════════════════════════════════════════
# CELL 6: Load DNABERT-2 Backbone (with Selective Unfreezing)
# ═══════════════════════════════════════════════════════════════════════

def load_dnabert2(model_name, device, unfreeze_last_n=3):
    """
    Load DNABERT-2 with selective layer unfreezing.
    Freezes layers 0 to (12 - unfreeze_last_n - 1), unfreezes the rest.
    """
    print("\n" + "=" * 60)
    print("LOADING DNABERT-2 BACKBONE (Selective Fine-Tuning)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 3

    model = None

    # Strategy 1: Direct load
    try:
        model = AutoModel.from_pretrained(
            model_name, config=config,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        )
        for name, param in model.named_parameters():
            if param.device == torch.device("meta"):
                raise RuntimeError(f"Parameter {name} on meta device")
        for name, buf in model.named_buffers():
            if buf.device == torch.device("meta"):
                raise RuntimeError(f"Buffer {name} on meta device")
        print("  Strategy 1 (direct load) ✅")
    except Exception as e:
        print(f"  Strategy 1 failed: {e}")
        model = None

    # Strategy 2: Empty init + manual state_dict
    if model is None:
        try:
            print("  Trying Strategy 2 (empty init + state_dict)...")
            from huggingface_hub import hf_hub_download

            with torch.no_grad():
                model = AutoModel.from_config(config, trust_remote_code=True)

            try:
                weight_file = hf_hub_download(repo_id=model_name, filename="model.safetensors")
                from safetensors.torch import load_file
                state_dict = load_file(weight_file, device="cpu")
            except Exception:
                weight_file = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
                state_dict = torch.load(weight_file, map_location="cpu", weights_only=False)

            clean_sd = {}
            for k, v in state_dict.items():
                clean_sd[k[5:] if k.startswith("bert.") else k] = v

            result = model.load_state_dict(clean_sd, strict=False)
            critical = ["embeddings.word_embeddings.weight",
                        "encoder.layer.0.attention.self.Wqkv.weight"]
            for ck in critical:
                if ck in set(result.missing_keys):
                    raise RuntimeError(f"Core weight '{ck}' missing!")

            model = model.to("cpu")
            print("  Strategy 2 ✅")
        except Exception as e:
            print(f"  Strategy 2 failed: {e}")
            model = None

    # Strategy 3: Monkey-patch torch.empty meta→cpu
    if model is None:
        try:
            print("  Trying Strategy 3 (monkey-patch ALiBi)...")
            _orig_empty = torch.empty
            def _patched_empty(*args, **kwargs):
                if kwargs.get("device") == torch.device("meta") or str(kwargs.get("device", "")) == "meta":
                    kwargs["device"] = "cpu"
                return _orig_empty(*args, **kwargs)
            torch.empty = _patched_empty
            try:
                model = AutoModel.from_pretrained(
                    model_name, config=config,
                    trust_remote_code=True,
                    low_cpu_mem_usage=False,
                )
            finally:
                torch.empty = _orig_empty
            print("  Strategy 3 ✅")
        except Exception as e:
            raise RuntimeError(
                f"All loading strategies failed. Last error: {e}\n"
                "Try: pip install --upgrade transformers torch safetensors"
            ) from e

    # Patch flash attention
    patch_flash_attention()

    # ── Selective Freezing / Unfreezing ──
    for param in model.parameters():
        param.requires_grad = False

    total_layers = len(model.encoder.layer)
    unfreeze_from = total_layers - unfreeze_last_n

    frozen_count = 0
    unfrozen_count = 0

    for i, layer in enumerate(model.encoder.layer):
        if i >= unfreeze_from:
            for param in layer.parameters():
                param.requires_grad = True
                unfrozen_count += param.numel()
        else:
            for param in layer.parameters():
                frozen_count += param.numel()

    for param in model.embeddings.parameters():
        frozen_count += param.numel()

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n  DNABERT-2 loaded on {device}")
    print(f"  Total encoder layers: {total_layers}")
    print(f"  Frozen layers:    0 – {unfreeze_from - 1} (embeddings + {unfreeze_from} layers)")
    print(f"  Unfrozen layers:  {unfreeze_from} – {total_layers - 1} ({unfreeze_last_n} layers)")
    print(f"  Total parameters:     {total_params:>12,}")
    print(f"  Frozen parameters:    {frozen_count:>12,}")
    print(f"  Trainable parameters: {trainable_params:>12,} ({100*trainable_params/total_params:.1f}%)")

    return model, tokenizer


dnabert_model, tokenizer = load_dnabert2(
    cfg.DNABERT_MODEL, DEVICE, unfreeze_last_n=cfg.UNFREEZE_LAST_N_LAYERS
)

# ═══════════════════════════════════════════════════════════════════════
# CELL 7: Balanced DualBranch Architecture (★ KEY CHANGES)
# ═══════════════════════════════════════════════════════════════════════

class ShapeCNN(nn.Module):
    """
    1D Convolutional network for DNAshape structural features.

    Input:  [batch, 5, 101]  (5 shape channels × 101 positions)
    Output: [batch, 128]     (compact structural embedding)

    Architecture:
      Conv1D(5→32, k=7) → BN → ReLU → MaxPool(2)
      Conv1D(32→64, k=5) → BN → ReLU → MaxPool(2)
      Conv1D(64→128, k=3) → BN → ReLU
      → GlobalAveragePooling → 128-dim
    """
    def __init__(self, in_channels=5, conv_channels=None, output_dim=128):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 128]

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels[0], kernel_size=7, padding=3),
            nn.BatchNorm1d(conv_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels[2]),
            nn.ReLU(inplace=True),
        )
        # Global Average Pooling → output_dim
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Optional projection if conv_channels[-1] != output_dim
        if conv_channels[-1] != output_dim:
            self.proj = nn.Linear(conv_channels[-1], output_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: [batch, 5, 101] — normalized shape features
        Returns:
            [batch, output_dim] — structural embedding
        """
        x = self.conv_block1(x)   # [batch, 32, 50]
        x = self.conv_block2(x)   # [batch, 64, 25]
        x = self.conv_block3(x)   # [batch, 128, 25]
        x = self.global_pool(x)   # [batch, 128, 1]
        x = x.squeeze(-1)         # [batch, 128]
        x = self.proj(x)          # [batch, output_dim]
        return x


class BalancedDualBranchClassifier(nn.Module):
    """
    ★ BALANCED Dual-Branch architecture (Script 12).

    Key differences from Script 11's DualBranchClassifier:
      A. Projection Layer compresses BERT 1536 → 256 dim
      B. LayerNorm applied AFTER concat for variance equalization
      C. Asymmetric Dropout: heavy on BERT (0.5), light on Shape (0.1)
      D. Slower head LR (configured externally in optimizer)

    Branch 1 — DNABERT-2:
      Input: tokenized DNA sequences
      Output: [CLS]+MeanPool → 1536 → Projection(256) → Dropout(0.5)

    Branch 2 — ShapeCNN:
      Input: normalized shape matrix [5, 101]
      Output: 128-dim → Dropout(0.1)

    Fusion:
      Concat(256 + 128) = 384-dim
      → LayerNorm → Linear(384→256) → GELU → Dropout(0.3) → Linear(256→4)
    """
    def __init__(self, backbone, embedding_dim=768, shape_cfg=None,
                 bert_proj_dim=256, bert_dropout=0.5, shape_dropout=0.1,
                 hidden_dim=256, num_classes=4, fusion_dropout=0.3):
        super().__init__()
        self.backbone = backbone

        # ── Technique A: BERT Projection Layer ──
        pooled_dim = embedding_dim * 2  # [CLS] + MeanPool = 1536
        self.bert_projection = nn.Sequential(
            nn.Linear(pooled_dim, bert_proj_dim),
            nn.LayerNorm(bert_proj_dim),
            nn.GELU(),
        )

        # ── Technique C: Asymmetric Dropout ──
        self.bert_dropout = nn.Dropout(p=bert_dropout)    # Heavy: 0.5
        self.shape_dropout = nn.Dropout(p=shape_dropout)  # Light: 0.1

        # Shape branch
        shape_in_channels = shape_cfg.get("in_channels", 5) if shape_cfg else 5
        shape_conv_channels = shape_cfg.get("conv_channels", [32, 64, 128]) if shape_cfg else [32, 64, 128]
        shape_output_dim = shape_cfg.get("output_dim", 128) if shape_cfg else 128

        self.shape_cnn = ShapeCNN(
            in_channels=shape_in_channels,
            conv_channels=shape_conv_channels,
            output_dim=shape_output_dim,
        )

        # ── Technique B: Fusion with LayerNorm AFTER concat ──
        fusion_dim = bert_proj_dim + shape_output_dim  # 256 + 128 = 384

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),         # Technique B: equalize variance
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=fusion_dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids, attention_mask, shape_features):
        """
        Args:
            input_ids:      [batch, seq_len]
            attention_mask: [batch, seq_len]
            shape_features: [batch, 5, 101]
        Returns:
            logits: [batch, num_classes]
        """
        # ── Branch 1: DNABERT-2 → Projection → Dropout ──
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]  # (batch, seq_len, 768)

        # [CLS] token
        cls_token = hidden_states[:, 0, :]  # (batch, 768)

        # Mean pooling (excluding padding tokens)
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_pooled = sum_hidden / count  # (batch, 768)

        bert_raw = torch.cat([cls_token, mean_pooled], dim=1)  # (batch, 1536)

        # ★ Technique A: Project 1536 → 256
        bert_out = self.bert_projection(bert_raw)  # (batch, 256)
        # ★ Technique C: Heavy dropout on BERT
        bert_out = self.bert_dropout(bert_out)      # (batch, 256)

        # ── Branch 2: ShapeCNN → Dropout ──
        shape_out = self.shape_cnn(shape_features)  # (batch, 128)
        # ★ Technique C: Light dropout on Shape
        shape_out = self.shape_dropout(shape_out)    # (batch, 128)

        # ── Fusion (★ Technique B: LayerNorm inside classifier) ──
        fused = torch.cat([bert_out, shape_out], dim=1)  # (batch, 384)

        return self.classifier(fused)

# ═══════════════════════════════════════════════════════════════════════
# CELL 8: DualBranchDataset & DataLoaders
# ═══════════════════════════════════════════════════════════════════════

class DualBranchDataset(Dataset):
    """
    Dataset that provides both tokenized DNA sequences and
    pre-computed/normalized DNAshape feature matrices.
    """
    def __init__(self, sequences, labels, shape_features, tokenizer, max_length=512):
        self.sequences = sequences
        self.labels = labels
        self.shape_features = shape_features  # np.ndarray [N, 5, 101]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Tokenize sequence for DNABERT-2
        encoding = self.tokenizer(
            self.sequences[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Shape features (already normalized)
        shape = torch.tensor(self.shape_features[idx], dtype=torch.float32)  # [5, 101]

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "shape_features": shape,
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


train_dataset = DualBranchDataset(
    seq_train, y_train, shape_train_norm, tokenizer, cfg.MAX_TOKEN_LENGTH
)
test_dataset = DualBranchDataset(
    seq_test, y_test, shape_test_norm, tokenizer, cfg.MAX_TOKEN_LENGTH
)

train_loader = DataLoader(
    train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
    num_workers=2, pin_memory=True, drop_last=True,
)
test_loader = DataLoader(
    test_dataset, batch_size=cfg.BATCH_SIZE * 2, shuffle=False,
    num_workers=2, pin_memory=True,
)

print(f"DataLoaders ready: {len(train_loader)} train batches, {len(test_loader)} test batches")
print(f"Gradient accumulation: {cfg.GRAD_ACCUM_STEPS} steps → effective batch = {cfg.BATCH_SIZE * cfg.GRAD_ACCUM_STEPS}")

# ═══════════════════════════════════════════════════════════════════════
# CELL 9: Build Model & Optimizer with 4-Group Differential LR
# ═══════════════════════════════════════════════════════════════════════

shape_cfg = {
    "in_channels": cfg.SHAPE_CHANNELS,
    "conv_channels": cfg.SHAPE_CONV_CHANNELS,
    "output_dim": cfg.SHAPE_OUTPUT_DIM,
}

model = BalancedDualBranchClassifier(
    backbone=dnabert_model,
    embedding_dim=cfg.EMBEDDING_DIM,
    shape_cfg=shape_cfg,
    bert_proj_dim=cfg.BERT_PROJ_DIM,
    bert_dropout=cfg.BERT_DROPOUT,
    shape_dropout=cfg.SHAPE_DROPOUT,
    hidden_dim=cfg.HIDDEN_DIM,
    num_classes=cfg.NUM_CLASSES,
    fusion_dropout=cfg.DROPOUT_RATE,
)
model = model.to(DEVICE)

# ── 4-Group Parameter Separation ──
# Group 1: DNABERT-2 backbone (unfrozen layers only) — lowest LR
backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
# Group 2: ShapeCNN branch — medium LR (train from scratch)
shape_params = list(model.shape_cnn.parameters())
# Group 3: BERT Projection layer — same LR as head (new parameters)
proj_params = list(model.bert_projection.parameters())
# Group 4: Fusion classifier head — controlled LR
head_params = list(model.classifier.parameters())

optimizer = optim.AdamW([
    {"params": backbone_params, "lr": cfg.BACKBONE_LR, "weight_decay": cfg.WEIGHT_DECAY},
    {"params": shape_params,    "lr": cfg.SHAPE_LR,    "weight_decay": cfg.WEIGHT_DECAY},
    {"params": proj_params,     "lr": cfg.HEAD_LR,     "weight_decay": cfg.WEIGHT_DECAY},
    {"params": head_params,     "lr": cfg.HEAD_LR,     "weight_decay": cfg.WEIGHT_DECAY},
])

# Total training steps and warmup
total_steps = (len(train_loader) // cfg.GRAD_ACCUM_STEPS) * cfg.EPOCHS
warmup_steps = int(total_steps * cfg.WARMUP_RATIO)

def lr_lambda(current_step):
    """Linear warmup then cosine decay."""
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Compute class weights for 4-class classification imbalance (SP1, SP2, SP4, Negative)
class_counts = np.bincount(y_train)
total_count = len(y_train)
num_classes = len(class_counts)

class_weights = total_count / (num_classes * class_counts.astype(np.float32))
class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
print(f"  [Auto-detect] Class weights for loss function: " + ", ".join([f"{cfg.CLASS_NAMES[i]}={class_weights[i]:.4f}" for i in range(num_classes)]))

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.LABEL_SMOOTHING)
scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

# Print model summary
backbone_trainable = sum(p.numel() for p in backbone_params)
shape_trainable = sum(p.numel() for p in shape_params)
proj_trainable = sum(p.numel() for p in proj_params)
head_trainable = sum(p.numel() for p in head_params)
total_trainable = backbone_trainable + shape_trainable + proj_trainable + head_trainable

print(f"\n{'='*60}")
print("MODEL SUMMARY — Balanced Dual-Branch Architecture")
print(f"{'='*60}")
print(f"  DNABERT-2 backbone (trainable): {backbone_trainable:>10,} params (LR={cfg.BACKBONE_LR})")
print(f"  ShapeCNN branch:                {shape_trainable:>10,} params (LR={cfg.SHAPE_LR})")
print(f"  ★ BERT Projection (1536→256):   {proj_trainable:>10,} params (LR={cfg.HEAD_LR})")
print(f"  Fusion head:                    {head_trainable:>10,} params (LR={cfg.HEAD_LR})")
print(f"  ─────────────────────────────────────────")
print(f"  Total trainable:                {total_trainable:>10,} params")
print(f"  Total training steps:           {total_steps}")
print(f"  Warmup steps:                   {warmup_steps}")
print(f"  Label smoothing:                {cfg.LABEL_SMOOTHING}")
print(f"\n  ★ KEY CHANGES vs Script 11:")
print(f"    Projection:      1536→{cfg.BERT_PROJ_DIM} (was: no projection)")
print(f"    Fusion dim:      {cfg.FUSION_DIM} (was: 1664)")
print(f"    BERT dropout:    {cfg.BERT_DROPOUT} (was: 0)")
print(f"    Shape dropout:   {cfg.SHAPE_DROPOUT} (was: 0)")
print(f"    Head LR:         {cfg.HEAD_LR} (was: 1e-3)")
print(f"    Weight decay:    {cfg.WEIGHT_DECAY} (was: 0.01)")
print(f"    Epochs:          {cfg.EPOCHS} (was: 15)")
print(f"    Patience:        {cfg.PATIENCE} (was: 5)")

# ═══════════════════════════════════════════════════════════════════════
# CELL 10: Training Loop with Gradient Accumulation & Early Stopping
# ═══════════════════════════════════════════════════════════════════════

def train_one_epoch(model, train_loader, optimizer, scheduler, criterion,
                    scaler, device, grad_accum_steps):
    """Train for one epoch with gradient accumulation (dual-input)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc="  Training", leave=False)
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        shape_features = batch["shape_features"].to(device)
        labels = batch["labels"].to(device)

        with autocast(enabled=(device.type == "cuda")):
            logits = model(input_ids, attention_mask, shape_features)
            loss = criterion(logits, labels)
            loss = loss / grad_accum_steps  # Scale loss for accumulation

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        running_loss += loss.item() * grad_accum_steps * labels.size(0)
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            "loss": f"{running_loss/total:.4f}",
            "acc": f"{correct/total:.4f}",
        })

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    """Evaluate on validation/test set (dual-input)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        shape_features = batch["shape_features"].to(device)
        labels = batch["labels"].to(device)

        with autocast(enabled=(device.type == "cuda")):
            logits = model(input_ids, attention_mask, shape_features)
            loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def train_model(model, train_loader, test_loader, optimizer, scheduler,
                criterion, scaler, cfg, device):
    """Full training loop with early stopping."""
    print("\n" + "=" * 60)
    print("TRAINING — Balanced Dual-Branch (DNABERT-2 + DNAshape CNN)")
    print("=" * 60)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(cfg.EPOCHS):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            scaler, device, cfg.GRAD_ACCUM_STEPS,
        )

        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        backbone_lr = optimizer.param_groups[0]["lr"]
        shape_lr = optimizer.param_groups[1]["lr"]
        head_lr = optimizer.param_groups[3]["lr"]

        print(
            f"Epoch {epoch+1:02d}/{cfg.EPOCHS} │ "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} │ "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} │ "
            f"LR: {backbone_lr:.1e}/{shape_lr:.1e}/{head_lr:.1e} │ {elapsed:.1f}s"
        )

        # Checkpoint based on val_acc (accuracy-based early stopping)
        if val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            ckpt_path = os.path.join(cfg.MODEL_DIR, "best_balanced_dual.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, ckpt_path)
            print(f"  → Saved best model (val_loss={val_loss:.4f}, val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                print(f"\n  ⏹ Early stopping at epoch {epoch+1} (patience={cfg.PATIENCE})")
                break

    print(f"\nTraining complete.")
    print(f"  Best val_loss: {best_val_loss:.4f}")
    print(f"  Best val_acc:  {best_val_acc:.4f}")
    return history


history = train_model(
    model, train_loader, test_loader, optimizer, scheduler,
    criterion, scaler, cfg, DEVICE,
)

# ═══════════════════════════════════════════════════════════════════════
# CELL 11: Load Best Model & Full Evaluation
# ═══════════════════════════════════════════════════════════════════════

best_path = os.path.join(cfg.MODEL_DIR, "best_balanced_dual.pt")
checkpoint = torch.load(best_path, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()
print(f"Loaded best model from epoch {checkpoint['epoch']} "
      f"(val_loss={checkpoint['val_loss']:.4f}, val_acc={checkpoint['val_acc']:.4f})")


@torch.no_grad()
def full_evaluation(model, test_loader, class_names, device):
    """Run full evaluation and return predictions/probabilities (dual-input)."""
    all_preds, all_targets, all_probs = [], [], []

    for batch in tqdm(test_loader, desc="  Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        shape_features = batch["shape_features"].to(device)
        labels = batch["labels"]

        with autocast(enabled=(device.type == "cuda")):
            logits = model(input_ids, attention_mask, shape_features)
        probs = torch.softmax(logits.float(), dim=1)
        _, preds = logits.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report_str = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
    print(report_str)

    # ★ Save classification report to file
    report_path = os.path.join(cfg.OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("CLASSIFICATION REPORT — Balanced Dual-Branch\n")
        f.write("=" * 60 + "\n")
        f.write(report_str)
    print(f"  → Classification report saved to: {report_path}")

    acc = accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average="macro")
    print(f"Overall Accuracy:  {acc:.4f}")
    print(f"Macro F1-Score:    {f1_macro:.4f}")

    return all_preds, all_targets, all_probs


all_preds, all_targets, all_probs = full_evaluation(model, test_loader, cfg.CLASS_NAMES, DEVICE)

# ── Post-processing: Export BED files for IGV Analysis ──
def parse_header_to_bed(header):
    """Parse FASTA header (e.g., 'chr8:142777203-142777304_revcomp') to BED format fields."""
    try:
        clean_hdr = header.split("_")[0]
        chrom, coords = clean_hdr.split(":")
        start, end = coords.split("-")
        return chrom, start, end
    except Exception:
        return None

def export_igv_bed_files(headers_test, predictions, targets, output_dir):
    """Export BED files for true predictions and confused predictions for IGV analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    true_sp1_coords = []
    true_sp4_coords = []
    confused_sp4_as_sp1_coords = []
    
    # Class indexing: SP1=0, SP2=1, SP4=2, Negative=3
    for idx, (pred, target) in enumerate(zip(predictions, targets)):
        header = headers_test[idx]
        bed_fields = parse_header_to_bed(header)
        if not bed_fields:
            continue
            
        chrom, start, end = bed_fields
        bed_line = f"{chrom}\t{start}\t{end}\n"
        
        if target == 0 and pred == 0:  # True SP1
            true_sp1_coords.append(bed_line)
        elif target == 2 and pred == 2:  # True SP4
            true_sp4_coords.append(bed_line)
        elif target == 2 and pred == 0:  # SP4 misclassified as SP1
            confused_sp4_as_sp1_coords.append(bed_line)
            
    # Write BED files
    with open(os.path.join(output_dir, "True_SP1.bed"), "w") as f:
        f.writelines(true_sp1_coords)
    with open(os.path.join(output_dir, "True_SP4.bed"), "w") as f:
        f.writelines(true_sp4_coords)
    with open(os.path.join(output_dir, "Confused_SP4_as_SP1.bed"), "w") as f:
        f.writelines(confused_sp4_as_sp1_coords)
        
    print(f"\n  Exported BED files for IGV analysis to {output_dir}:")
    print(f"    - True_SP1.bed: {len(true_sp1_coords)} lines")
    print(f"    - True_SP4.bed: {len(true_sp4_coords)} lines")
    print(f"    - Confused_SP4_as_SP1.bed: {len(confused_sp4_as_sp1_coords)} lines")

export_igv_bed_files(headers_test, all_preds, all_targets, cfg.OUTPUT_DIR)

# ═══════════════════════════════════════════════════════════════════════
# CELL 12: Generate All Performance Figures
# ═══════════════════════════════════════════════════════════════════════

TITLE_PREFIX = "Balanced Dual-Branch (DNABERT-2 + DNAshape)"

def plot_training_curves(history, save_dir):
    """Plot loss and accuracy curves."""
    epochs = len(history["train_loss"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(range(1, epochs+1), history["train_loss"], label="Train Loss",
             color="#2196F3", linewidth=2)
    ax1.plot(range(1, epochs+1), history["val_loss"], label="Val Loss",
             color="#FF5722", linewidth=2)
    ax1.set_title("Loss Convergence", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.plot(range(1, epochs+1), history["train_acc"], label="Train Acc",
             color="#4CAF50", linewidth=2)
    ax2.plot(range(1, epochs+1), history["val_acc"], label="Val Acc",
             color="#E91E63", linewidth=2)
    ax2.axhline(y=0.25, color="gray", linestyle="--", alpha=0.5, label="Random Guess")
    ax2.set_title("Accuracy Performance", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle(f"{TITLE_PREFIX} — Training Progress", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "balanced_dual_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Training curves: {path}")


def plot_confusion_matrix(all_targets, all_preds, class_names, save_dir):
    """Plot confusion matrix with annotations."""
    cm = confusion_matrix(all_targets, all_preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, data, title, fmt in [
        (ax1, cm, "Confusion Matrix (Counts)", "d"),
        (ax2, cm_norm, "Confusion Matrix (Normalized)", ".2%"),
    ]:
        im = ax.imshow(data, cmap="Blues", interpolation="nearest")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        thresh = data.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, format(data[i, j], fmt),
                        ha="center", va="center",
                        color="white" if data[i, j] > thresh else "black",
                        fontsize=12, fontweight="bold")

    plt.suptitle(TITLE_PREFIX, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "balanced_dual_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Confusion matrix: {path}")


def plot_roc_curves(all_targets, all_probs, class_names, save_dir):
    """Plot per-class and macro-average ROC curves."""
    n_classes = len(class_names)
    y_bin = label_binarize(all_targets, classes=list(range(n_classes)))

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    plt.figure(figsize=(9, 7))

    fpr_dict, tpr_dict, auc_dict = {}, {}, {}
    for i in range(n_classes):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_bin[:, i], all_probs[:, i])
        auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
        plt.plot(fpr_dict[i], tpr_dict[i], color=colors[i], linewidth=2,
                 label=f"{class_names[i]} (AUC = {auc_dict[i]:.4f})")

    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr, color="#9C27B0", linewidth=2.5, linestyle="--",
             label=f"Macro Average (AUC = {macro_auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Guess (0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curves — {TITLE_PREFIX}", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "balanced_dual_roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → ROC curves: {path}")


def plot_precision_recall_curves(all_targets, all_probs, class_names, save_dir):
    """Plot per-class Precision-Recall curves."""
    n_classes = len(class_names)
    y_bin = label_binarize(all_targets, classes=list(range(n_classes)))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    plt.figure(figsize=(9, 7))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], all_probs[:, i])
        ap = average_precision_score(y_bin[:, i], all_probs[:, i])
        plt.plot(recall, precision, color=colors[i], linewidth=2,
                 label=f"{class_names[i]} (AP = {ap:.4f})")

    plt.axhline(y=0.25, color="gray", linestyle="--", alpha=0.5, label="Random Baseline")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"Precision-Recall Curves — {TITLE_PREFIX}", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "balanced_dual_pr_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → PR curves: {path}")


def plot_per_class_metrics_bar(all_targets, all_preds, class_names, save_dir):
    """Per-class bar chart of Precision, Recall, F1."""
    prec, rec, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, average=None
    )

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, prec, width, label="Precision", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x, rec, width, label="Recall", color="#4CAF50", alpha=0.85)
    bars3 = ax.bar(x + width, f1, width, label="F1-Score", color="#FF9800", alpha=0.85)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{cn}\n(n={s})" for cn, s in zip(class_names, support)])
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Per-Class Performance — {TITLE_PREFIX}", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(save_dir, "balanced_dual_per_class_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Per-class metrics: {path}")


print("\n" + "=" * 60)
print("GENERATING PERFORMANCE FIGURES")
print("=" * 60)

plot_training_curves(history, cfg.FIG_DIR)
plot_confusion_matrix(all_targets, all_preds, cfg.CLASS_NAMES, cfg.FIG_DIR)
plot_roc_curves(all_targets, all_probs, cfg.CLASS_NAMES, cfg.FIG_DIR)
plot_precision_recall_curves(all_targets, all_probs, cfg.CLASS_NAMES, cfg.FIG_DIR)
plot_per_class_metrics_bar(all_targets, all_preds, cfg.CLASS_NAMES, cfg.FIG_DIR)

print("\n✅ All figures saved to:", cfg.FIG_DIR)

# ═══════════════════════════════════════════════════════════════════════
# CELL 13: Summary & Comparison
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PIPELINE SUMMARY — Balanced Dual-Branch Architecture")
print("=" * 60)
print(f"  Model:             Balanced Dual-Branch (DNABERT-2 + DNAshape CNN)")
print(f"  DNABERT-2:         Last {cfg.UNFREEZE_LAST_N_LAYERS} layers fine-tuned")
print(f"  ★ BERT Projection: {cfg.EMBEDDING_DIM*2} → {cfg.BERT_PROJ_DIM}")
print(f"  ★ Asymm. Dropout:  BERT={cfg.BERT_DROPOUT}, Shape={cfg.SHAPE_DROPOUT}")
print(f"  ShapeCNN:          Conv1D({cfg.SHAPE_CHANNELS}→{cfg.SHAPE_CONV_CHANNELS}) → {cfg.SHAPE_OUTPUT_DIM}-dim")
print(f"  Fusion:            Concat({cfg.BERT_PROJ_DIM}+{cfg.SHAPE_OUTPUT_DIM})={cfg.FUSION_DIM} → FC({cfg.HIDDEN_DIM}) → {cfg.NUM_CLASSES}")
print(f"  Normalization:     Robust Scaler (P1-P99) on shape features")
print(f"  Tokenizer:         BPE (Byte Pair Encoding)")
print(f"  Attention:         ALiBi (patched to pure PyTorch)")
print(f"  Data split:        GroupShuffleSplit (revcomp-aware)")
print(f"  Training samples:  {len(seq_train)}")
print(f"  Test samples:      {len(seq_test)}")
print(f"  Backbone LR:       {cfg.BACKBONE_LR}")
print(f"  ShapeCNN LR:       {cfg.SHAPE_LR}")
print(f"  Head LR:           {cfg.HEAD_LR}")
print(f"  Weight decay:      {cfg.WEIGHT_DECAY}")
print(f"  Label smoothing:   {cfg.LABEL_SMOOTHING}")
print(f"  Grad accumulation: {cfg.GRAD_ACCUM_STEPS} steps")
print(f"  Best val loss:     {min(history['val_loss']):.4f}")
print(f"  Best val acc:      {max(history['val_acc']):.4f}")
print(f"  Figures saved to:  {cfg.FIG_DIR}")
print(f"  Model saved to:    {cfg.MODEL_DIR}")
print()
print("  ┌─────────────────────────────────────────────────────────┐")
print("  │  COMPARISON — Script 10 vs 11 vs 12                    │")
print("  ├─────────────────────────────────────────────────────────┤")
print("  │  Script 10: DNABERT-2 only                             │")
print("  │    → Val Acc: ~48.38%                                  │")
print("  │                                                         │")
print("  │  Script 11: DNABERT-2 + DNAshape (Unbalanced)          │")
print("  │    → Val Acc: ~47.05%  (12:1 dim ratio)                │")
print("  │                                                         │")
print("  │  Script 12: DNABERT-2 + DNAshape (★ Balanced)          │")
print(f"  │    → Val Acc: {max(history['val_acc']):.2%}   (2:1 dim ratio)                │")
print("  └─────────────────────────────────────────────────────────┘")
print("=" * 60)
