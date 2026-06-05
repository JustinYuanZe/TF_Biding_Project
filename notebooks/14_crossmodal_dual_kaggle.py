#!/usr/bin/env python3
"""
Script 14: Cross-Modal Attention Dual-Branch (DNABERT-2 + DNAshape CNN)
Designed for: Kaggle GPU (T4/P100) or Google Colab
Task: 4-class SP1/SP2/SP4/Negative TF-binding classification

ARCHITECTURAL LEAP OVER Script 12 (Val Acc ~50.33%):
  A. SpatialShapeCNN:
     Removes GlobalAvgPool → preserves positional information
     Output: [B, L_shape, d_model] instead of [B, 128]
  B. Full BERT Token Outputs:
     Uses all token embeddings [B, T, 768] → project to [B, T, d_model]
     Instead of [CLS]+MeanPool → single vector
  C. Bidirectional Cross-Modal Attention:
     1D→3D: Sequence queries structure ("what shape at this motif?")
     3D→1D: Structure queries sequence ("what nucleotide causes this fold?")
  D. Late Pooling:
     Position information preserved until AFTER cross-attention

ARCHITECTURE:
  Branch 1 — DNABERT-2:
    Full tokens [B, T, 768] → Linear(768→128) → [B, T, 128]

  Branch 2 — SpatialShapeCNN:
    [B, 5, 101] → Conv1D×3 (no GlobalPool) → [B, 25, 128]

  Cross-Modal Attention (Bidirectional):
    1D→3D: Q=seq[B,T,128], K/V=shape[B,25,128] → attended_seq[B,T,128]
    3D→1D: Q=shape[B,25,128], K/V=seq[B,T,128] → attended_shape[B,25,128]

  Fusion:
    MeanPool(attended_seq) ∥ MeanPool(attended_shape) → [B, 256]
    → LayerNorm → FC(256→256) → GELU → Dropout → FC(256→4)

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
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import math
import time
import random
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
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
)
from sklearn.preprocessing import label_binarize
from transformers import AutoTokenizer, AutoModel, AutoConfig

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()

# ── Initialize Accelerator ──
from accelerate import Accelerator
accelerator = Accelerator()
DEVICE = accelerator.device

# Redefine print globally to suppress non-main process logging in DDP/Multi-GPU
if not accelerator.is_main_process:
    import builtins
    builtins.print = lambda *args, **kwargs: None

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
    """Central configuration for Cross-Modal Attention Dual-Branch pipeline."""

    # ── Paths ──
    FASTA_DIR = auto_detect_dir("sp1_positive_final.fasta", "data/processed")
    SHAPE_DIR = auto_detect_dir("dnashape_sp1.npy", "data/processed")
    OUTPUT_DIR = "outputs_crossmodal_dual"
    FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

    # ── DNABERT-2 ──
    DNABERT_MODEL = "zhihan1996/DNABERT-2-117M"
    EMBEDDING_DIM = 768
    MAX_TOKEN_LENGTH = 512

    # ── Fine-Tuning Strategy ──
    UNFREEZE_LAST_N_LAYERS = 6
    BACKBONE_LR = 1.0e-5

    # ── Cross-Modal Attention (NEW) ──
    CROSS_ATTN_D_MODEL = 128      # Shared dimension for both branches
    CROSS_ATTN_NHEAD = 4           # Number of attention heads (128/4=32 dim/head)
    CROSS_ATTN_LAYERS = 1          # Number of cross-attention layers
    CROSS_ATTN_DROPOUT = 0.1       # Dropout in attention
    CROSS_ATTN_LR = 2e-4           # LR for cross-attention module
    SEQ_PROJ_LR = 2e-4             # LR for sequence projection

    # ── DNAshape Branch ──
    SHAPE_CHANNELS = 5             # MGW, ProT, Roll, HelT, EP
    SHAPE_SEQ_LEN = 101            # 101bp input sequences
    SHAPE_CONV_CHANNELS = [32, 64, 128]
    SHAPE_LR = 3.0e-4

    # ── Fusion Head ──
    HIDDEN_DIM = 256
    FUSION_DROPOUT = 0.3
    HEAD_LR = 5e-5
    LABEL_SMOOTHING = 0.1
    NUM_CLASSES = 4
    WEIGHT_DECAY = 0.1

    # ── Training ──
    BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = 4
    EPOCHS = 25
    PATIENCE = 10
    MAX_OVERFITTING_GAP = 30.0  # Max train-val gap (%) to prevent severe overfitting
    WARMUP_RATIO = 0.1

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
print(f"Architecture: Cross-Modal Attention Dual-Branch (DNABERT-2 + DNAshape)")
print(f"  * Cross-Attention d_model: {cfg.CROSS_ATTN_D_MODEL}")
print(f"  * Cross-Attention heads: {cfg.CROSS_ATTN_NHEAD}")
print(f"  * Cross-Attention layers: {cfg.CROSS_ATTN_LAYERS}")
print(f"  * Bidirectional: 1D<->3D")
print(f"Fine-tuning: Unfreeze last {cfg.UNFREEZE_LAST_N_LAYERS} layers")
print(f"Effective batch size: {cfg.BATCH_SIZE} x {cfg.GRAD_ACCUM_STEPS} = {cfg.BATCH_SIZE * cfg.GRAD_ACCUM_STEPS}")

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
    """Load pre-computed DNAshape feature matrices (.npy)."""
    all_shapes = []

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
        shape_data = np.load(fpath)
        print(f"  {cls_name} shape: {shape_data.shape} ({os.path.basename(fpath)})")
        all_shapes.append(shape_data)

    return np.concatenate(all_shapes, axis=0)


def load_all_data(fasta_dir, shape_dir, shape_files):
    """Load all 4 classes: sequences + shape features + group-aware labels."""
    print("=" * 60)
    print("LOADING DATASETS (Sequences + DNAshape Features)")
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
        "Negative": neg_fasta_path,
    }

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
    """Group-aware train/test split."""
    print("\n" + "=" * 60)
    print("SPLITTING DATA (GroupShuffleSplit — no revcomp leakage)")
    print("=" * 60)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(sequences, labels, groups))

    seq_train = [sequences[i] for i in train_idx]
    seq_test = [sequences[i] for i in test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]
    shape_train = shapes[train_idx]
    shape_test = shapes[test_idx]
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

del all_sequences, all_labels, all_groups, all_shapes, all_headers
gc.collect()

# ═══════════════════════════════════════════════════════════════════════
# CELL 4: DNAshape Normalization — Robust Scaler (P1-P99)
# ═══════════════════════════════════════════════════════════════════════

def robust_normalize_shapes(shape_train, shape_test):
    """Apply Robust Scaler normalization per channel. Stats from TRAINING SET ONLY."""
    print("\n" + "=" * 60)
    print("NORMALIZING DNAshape FEATURES (Robust Scaler P1-P99)")
    print("=" * 60)

    n_channels = shape_train.shape[1]
    channel_names = ["MGW", "ProT", "Roll", "HelT", "EP"]

    shape_train_norm = np.copy(shape_train).astype(np.float32)
    shape_test_norm = np.copy(shape_test).astype(np.float32)

    for ch in range(n_channels):
        train_vals = shape_train[:, ch, :].flatten()
        valid_vals = train_vals[~np.isnan(train_vals)]

        median_val = np.median(valid_vals)
        p1_val = np.percentile(valid_vals, 1)
        p99_val = np.percentile(valid_vals, 99)
        scale = p99_val - p1_val

        if scale < 1e-9:
            scale = 1.0

        shape_train_norm[:, ch, :] = (shape_train_norm[:, ch, :] - median_val) / scale
        shape_test_norm[:, ch, :] = (shape_test_norm[:, ch, :] - median_val) / scale

        print(f"  {channel_names[ch]:>5s}: median={median_val:>8.4f}, "
              f"P1={p1_val:>8.4f}, P99={p99_val:>8.4f}, scale={scale:>8.4f}")

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
        print(f"  Patched {patched} flash-attention references -> pure PyTorch.")
    else:
        print("  No flash-attention references found to patch (may already be clean).")

# ═══════════════════════════════════════════════════════════════════════
# CELL 6: Load DNABERT-2 Backbone (with Selective Unfreezing)
# ═══════════════════════════════════════════════════════════════════════

def load_dnabert2(model_name, device, unfreeze_last_n=3):
    """Load DNABERT-2 with selective layer unfreezing directly using state_dict load (avoiding meta device issues)."""
    print("\n" + "=" * 60)
    print("LOADING DNABERT-2 BACKBONE (Selective Fine-Tuning)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 3

    # Load empty config model directly to bypass meta-device errors on Kaggle
    with torch.no_grad():
        model = AutoModel.from_config(config, trust_remote_code=True)

    from huggingface_hub import hf_hub_download
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
    patch_flash_attention()

    # Selective Freezing / Unfreezing
    for param in model.parameters():
        param.requires_grad = False

    total_layers = len(model.encoder.layer)
    unfreeze_from = total_layers - unfreeze_last_n

    for i, layer in enumerate(model.encoder.layer):
        if i >= unfreeze_from:
            for param in layer.parameters():
                param.requires_grad = True

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n  DNABERT-2 loaded on {device}")
    print(f"  Total encoder layers: {total_layers}")
    print(f"  Frozen layers:    0 - {unfreeze_from - 1}")
    print(f"  Unfrozen layers:  {unfreeze_from} - {total_layers - 1} ({unfreeze_last_n} layers)")
    print(f"  Total parameters:     {total_params:>12,}")
    print(f"  Trainable parameters: {trainable_params:>12,} ({100*trainable_params/total_params:.1f}%)")

    return model, tokenizer


dnabert_model, tokenizer = load_dnabert2(
    cfg.DNABERT_MODEL, DEVICE, unfreeze_last_n=cfg.UNFREEZE_LAST_N_LAYERS
)

# ═══════════════════════════════════════════════════════════════════════
# CELL 7: Cross-Modal Attention Architecture (KEY CHANGE vs Script 12)
# ═══════════════════════════════════════════════════════════════════════

class SpatialShapeCNN(nn.Module):
    """
    Conv1D on DNAshape features, PRESERVING spatial dimension.
    NO GlobalAveragePooling — returns a sequence of vectors.

    Input:  [B, 5, 101]
    Output: [B, L_shape, d_model]  where L_shape=25 (after 2x MaxPool)
    """
    def __init__(self, in_channels=5, conv_channels=None, d_model=128):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 128]

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels[0], kernel_size=7, padding=3),
            nn.BatchNorm1d(conv_channels[0]),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
        )  # [B, 32, 50]

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels[1]),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
        )  # [B, 64, 25]

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels[2]),
            nn.GELU(),
        )  # [B, 128, 25]

        # Project to d_model if needed
        if conv_channels[-1] != d_model:
            self.proj = nn.Linear(conv_channels[-1], d_model)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: [B, 5, 101] — normalized shape features
        Returns:
            [B, L_shape, d_model] — spatial structural embeddings
        """
        x = self.conv_block1(x)   # [B, 32, 50]
        x = self.conv_block2(x)   # [B, 64, 25]
        x = self.conv_block3(x)   # [B, 128, 25]
        x = x.transpose(1, 2)    # [B, 25, 128]
        x = self.proj(x)         # [B, 25, d_model]
        return x


class FeedForward(nn.Module):
    """Post-attention Feed-Forward Network with residual connection."""
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
    """
    Single layer of Bidirectional Cross-Modal Attention.

    1D->3D: Sequence queries structural features
    3D->1D: Structure queries sequence features

    Each direction uses:
      - MultiheadAttention (cross)
      - Residual + LayerNorm
      - FeedForward (with residual)
    """
    def __init__(self, d_model=128, nhead=4, dropout=0.1):
        super().__init__()
        # 1D -> 3D direction: sequence asks about structure
        self.cross_attn_seq2shape = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm_seq = nn.LayerNorm(d_model)
        self.ffn_seq = FeedForward(d_model, expansion=4, dropout=dropout)

        # 3D -> 1D direction: structure asks about sequence
        self.cross_attn_shape2seq = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm_shape = nn.LayerNorm(d_model)
        self.ffn_shape = FeedForward(d_model, expansion=4, dropout=dropout)

    def forward(self, seq_features, shape_features, seq_key_padding_mask=None):
        """
        Args:
            seq_features:   [B, T, d_model]  — projected DNABERT-2 token embeddings
            shape_features: [B, L, d_model]  — spatial ShapeCNN embeddings
            seq_key_padding_mask: [B, T] bool — True for padding positions

        Returns:
            seq_out:   [B, T, d_model]  — sequence enriched by structure
            shape_out: [B, L, d_model]  — structure enriched by sequence
        """
        # 1D -> 3D: sequence queries structure
        attended_seq, _ = self.cross_attn_seq2shape(
            query=seq_features, key=shape_features, value=shape_features
        )
        seq_out = self.norm_seq(seq_features + attended_seq)
        seq_out = self.ffn_seq(seq_out)

        # 3D -> 1D: structure queries sequence
        attended_shape, _ = self.cross_attn_shape2seq(
            query=shape_features, key=seq_features, value=seq_features,
            key_padding_mask=seq_key_padding_mask
        )
        shape_out = self.norm_shape(shape_features + attended_shape)
        shape_out = self.ffn_shape(shape_out)

        return seq_out, shape_out


class CrossModalDualBranchClassifier(nn.Module):
    """
    Cross-Modal Attention Dual-Branch Classifier.

    Instead of simple concatenation (Script 12), the two branches
    communicate through bidirectional cross-attention BEFORE pooling.

    Branch 1 — DNABERT-2:
      Full token outputs [B, T, 768] -> Linear(768, d_model) -> [B, T, d_model]

    Branch 2 — SpatialShapeCNN:
      [B, 5, 101] -> Conv1D x3 (no pool) -> [B, 25, d_model]

    Cross-Modal Attention:
      Bidirectional: seq <-> shape

    Fusion:
      MeanPool(attended_seq) || MeanPool(attended_shape) -> [B, 2*d_model]
      -> LayerNorm -> FC -> GELU -> Dropout -> FC(num_classes)
    """
    def __init__(self, backbone, embedding_dim=768,
                 shape_in_channels=5, shape_conv_channels=None,
                 d_model=128, nhead=4, num_cross_layers=1, cross_dropout=0.1,
                 hidden_dim=256, num_classes=4, fusion_dropout=0.3):
        super().__init__()
        self.backbone = backbone
        self.d_model = d_model

        # ── Branch 1: DNABERT-2 token projection ──
        self.seq_projection = nn.Sequential(
            nn.Linear(embedding_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # ── Branch 2: SpatialShapeCNN ──
        if shape_conv_channels is None:
            shape_conv_channels = [32, 64, 128]
        self.shape_cnn = SpatialShapeCNN(
            in_channels=shape_in_channels,
            conv_channels=shape_conv_channels,
            d_model=d_model,
        )

        # ── Cross-Modal Attention stack ──
        self.cross_attention_layers = nn.ModuleList([
            CrossModalAttentionLayer(d_model=d_model, nhead=nhead, dropout=cross_dropout)
            for _ in range(num_cross_layers)
        ])

        # ── Fusion Classifier ──
        fusion_dim = d_model * 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=fusion_dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids, attention_mask, shape_features):
        """
        Args:
            input_ids:      [B, T]
            attention_mask: [B, T]
            shape_features: [B, 5, 101]
        Returns:
            logits: [B, num_classes]
        """
        # ── Branch 1: DNABERT-2 full token outputs ──
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]  # [B, T, 768]

        seq_features = self.seq_projection(hidden_states)  # [B, T, d_model]

        # ── Branch 2: SpatialShapeCNN ──
        shape_feats = self.shape_cnn(shape_features)  # [B, 25, d_model]

        # ── Cross-Modal Attention ──
        # Create padding mask: True where padding (attention_mask == 0)
        seq_key_padding_mask = (attention_mask == 0)

        for cross_layer in self.cross_attention_layers:
            seq_features, shape_feats = cross_layer(
                seq_features, shape_feats,
                seq_key_padding_mask=seq_key_padding_mask
            )

        # ── Late Pooling (AFTER cross-attention) ──
        # Sequence: masked mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
        seq_pooled = (seq_features * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        # [B, d_model]

        # Shape: simple mean pooling
        shape_pooled = shape_feats.mean(dim=1)  # [B, d_model]

        # ── Fusion + Classification ──
        fused = torch.cat([seq_pooled, shape_pooled], dim=1)  # [B, 2*d_model]

        return self.classifier(fused)


# ═══════════════════════════════════════════════════════════════════════
# CELL 8: DualBranchDataset & DataLoaders
# ═══════════════════════════════════════════════════════════════════════

class DualBranchDataset(Dataset):
    """Dataset providing tokenized DNA sequences and DNAshape features."""
    def __init__(self, sequences, labels, shape_features, tokenizer, max_length=512):
        self.sequences = sequences
        self.labels = labels
        self.shape_features = shape_features
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.sequences[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        shape = torch.tensor(self.shape_features[idx], dtype=torch.float32)

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
print(f"Gradient accumulation: {cfg.GRAD_ACCUM_STEPS} steps -> effective batch = {cfg.BATCH_SIZE * cfg.GRAD_ACCUM_STEPS}")

# ═══════════════════════════════════════════════════════════════════════
# CELL 9: Build Model & Optimizer with 5-Group Differential LR
# ═══════════════════════════════════════════════════════════════════════

model = CrossModalDualBranchClassifier(
    backbone=dnabert_model,
    embedding_dim=cfg.EMBEDDING_DIM,
    shape_in_channels=cfg.SHAPE_CHANNELS,
    shape_conv_channels=cfg.SHAPE_CONV_CHANNELS,
    d_model=cfg.CROSS_ATTN_D_MODEL,
    nhead=cfg.CROSS_ATTN_NHEAD,
    num_cross_layers=cfg.CROSS_ATTN_LAYERS,
    cross_dropout=cfg.CROSS_ATTN_DROPOUT,
    hidden_dim=cfg.HIDDEN_DIM,
    num_classes=cfg.NUM_CLASSES,
    fusion_dropout=cfg.FUSION_DROPOUT,
)
model = model.to(DEVICE)

# ── 5-Group Parameter Separation ──
backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
seq_proj_params = list(model.seq_projection.parameters())
shape_params = list(model.shape_cnn.parameters())
cross_attn_params = list(model.cross_attention_layers.parameters())
head_params = list(model.classifier.parameters())

optimizer = optim.AdamW([
    {"params": backbone_params,    "lr": cfg.BACKBONE_LR,    "weight_decay": cfg.WEIGHT_DECAY},
    {"params": seq_proj_params,    "lr": cfg.SEQ_PROJ_LR,    "weight_decay": cfg.WEIGHT_DECAY},
    {"params": shape_params,       "lr": cfg.SHAPE_LR,       "weight_decay": cfg.WEIGHT_DECAY},
    {"params": cross_attn_params,  "lr": cfg.CROSS_ATTN_LR,  "weight_decay": cfg.WEIGHT_DECAY},
    {"params": head_params,        "lr": cfg.HEAD_LR,        "weight_decay": cfg.WEIGHT_DECAY},
])

# Prepare model, optimizer, and loaders first with accelerator
model, optimizer, train_loader, test_loader = accelerator.prepare(
    model, optimizer, train_loader, test_loader
)

total_steps = (len(train_loader) // cfg.GRAD_ACCUM_STEPS) * cfg.EPOCHS
warmup_steps = int(total_steps * cfg.WARMUP_RATIO)

def lr_lambda(current_step):
    """Linear warmup then cosine decay."""
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scheduler = accelerator.prepare(scheduler)

# Class weights
class_counts = np.bincount(y_train)
total_count = len(y_train)
num_classes = len(class_counts)

class_weights = total_count / (num_classes * class_counts.astype(np.float32))
class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
print(f"  Class weights: " + ", ".join([f"{cfg.CLASS_NAMES[i]}={class_weights[i]:.4f}" for i in range(num_classes)]))

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.LABEL_SMOOTHING)

# Print model summary
n_backbone = sum(p.numel() for p in backbone_params)
n_seq_proj = sum(p.numel() for p in seq_proj_params)
n_shape = sum(p.numel() for p in shape_params)
n_cross = sum(p.numel() for p in cross_attn_params)
n_head = sum(p.numel() for p in head_params)
n_total = n_backbone + n_seq_proj + n_shape + n_cross + n_head

print(f"\n{'='*60}")
print("MODEL SUMMARY -- Cross-Modal Attention Dual-Branch")
print(f"{'='*60}")
print(f"  DNABERT-2 backbone (trainable): {n_backbone:>10,} params (LR={cfg.BACKBONE_LR})")
print(f"  Seq Projection (768->{cfg.CROSS_ATTN_D_MODEL}):     {n_seq_proj:>10,} params (LR={cfg.SEQ_PROJ_LR})")
print(f"  SpatialShapeCNN:                {n_shape:>10,} params (LR={cfg.SHAPE_LR})")
print(f"  * CrossModalAttention:          {n_cross:>10,} params (LR={cfg.CROSS_ATTN_LR})")
print(f"  Fusion classifier:              {n_head:>10,} params (LR={cfg.HEAD_LR})")
print(f"  -----------------------------------------")
print(f"  Total trainable:                {n_total:>10,} params")
print(f"  Total training steps:           {total_steps}")
print(f"  Warmup steps:                   {warmup_steps}")
print(f"\n  * KEY CHANGES vs Script 12:")
print(f"    ShapeCNN:       GlobalAvgPool REMOVED -> spatial features preserved")
print(f"    BERT output:    [CLS]+MeanPool -> Full token outputs [B, T, {cfg.CROSS_ATTN_D_MODEL}]")
print(f"    Fusion:         torch.cat -> Bidirectional Cross-Attention ({cfg.CROSS_ATTN_NHEAD} heads)")
print(f"    Late Pooling:   MeanPool AFTER cross-attention (not before)")

# ═══════════════════════════════════════════════════════════════════════
# CELL 10: Training Loop with Gradient Accumulation & Early Stopping
# ═══════════════════════════════════════════════════════════════════════

def train_one_epoch(model, train_loader, optimizer, scheduler, criterion,
                    accelerator, grad_accum_steps):
    """Train for one epoch with gradient accumulation and clean 1% progress logging using Accelerator."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()

    total_steps = len(train_loader)
    last_percent = -1

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        shape_features = batch["shape_features"]
        labels = batch["labels"]

        with accelerator.accumulate(model):
            with accelerator.autocast():
                logits = model(input_ids, attention_mask, shape_features)
                loss = criterion(logits, labels)

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Update training metrics locally (not gathered to save communication overhead, identical to Script 20)
        running_loss += loss.item() * labels.size(0)
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update and print progress every 1% increment with a visual progress bar (only on main process)
        percent = int((step + 1) / total_steps * 100)
        if percent > last_percent:
            if accelerator.is_main_process:
                bar_len = 20
                filled_len = int(bar_len * percent / 100)
                bar = "█" * filled_len + "░" * (bar_len - filled_len)
                sys.stdout.write(f"\r  Training: [{bar}] {percent}% | Loss: {running_loss/total:.4f} | Acc: {correct/total:.4f}   ")
                sys.stdout.flush()
            last_percent = percent

    # Clear the temporary training line
    if accelerator.is_main_process:
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, test_loader, criterion, accelerator):
    """Evaluate on validation/test set using Accelerator to gather metrics."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in test_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        shape_features = batch["shape_features"]
        labels = batch["labels"]

        with accelerator.autocast():
            logits = model(input_ids, attention_mask, shape_features)
            loss = criterion(logits, labels)

        _, predicted = logits.max(1)

        # Gather predictions, labels, and losses across all processes
        predicted, labels_gathered = accelerator.gather_for_metrics((predicted, labels))
        loss_gathered = accelerator.gather_for_metrics(loss.repeat(labels.size(0)))

        running_loss += loss_gathered.sum().item()
        total += labels_gathered.size(0)
        correct += predicted.eq(labels_gathered).sum().item()

    return running_loss / max(total, 1), correct / max(total, 1)


def train_model(model, train_loader, test_loader, optimizer, scheduler,
                criterion, accelerator, cfg):
    """Full training loop with early stopping on val_acc."""
    print("\n" + "=" * 60)
    print("TRAINING -- Cross-Modal Attention Dual-Branch")
    print("=" * 60)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(cfg.EPOCHS):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            accelerator, cfg.GRAD_ACCUM_STEPS,
        )

        val_loss, val_acc = evaluate(model, test_loader, criterion, accelerator)

        elapsed = time.time() - t0

                # Synchronize training metrics across all processes for consistent logging and stopping decisions
        train_loss_tensor = torch.tensor(train_loss, device=accelerator.device)
        train_acc_tensor = torch.tensor(train_acc, device=accelerator.device)
        train_loss = accelerator.gather(train_loss_tensor).mean().item()
        train_acc = accelerator.gather(train_acc_tensor).mean().item()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Synchronize before printing and saving checkpoint
        accelerator.wait_for_everyone()

        backbone_lr = optimizer.param_groups[0]["lr"]
        cross_lr = optimizer.param_groups[3]["lr"]
        head_lr = optimizer.param_groups[4]["lr"]

        gap_percent = (train_acc - val_acc) * 100
        print(
            f"Epoch {epoch+1:02d}/{cfg.EPOCHS} | "
            f"Train: {train_loss:.4f}/{train_acc:.4f} | "
            f"Val: {val_loss:.4f}/{val_acc:.4f} | "
            f"Gap: {gap_percent:+.2f}% | "
            f"LR: {backbone_lr:.1e} | {elapsed:.0f}s"
        )

        if gap_percent >= cfg.MAX_OVERFITTING_GAP:
            print(f"\n  ⏹ Early stopping at epoch {epoch+1} due to severe overfitting (Gap: {gap_percent:+.2f}% >= {cfg.MAX_OVERFITTING_GAP}%)")
            break

        if val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save checkpoint only on the main process
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                ckpt_path = os.path.join(cfg.MODEL_DIR, "best_crossmodal_dual.pt")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": unwrapped_model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }, ckpt_path)
                print(f"  -> Saved best model (val_loss={val_loss:.4f}, val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1} (patience={cfg.PATIENCE})")
                break

        # Synchronize after checkpoint operations
        accelerator.wait_for_everyone()

    print(f"\nTraining complete.")
    print(f"  Best val_loss: {best_val_loss:.4f}")
    print(f"  Best val_acc:  {best_val_acc:.4f}")
    return history


history = train_model(
    model, train_loader, test_loader, optimizer, scheduler,
    criterion, accelerator, cfg,
)

# ═══════════════════════════════════════════════════════════════════════
# CELL 11: Load Best Model & Full Evaluation
# ═══════════════════════════════════════════════════════════════════════

best_path = os.path.join(cfg.MODEL_DIR, "best_crossmodal_dual.pt")
checkpoint = torch.load(best_path, map_location=DEVICE)
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Loaded best model from epoch {checkpoint['epoch']} "
      f"(val_loss={checkpoint['val_loss']:.4f}, val_acc={checkpoint['val_acc']:.4f})")


@torch.no_grad()
def full_evaluation(model, test_loader, class_names, accelerator):
    """Run full evaluation and return predictions/probabilities using Accelerator."""
    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    total_batches = len(test_loader)
    last_percent = -1

    for step, batch in enumerate(test_loader):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        shape_features = batch["shape_features"]
        labels = batch["labels"]

        with accelerator.autocast():
            logits = model(input_ids, attention_mask, shape_features)
        probs = torch.softmax(logits.float(), dim=1)
        _, preds = logits.max(1)

        # Gather predictions, labels, and probabilities across all processes
        preds_g, labels_g, probs_g = accelerator.gather_for_metrics((preds, labels, probs))

        all_preds.extend(preds_g.cpu().numpy())
        all_targets.extend(labels_g.cpu().numpy())
        all_probs.extend(probs_g.cpu().numpy())

        # Update and print progress every 1% increment with a visual progress bar
        percent = int((step + 1) / total_batches * 100)
        if percent > last_percent:
            if accelerator.is_main_process:
                bar_len = 20
                filled_len = int(bar_len * percent / 100)
                bar = "█" * filled_len + "░" * (bar_len - filled_len)
                sys.stdout.write(f"\r  Evaluating: [{bar}] {percent}%   ")
                sys.stdout.flush()
            last_percent = percent

    # Clear the temporary evaluation line
    if accelerator.is_main_process:
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        report_str = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
        print(report_str)

        report_path = os.path.join(cfg.OUTPUT_DIR, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write("CLASSIFICATION REPORT -- Cross-Modal Attention Dual-Branch\n")
            f.write("=" * 60 + "\n")
            f.write(report_str)
        print(f"  -> Classification report saved to: {report_path}")

        acc = accuracy_score(all_targets, all_preds)
        f1_macro = f1_score(all_targets, all_preds, average="macro")
        print(f"Overall Accuracy:  {acc:.4f}")
        print(f"Macro F1-Score:    {f1_macro:.4f}")

    return all_preds, all_targets, all_probs


all_preds, all_targets, all_probs = full_evaluation(model, test_loader, cfg.CLASS_NAMES, accelerator)

# ── Post-processing: Export BED files for IGV Analysis ──
def parse_header_to_bed(header):
    try:
        clean_hdr = header.split("_")[0]
        chrom, coords = clean_hdr.split(":")
        start, end = coords.split("-")
        return chrom, start, end
    except Exception:
        return None

def export_igv_bed_files(headers_test, predictions, targets, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    true_sp1_coords = []
    true_sp4_coords = []
    confused_sp4_as_sp1_coords = []

    for idx, (pred, target) in enumerate(zip(predictions, targets)):
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

    print(f"\n  Exported BED files for IGV analysis to {output_dir}:")
    print(f"    - True_SP1.bed: {len(true_sp1_coords)} lines")
    print(f"    - True_SP4.bed: {len(true_sp4_coords)} lines")
    print(f"    - Confused_SP4_as_SP1.bed: {len(confused_sp4_as_sp1_coords)} lines")

if accelerator.is_main_process:
    export_igv_bed_files(headers_test, all_preds, all_targets, cfg.OUTPUT_DIR)

# ═══════════════════════════════════════════════════════════════════════
# CELL 12: Generate All Performance Figures
# ═══════════════════════════════════════════════════════════════════════

TITLE_PREFIX = "Cross-Modal Attention (DNABERT-2 + DNAshape)"

def plot_training_curves(history, save_dir):
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

    plt.suptitle(f"{TITLE_PREFIX} -- Training Progress", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "crossmodal_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Training curves: {path}")


def plot_confusion_matrix(all_targets, all_preds, class_names, save_dir):
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
    path = os.path.join(save_dir, "crossmodal_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Confusion matrix: {path}")


def plot_roc_curves(all_targets, all_probs, class_names, save_dir):
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
    plt.title(f"ROC Curves -- {TITLE_PREFIX}", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "crossmodal_roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> ROC curves: {path}")


def plot_precision_recall_curves(all_targets, all_probs, class_names, save_dir):
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
    plt.title(f"Precision-Recall Curves -- {TITLE_PREFIX}", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "crossmodal_pr_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> PR curves: {path}")


def plot_per_class_metrics_bar(all_targets, all_preds, class_names, save_dir):
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
    ax.set_title(f"Per-Class Performance -- {TITLE_PREFIX}", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(save_dir, "crossmodal_per_class_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Per-class metrics: {path}")


if accelerator.is_main_process:
    print("\n" + "=" * 60)
    print("GENERATING PERFORMANCE FIGURES")
    print("=" * 60)

    plot_training_curves(history, cfg.FIG_DIR)
    plot_confusion_matrix(all_targets, all_preds, cfg.CLASS_NAMES, cfg.FIG_DIR)
    plot_roc_curves(all_targets, all_probs, cfg.CLASS_NAMES, cfg.FIG_DIR)
    plot_precision_recall_curves(all_targets, all_probs, cfg.CLASS_NAMES, cfg.FIG_DIR)
    plot_per_class_metrics_bar(all_targets, all_preds, cfg.CLASS_NAMES, cfg.FIG_DIR)

    print("\nAll figures saved to:", cfg.FIG_DIR)

# ═══════════════════════════════════════════════════════════════════════
# CELL 13: Summary & Comparison
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PIPELINE SUMMARY -- Cross-Modal Attention Dual-Branch")
print("=" * 60)
print(f"  Model:             Cross-Modal Attention Dual-Branch")
print(f"  DNABERT-2:         Last {cfg.UNFREEZE_LAST_N_LAYERS} layers fine-tuned")
print(f"  Seq Projection:    {cfg.EMBEDDING_DIM} -> {cfg.CROSS_ATTN_D_MODEL}")
print(f"  SpatialShapeCNN:   Conv1D({cfg.SHAPE_CHANNELS}->{cfg.SHAPE_CONV_CHANNELS}) -> [B, 25, {cfg.CROSS_ATTN_D_MODEL}]")
print(f"  Cross-Attention:   {cfg.CROSS_ATTN_LAYERS} layer(s), {cfg.CROSS_ATTN_NHEAD} heads, d={cfg.CROSS_ATTN_D_MODEL}")
print(f"  Fusion:            MeanPool(seq) || MeanPool(shape) -> FC({cfg.HIDDEN_DIM}) -> {cfg.NUM_CLASSES}")
print(f"  Training samples:  {len(seq_train)}")
print(f"  Test samples:      {len(seq_test)}")
print(f"  Best val acc:      {max(history['val_acc']):.4f}")
print(f"  Figures saved to:  {cfg.FIG_DIR}")
print(f"  Model saved to:    {cfg.MODEL_DIR}")
print()
print("  +-----------------------------------------------------------+")
print("  |  COMPARISON -- Script 12 vs Script 14                     |")
print("  +-----------------------------------------------------------+")
print("  |  Script 12: Balanced Dual-Branch (Simple Concatenation)   |")
print("  |    -> Val Acc: ~50.33%  (torch.cat fusion)                |")
print("  |                                                           |")
print("  |  Script 14: Cross-Modal Attention Dual-Branch             |")
print(f"  |    -> Val Acc: {max(history['val_acc']):.2%}  (Bidirectional Cross-Attention) |")
print("  +-----------------------------------------------------------+")
print("=" * 60)
