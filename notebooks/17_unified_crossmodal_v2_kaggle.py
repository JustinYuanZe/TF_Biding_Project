#!/usr/bin/env python3
"""
Script 17: Unified Cross-Modal Attention v2 — Critical Bug Fixes
Designed for: Kaggle GPU (T4/P100)
Task: 4-class SP1/SP2/SP4/Negative TF-binding classification

═══════════════════════════════════════════════════════════════════════════
WHAT WENT WRONG IN SCRIPT 16 (Accuracy stuck at 25% = random chance):
═══════════════════════════════════════════════════════════════════════════

  BUG 1: HEAD_LR = 5e-5 → too small.  The classifier head couldn't learn
         decision boundaries fast enough. Other heads at 2-3e-4.
         FIX: HEAD_LR = 2e-4 (match other components)

  BUG 2: Triple-stacked loss regularization:
         Focal(gamma=1) + label_smoothing=0.1 + class_weights → conflicting
         gradients, washing out the signal on a balanced dataset.
         FIX: Use ONLY weighted CrossEntropy. No focal. No label smoothing.
              Class weights are mild (all ~1.0) so they're fine.

  BUG 3: d_model=128 bottleneck. DNABERT-2 outputs 768-dim embeddings.
         Projecting to 128-dim loses 83% of information BEFORE cross-attention.
         FIX: d_model=256 (keeps 33% of info, 2x more than before)

  BUG 4: Sequence projection had dropout=0.1 INSIDE the projection, adding
         noise to already-bottlenecked features.
         FIX: Remove dropout from seq_projection, keep dropout only at fusion.

  BUG 5: Warmup was only 10% of total steps (422 steps out of 4225).
         With layer-wise LR decay already slowing backbone, warmup was
         too short for the head to "catch up".
         FIX: Warmup ratio 0.1 → 0.15 (gives ~630 steps of warmup)

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
import copy
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
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
          if hasattr(torch.cuda.get_device_properties(0), 'total_mem')
          else f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.backends.cudnn.benchmark = True

# ═══════════════════════════════════════════════════════════════════════
# CELL 2: Configuration
# ═══════════════════════════════════════════════════════════════════════

def find_file(filename, fallback_dir="data/processed"):
    """Search for target_file in absolute paths, Kaggle input, fallback dirs, or CWD."""
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if filename in files:
                fpath = os.path.join(root, filename)
                print(f"  [Auto-detect] Found {filename} at {fpath}")
                return fpath
    if fallback_dir and os.path.exists(fallback_dir):
        p1 = os.path.join(fallback_dir, filename)
        if os.path.exists(p1):
            return p1
        p2 = os.path.join(fallback_dir, "fixed_negative", filename)
        if os.path.exists(p2):
            return p2
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
    """Script 17: Unified Cross-Modal Attention v2 — Bug Fixes."""

    # ── Paths ──
    FASTA_DIR = auto_detect_dir("sp1_positive_final.fasta", "data/processed")
    SHAPE_DIR = auto_detect_dir("dnashape_sp1.npy", "data/processed")
    OUTPUT_DIR = "outputs_unified_crossmodal_v2"
    FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

    # ── DNABERT-2 ──
    DNABERT_MODEL = "zhihan1996/DNABERT-2-117M"
    EMBEDDING_DIM = 768

    # ── Sequence length ──
    AUTO_MAX_LENGTH = True
    MAX_TOKEN_LENGTH = 48
    MAX_LENGTH_CAP = 96
    MAX_LENGTH_FLOOR = 32

    # ── Fine-Tuning Strategy ──
    UNFREEZE_LAST_N_LAYERS = 6
    BACKBONE_LR = 2e-5              # FIX: 1e-5 → 2e-5 (was too conservative)
    USE_LAYERWISE_LR_DECAY = True
    LAYERWISE_LR_DECAY = 0.85       # FIX: 0.9 → 0.85 (gentler decay between layers)

    # ── Cross-Modal Attention ──
    CROSS_ATTN_D_MODEL = 256        # FIX: 128 → 256 (less bottleneck from 768-dim BERT)
    CROSS_ATTN_NHEAD = 8            # FIX: 4 → 8 (more attention patterns with larger d_model)
    CROSS_ATTN_LAYERS = 1
    CROSS_ATTN_DROPOUT = 0.1
    CROSS_ATTN_LR = 2e-4
    SEQ_PROJ_LR = 2e-4

    # ── DNAshape Branch ──
    SHAPE_CHANNELS = 5
    SHAPE_SEQ_LEN = 101
    SHAPE_CONV_CHANNELS = [32, 64, 128]
    SHAPE_LR = 3e-4
    USE_SHAPE_POS_EMBED = True

    # ── Pooling ──
    USE_ATTENTION_POOLING = True
    ATTN_POOL_LR = 2e-4

    # ── Classifier Head ──
    HEAD_TYPE = "mlp"
    HIDDEN_DIM = 256
    FUSION_DROPOUT = 0.3             # FIX: 0.4 → 0.3 (too much dropout was killing learning)
    HEAD_LR = 2e-4                   # FIX: 5e-5 → 2e-4 (CRITICAL: head couldn't learn at all)

    # ── Loss ──
    LOSS_TYPE = "weighted_ce"        # FIX: "focal" → "weighted_ce" (remove triple-stacking)
    FOCAL_GAMMA = 1.0                # only used if LOSS_TYPE == "focal"
    LABEL_SMOOTHING = 0.0            # FIX: 0.1 → 0.0 (remove on balanced dataset)
    NUM_CLASSES = 4
    WEIGHT_DECAY = 0.05              # FIX: 0.1 → 0.05 (too aggressive decay)

    # ── EMA ──
    USE_EMA = True
    EMA_DECAY = 0.999

    # ── Training ──
    BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = 4
    EPOCHS = 30                      # FIX: 25 → 30 (give model more time to learn)
    PATIENCE = 12                    # FIX: 10 → 12
    WARMUP_RATIO = 0.15              # FIX: 0.1 → 0.15 (longer warmup)
    MAX_GRAD_NORM = 1.0

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

for d in [cfg.OUTPUT_DIR, cfg.FIG_DIR, cfg.MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

random.seed(cfg.RANDOM_SEED)
np.random.seed(cfg.RANDOM_SEED)
torch.manual_seed(cfg.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {DEVICE}")
print(f"Architecture: Unified Cross-Modal Attention v2 (Script 17 — Bug Fixes)")
print(f"  FIXES: HEAD_LR={cfg.HEAD_LR}, d_model={cfg.CROSS_ATTN_D_MODEL}, "
      f"loss={cfg.LOSS_TYPE}, label_smooth={cfg.LABEL_SMOOTHING}")
print(f"  CORE: unfreeze={cfg.UNFREEZE_LAST_N_LAYERS}, "
      f"cross_layers={cfg.CROSS_ATTN_LAYERS}, head={cfg.HEAD_TYPE.upper()}")
print(f"  BACKBONE_LR={cfg.BACKBONE_LR}, decay={cfg.LAYERWISE_LR_DECAY}, "
      f"dropout={cfg.FUSION_DROPOUT}, ema={cfg.USE_EMA}")

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

    all_sequences, all_headers, all_labels, all_groups = [], [], [], []
    group_id = 0

    for cls_idx, (cls_name, fpath) in enumerate(fasta_files.items()):
        seqs, hdrs = load_fasta(fpath)
        print(f"  {cls_name}: {len(seqs)} sequences ({os.path.basename(fpath)})")
        all_sequences.extend(seqs)
        all_headers.extend(hdrs)
        all_labels.extend([cls_idx] * len(seqs))
        # Positives are stored as adjacent (fwd, revcomp) pairs → same group
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
    """Group-aware train/test split (no revcomp leakage)."""
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

    print(f"  Train: {len(seq_train)} sequences")
    print(f"  Test:  {len(seq_test)} sequences")
    print(f"  Train class dist: {np.bincount(y_train)}")
    print(f"  Test  class dist: {np.bincount(y_test)}")
    return seq_train, seq_test, y_train, y_test, shape_train, shape_test, headers_train, headers_test


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
        p1_val, p99_val = np.percentile(valid_vals, 1), np.percentile(valid_vals, 99)
        scale = max(p99_val - p1_val, 1e-9)
        shape_train_norm[:, ch, :] = (shape_train_norm[:, ch, :] - median_val) / scale
        shape_test_norm[:, ch, :] = (shape_test_norm[:, ch, :] - median_val) / scale
        print(f"  {channel_names[ch]:>5s}: median={median_val:>8.4f}, "
              f"P1={p1_val:>8.4f}, P99={p99_val:>8.4f}, scale={scale:>8.4f}")

    nan_train = np.isnan(shape_train_norm).sum()
    nan_test = np.isnan(shape_test_norm).sum()
    shape_train_norm = np.nan_to_num(shape_train_norm, nan=0.0)
    shape_test_norm = np.nan_to_num(shape_test_norm, nan=0.0)
    print(f"\n  NaN filled with 0: train={nan_train}, test={nan_test}")
    print(f"  Train shape range: [{shape_train_norm.min():.4f}, {shape_train_norm.max():.4f}]")
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
    print(f"  Patched {patched} flash-attention refs -> pure PyTorch."
          if patched else "  No flash-attn refs to patch.")

# ═══════════════════════════════════════════════════════════════════════
# CELL 6: Load DNABERT-2 Backbone (with Selective Unfreezing)
# ═══════════════════════════════════════════════════════════════════════

def load_dnabert2(model_name, device, unfreeze_last_n=6):
    """Load DNABERT-2 with selective layer unfreezing."""
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
        model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True, low_cpu_mem_usage=False)
        for name, param in model.named_parameters():
            if param.device == torch.device("meta"):
                raise RuntimeError(f"Parameter {name} on meta device")
        for name, buf in model.named_buffers():
            if buf.device == torch.device("meta"):
                raise RuntimeError(f"Buffer {name} on meta device")
        print("  Strategy 1 (direct load) OK")
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
            print("  Strategy 2 OK")
        except Exception as e:
            print(f"  Strategy 2 failed: {e}")
            model = None

    # Strategy 3: Monkey-patch torch.empty meta->cpu
    if model is None:
        try:
            print("  Trying Strategy 3 (monkey-patch ALiBi)...")
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
            print("  Strategy 3 OK")
        except Exception as e:
            raise RuntimeError(f"All loading strategies failed: {e}") from e

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
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  DNABERT-2 loaded on {device}")
    print(f"  Total encoder layers: {total_layers}")
    print(f"  Frozen: layers 0-{unfreeze_from-1} | Unfrozen: layers {unfreeze_from}-{total_layers-1} ({unfreeze_last_n})")
    print(f"  Total: {total_params:,} | Trainable: {trainable:,} ({100*trainable/total_params:.1f}%)")
    return model, tokenizer


dnabert_model, tokenizer = load_dnabert2(cfg.DNABERT_MODEL, DEVICE, cfg.UNFREEZE_LAST_N_LAYERS)

# ── Auto-detect max token length from data ──
def compute_max_token_length(sequences, tok, sample_size=2000, percentile=99, floor=32, cap=96):
    """Tokenize a sample to find the p99 token length, then clamp to [floor, cap]."""
    if len(sequences) > sample_size:
        idx = random.sample(range(len(sequences)), sample_size)
        sample = [sequences[i] for i in idx]
    else:
        sample = sequences
    lengths = [len(tok(s, add_special_tokens=True)["input_ids"]) for s in sample]
    p_val = int(np.percentile(lengths, percentile))
    chosen = int(max(floor, min(cap, p_val + 2)))  # +2 slack for special tokens
    print("\n" + "=" * 60)
    print("AUTO MAX TOKEN LENGTH")
    print("=" * 60)
    print(f"  Token length (sample n={len(sample)}): "
          f"min={min(lengths)}, mean={np.mean(lengths):.1f}, "
          f"p{percentile}={p_val}, max={max(lengths)}")
    print(f"  Chosen MAX_TOKEN_LENGTH = {chosen}  (was 512 in Scripts 14/15 → "
          f"~{512/chosen:.0f}x fewer padding tokens)")
    return chosen

if cfg.AUTO_MAX_LENGTH:
    MAX_LENGTH = compute_max_token_length(
        seq_train, tokenizer,
        floor=cfg.MAX_LENGTH_FLOOR, cap=cfg.MAX_LENGTH_CAP,
    )
else:
    MAX_LENGTH = cfg.MAX_TOKEN_LENGTH
    print(f"\n  Using fixed MAX_TOKEN_LENGTH = {MAX_LENGTH}")

# ═══════════════════════════════════════════════════════════════════════
# CELL 7: Architecture
# ═══════════════════════════════════════════════════════════════════════

class SpatialShapeCNN(nn.Module):
    """
    Conv1D on DNAshape features, PRESERVING spatial dimension (no GlobalAvgPool).
    Optionally adds a learnable positional embedding.

    Input:  [B, 5, 101]  → Output: [B, 25, d_model]
    """
    def __init__(self, in_channels=5, conv_channels=None, d_model=256,
                 use_pos_embed=True, max_positions=26):
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

        self.proj = nn.Linear(conv_channels[-1], d_model)

        self.use_pos_embed = use_pos_embed
        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.randn(1, max_positions, d_model) * 0.02)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.transpose(1, 2)    # [B, 25, conv_channels[-1]]
        x = self.proj(x)         # [B, 25, d_model]
        if self.use_pos_embed:
            x = x + self.pos_embed[:, :x.size(1), :]
        return x


class FeedForward(nn.Module):
    """Post-attention FFN with residual."""
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
    """
    def __init__(self, d_model=256, nhead=8, dropout=0.1):
        super().__init__()
        self.cross_attn_seq2shape = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm_seq = nn.LayerNorm(d_model)
        self.ffn_seq = FeedForward(d_model, expansion=4, dropout=dropout)

        self.cross_attn_shape2seq = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm_shape = nn.LayerNorm(d_model)
        self.ffn_shape = FeedForward(d_model, expansion=4, dropout=dropout)

    def forward(self, seq_features, shape_features, seq_key_padding_mask=None):
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


class AttentionPooling(nn.Module):
    """Learnable query token that attends over a sequence to produce one vector."""
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask)
        return self.norm(out.squeeze(1))


class UnifiedCrossModalClassifier(nn.Module):
    """
    Script 17: Unified Cross-Modal Attention v2 (Bug Fixes).

    Key changes from Script 16:
    - d_model: 128 → 256 (less information bottleneck)
    - No dropout in seq_projection (was adding noise to bottleneck)
    - Cleaner architecture with fewer regularization conflicts
    """
    def __init__(self, backbone, embedding_dim=768,
                 shape_in_channels=5, shape_conv_channels=None,
                 d_model=256, nhead=8, num_cross_layers=1, cross_dropout=0.1,
                 use_shape_pos_embed=True, use_attention_pooling=True,
                 hidden_dim=256, num_classes=4, fusion_dropout=0.3):
        super().__init__()
        self.backbone = backbone
        self.d_model = d_model
        self.use_attention_pooling = use_attention_pooling

        # Branch 1: DNABERT-2 token projection (NO dropout here — FIX)
        self.seq_projection = nn.Sequential(
            nn.Linear(embedding_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Branch 2: SpatialShapeCNN
        if shape_conv_channels is None:
            shape_conv_channels = [32, 64, 128]
        self.shape_cnn = SpatialShapeCNN(
            in_channels=shape_in_channels, conv_channels=shape_conv_channels,
            d_model=d_model, use_pos_embed=use_shape_pos_embed,
        )

        # Cross-Modal Attention stack
        self.cross_attention_layers = nn.ModuleList([
            CrossModalAttentionLayer(d_model=d_model, nhead=nhead, dropout=cross_dropout)
            for _ in range(num_cross_layers)
        ])

        # Sequence pooling
        if use_attention_pooling:
            self.seq_attn_pool = AttentionPooling(d_model, nhead=4, dropout=cross_dropout)
        else:
            self.seq_attn_pool = None

        # Classifier head — clean MLP
        fusion_dim = d_model * 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=fusion_dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids, attention_mask, shape_features):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]                          # [B, T, 768]
        seq_features = self.seq_projection(hidden_states)   # [B, T, d_model]

        shape_feats = self.shape_cnn(shape_features)        # [B, 25, d_model]

        seq_key_padding_mask = (attention_mask == 0)
        for cross_layer in self.cross_attention_layers:
            seq_features, shape_feats = cross_layer(
                seq_features, shape_feats, seq_key_padding_mask=seq_key_padding_mask
            )

        # Late pooling (AFTER cross-attention)
        if self.use_attention_pooling:
            seq_pooled = self.seq_attn_pool(seq_features, key_padding_mask=seq_key_padding_mask)
        else:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            seq_pooled = (seq_features * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        shape_pooled = shape_feats.mean(dim=1)

        fused = torch.cat([seq_pooled, shape_pooled], dim=1)
        return self.classifier(fused)


# ═══════════════════════════════════════════════════════════════════════
# CELL 8: Dataset & DataLoaders
# ═══════════════════════════════════════════════════════════════════════

class DualBranchDataset(Dataset):
    """Dataset providing tokenized DNA sequences and DNAshape features."""
    def __init__(self, sequences, labels, shape_features, tokenizer, max_length=48):
        self.sequences = sequences
        self.labels = labels
        self.shape_features = shape_features
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.sequences[idx], padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        shape = torch.tensor(self.shape_features[idx], dtype=torch.float32)
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "shape_features": shape,
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


train_dataset = DualBranchDataset(seq_train, y_train, shape_train_norm, tokenizer, MAX_LENGTH)
test_dataset = DualBranchDataset(seq_test, y_test, shape_test_norm, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE * 2, shuffle=False,
                         num_workers=2, pin_memory=True)

print(f"\nDataLoaders ready: {len(train_loader)} train batches, {len(test_loader)} test batches")
print(f"Sequence max_length = {MAX_LENGTH} | effective batch = {cfg.BATCH_SIZE * cfg.GRAD_ACCUM_STEPS}")

# ═══════════════════════════════════════════════════════════════════════
# CELL 9: Build Model, EMA, Optimizer (layer-wise LR), Loss
# ═══════════════════════════════════════════════════════════════════════

model = UnifiedCrossModalClassifier(
    backbone=dnabert_model,
    embedding_dim=cfg.EMBEDDING_DIM,
    shape_in_channels=cfg.SHAPE_CHANNELS,
    shape_conv_channels=cfg.SHAPE_CONV_CHANNELS,
    d_model=cfg.CROSS_ATTN_D_MODEL,
    nhead=cfg.CROSS_ATTN_NHEAD,
    num_cross_layers=cfg.CROSS_ATTN_LAYERS,
    cross_dropout=cfg.CROSS_ATTN_DROPOUT,
    use_shape_pos_embed=cfg.USE_SHAPE_POS_EMBED,
    use_attention_pooling=cfg.USE_ATTENTION_POOLING,
    hidden_dim=cfg.HIDDEN_DIM,
    num_classes=cfg.NUM_CLASSES,
    fusion_dropout=cfg.FUSION_DROPOUT,
)
model = model.to(DEVICE)


# ── EMA wrapper ──
class ModelEMA:
    """Exponential Moving Average of model parameters for smoother eval weights."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)  # buffers like num_batches_tracked

    def state_dict(self):
        return self.shadow


ema = ModelEMA(model, decay=cfg.EMA_DECAY) if cfg.USE_EMA else None


# ── Optimizer with layer-wise LR decay on backbone ──
param_groups = []

total_bert_layers = len(model.backbone.encoder.layer)
unfreeze_from = total_bert_layers - cfg.UNFREEZE_LAST_N_LAYERS
if cfg.USE_LAYERWISE_LR_DECAY:
    for i, layer in enumerate(model.backbone.encoder.layer):
        layer_params = [p for p in layer.parameters() if p.requires_grad]
        if not layer_params:
            continue
        dist_from_top = (total_bert_layers - 1) - i
        layer_lr = cfg.BACKBONE_LR * (cfg.LAYERWISE_LR_DECAY ** dist_from_top)
        param_groups.append({"params": layer_params, "lr": layer_lr, "weight_decay": cfg.WEIGHT_DECAY})
else:
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    param_groups.append({"params": backbone_params, "lr": cfg.BACKBONE_LR, "weight_decay": cfg.WEIGHT_DECAY})

# Heads — all at the SAME lr now (FIX: head was 4x lower before)
param_groups.append({"params": list(model.seq_projection.parameters()),       "lr": cfg.SEQ_PROJ_LR,   "weight_decay": cfg.WEIGHT_DECAY})
param_groups.append({"params": list(model.shape_cnn.parameters()),            "lr": cfg.SHAPE_LR,      "weight_decay": cfg.WEIGHT_DECAY})
param_groups.append({"params": list(model.cross_attention_layers.parameters()), "lr": cfg.CROSS_ATTN_LR, "weight_decay": cfg.WEIGHT_DECAY})
if cfg.USE_ATTENTION_POOLING:
    param_groups.append({"params": list(model.seq_attn_pool.parameters()),    "lr": cfg.ATTN_POOL_LR,  "weight_decay": cfg.WEIGHT_DECAY})
param_groups.append({"params": list(model.classifier.parameters()),           "lr": cfg.HEAD_LR,       "weight_decay": cfg.WEIGHT_DECAY})

optimizer = optim.AdamW(param_groups)

total_steps = (len(train_loader) // cfg.GRAD_ACCUM_STEPS) * cfg.EPOCHS
warmup_steps = int(total_steps * cfg.WARMUP_RATIO)

def lr_lambda(current_step):
    """Linear warmup then cosine decay."""
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Class weights (FIX: only used in weighted CE, not stacked with focal+smoothing)
class_counts = np.bincount(y_train)
class_weights = len(y_train) / (len(class_counts) * class_counts.astype(np.float32))
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
print(f"  Class weights: " + ", ".join([f"{cfg.CLASS_NAMES[i]}={class_weights[i]:.4f}" for i in range(len(class_counts))]))

# Loss — CLEAN, single objective (FIX: no more triple-stacking)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=cfg.LABEL_SMOOTHING)

scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

# Model summary
trainable_total = sum(p.numel() for g in param_groups for p in g["params"])
print(f"\n{'='*60}")
print("MODEL SUMMARY -- Unified Cross-Modal Attention v2 (Script 17)")
print(f"{'='*60}")
print(f"  Param groups:        {len(param_groups)} (layer-wise LR decay={cfg.USE_LAYERWISE_LR_DECAY})")
print(f"  Trainable params:    {trainable_total:,}")
print(f"  d_model:             {cfg.CROSS_ATTN_D_MODEL} (was 128 in S16)")
print(f"  Head type:           {cfg.HEAD_TYPE.upper()} (hidden={cfg.HIDDEN_DIM})")
print(f"  HEAD_LR:             {cfg.HEAD_LR} (was 5e-5 in S16)")
print(f"  BACKBONE_LR:         {cfg.BACKBONE_LR} (was 1e-5 in S16)")
print(f"  Cross-attn layers:   {cfg.CROSS_ATTN_LAYERS}")
print(f"  Loss:                {cfg.LOSS_TYPE} (no label smoothing)")
print(f"  EMA:                 {cfg.USE_EMA} (decay={cfg.EMA_DECAY})")
print(f"  Total / warmup steps:{total_steps} / {warmup_steps}")

# ═══════════════════════════════════════════════════════════════════════
# CELL 10: Training Loop with Early Stopping + Gradient Monitoring
# ═══════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler, device,
                    grad_accum_steps, ema=None, max_grad_norm=1.0, epoch_num=0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()

    # Gradient monitoring (check if gradients are flowing)
    grad_norms = []

    pbar = tqdm(loader, desc="  Training", leave=False)
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        shape_features = batch["shape_features"].to(device)
        labels = batch["labels"].to(device)

        with autocast(enabled=(device.type == "cuda")):
            logits = model(input_ids, attention_mask, shape_features)
            loss = criterion(logits, labels)
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            grad_norms.append(total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)

        running_loss += loss.item() * grad_accum_steps * labels.size(0)
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({"loss": f"{running_loss/total:.4f}", "acc": f"{correct/total:.4f}"})

    # Print gradient diagnostics for first few epochs
    if epoch_num < 5 and grad_norms:
        avg_norm = np.mean(grad_norms)
        max_norm = np.max(grad_norms)
        print(f"  [Grad Monitor] avg_norm={avg_norm:.4f}, max_norm={max_norm:.4f}, "
              f"steps={len(grad_norms)}")
        if avg_norm < 1e-6:
            print("  ⚠️  WARNING: Gradients near zero! Model may not be learning.")

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
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
                criterion, scaler, cfg, device, ema=None):
    print("\n" + "=" * 60)
    print("TRAINING -- Unified Cross-Modal Attention v2 (Script 17)")
    print("=" * 60)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(cfg.EPOCHS):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler, device,
            cfg.GRAD_ACCUM_STEPS, ema=ema, max_grad_norm=cfg.MAX_GRAD_NORM,
            epoch_num=epoch,
        )

        # Evaluate with EMA weights if enabled
        if ema is not None:
            backup = copy.deepcopy(model.state_dict())
            model.load_state_dict(ema.state_dict(), strict=True)
            val_loss, val_acc = evaluate(model, test_loader, criterion, device)
            model.load_state_dict(backup, strict=True)
        else:
            val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        elapsed = time.time() - t0
        gap = train_acc - val_acc

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1:02d}/{cfg.EPOCHS} | "
            f"Train: {train_loss:.4f}/{train_acc:.4f} | "
            f"Val: {val_loss:.4f}/{val_acc:.4f} | "
            f"Gap: {gap:.2%} | LR: {optimizer.param_groups[0]['lr']:.1e} | {elapsed:.0f}s"
        )

        # Early warning if model is stuck at random
        if epoch >= 3 and max(history["val_acc"]) < 0.30:
            print("  ⚠️  WARNING: Val accuracy still near random chance after 3 epochs!")

        if val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            state_to_save = ema.state_dict() if ema is not None else model.state_dict()
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": state_to_save,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, os.path.join(cfg.MODEL_DIR, "best_unified_crossmodal_v2.pt"))
            print(f"  -> Saved best (val_acc={val_acc:.4f}, gap={gap:.2%})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1} (patience={cfg.PATIENCE})")
                break

    print(f"\nTraining complete. Best val_acc: {best_val_acc:.4f}")
    return history


history = train_model(model, train_loader, test_loader, optimizer, scheduler,
                      criterion, scaler, cfg, DEVICE, ema=ema)

# ═══════════════════════════════════════════════════════════════════════
# CELL 11: Load Best Model & Full Evaluation
# ═══════════════════════════════════════════════════════════════════════

best_path = os.path.join(cfg.MODEL_DIR, "best_unified_crossmodal_v2.pt")
checkpoint = torch.load(best_path, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()
print(f"Loaded best model from epoch {checkpoint['epoch']} "
      f"(val_loss={checkpoint['val_loss']:.4f}, val_acc={checkpoint['val_acc']:.4f})")


@torch.no_grad()
def full_evaluation(model, test_loader, class_names, device):
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

    report_path = os.path.join(cfg.OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("CLASSIFICATION REPORT -- Unified Cross-Modal Attention v2 (Script 17)\n")
        f.write("=" * 60 + "\n")
        f.write(report_str)
    print(f"  -> Report saved: {report_path}")

    acc = accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average="macro")
    print(f"Overall Accuracy:  {acc:.4f}")
    print(f"Macro F1-Score:    {f1_macro:.4f}")
    return all_preds, all_targets, all_probs


all_preds, all_targets, all_probs = full_evaluation(model, test_loader, cfg.CLASS_NAMES, DEVICE)

# ── Export BED files for IGV Analysis ──
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
    beds = {"True_SP1": [], "True_SP4": [], "Confused_SP4_as_SP1": []}
    for idx, (pred, target) in enumerate(zip(predictions, targets)):
        fields = parse_header_to_bed(headers_test[idx])
        if not fields:
            continue
        line = f"{fields[0]}\t{fields[1]}\t{fields[2]}\n"
        if target == 0 and pred == 0:
            beds["True_SP1"].append(line)
        elif target == 2 and pred == 2:
            beds["True_SP4"].append(line)
        elif target == 2 and pred == 0:
            beds["Confused_SP4_as_SP1"].append(line)
    for name, lines in beds.items():
        with open(os.path.join(output_dir, f"{name}.bed"), "w") as f:
            f.writelines(lines)
        print(f"    {name}.bed: {len(lines)} lines")


print("\n  BED files for IGV:")
export_igv_bed_files(headers_test, all_preds, all_targets, cfg.OUTPUT_DIR)

# ═══════════════════════════════════════════════════════════════════════
# CELL 12: Generate All Performance Figures
# ═══════════════════════════════════════════════════════════════════════

TITLE_PREFIX = "Unified Cross-Modal Attention v2 (DNABERT-2 + DNAshape)"

def plot_training_curves(history, save_dir):
    epochs = len(history["train_loss"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(range(1, epochs+1), history["train_loss"], label="Train Loss", color="#2196F3", linewidth=2)
    ax1.plot(range(1, epochs+1), history["val_loss"], label="Val Loss", color="#FF5722", linewidth=2)
    ax1.set_title("Loss Convergence", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.plot(range(1, epochs+1), history["train_acc"], label="Train Acc", color="#4CAF50", linewidth=2)
    ax2.plot(range(1, epochs+1), history["val_acc"], label="Val Acc", color="#E91E63", linewidth=2)
    ax2.axhline(y=0.25, color="gray", linestyle="--", alpha=0.5, label="Random Guess")
    ax2.set_title("Accuracy Performance", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend(fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle(f"{TITLE_PREFIX} -- Training Progress", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "unified_v2_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
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
        ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted Label", fontsize=11); ax.set_ylabel("True Label", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        thresh = data.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if data[i, j] > thresh else "black", fontsize=12, fontweight="bold")
    plt.suptitle(TITLE_PREFIX, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "unified_v2_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
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
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12); plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curves -- {TITLE_PREFIX}", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10); plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "unified_v2_roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
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
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=12); plt.ylabel("Precision", fontsize=12)
    plt.title(f"Precision-Recall Curves -- {TITLE_PREFIX}", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10); plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "unified_v2_pr_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  -> PR curves: {path}")


def plot_per_class_metrics_bar(all_targets, all_preds, class_names, save_dir):
    prec, rec, f1, support = precision_recall_fscore_support(all_targets, all_preds, average=None)
    x = np.arange(len(class_names)); width = 0.25
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
    ax.set_ylim(0, 1.15); ax.legend(fontsize=11); ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(save_dir, "unified_v2_per_class_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  -> Per-class metrics: {path}")


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
# CELL 13: Summary
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PIPELINE SUMMARY -- Unified Cross-Modal Attention v2 (Script 17)")
print("=" * 60)
print(f"  DNABERT-2:         Last {cfg.UNFREEZE_LAST_N_LAYERS} layers fine-tuned "
      f"(layer-wise LR decay={cfg.USE_LAYERWISE_LR_DECAY})")
print(f"  Max token length:  {MAX_LENGTH} (was 512 in Scripts 14/15)")
print(f"  d_model:           {cfg.CROSS_ATTN_D_MODEL} (was 128 in Script 16)")
print(f"  Cross-Attention:   {cfg.CROSS_ATTN_LAYERS} layer(s), {cfg.CROSS_ATTN_NHEAD} heads")
print(f"  Shape pos-embed:   {cfg.USE_SHAPE_POS_EMBED} | Attn-pooling: {cfg.USE_ATTENTION_POOLING}")
print(f"  Head:              {cfg.HEAD_TYPE.upper()} (hidden={cfg.HIDDEN_DIM}) | "
      f"HEAD_LR: {cfg.HEAD_LR} (was 5e-5)")
print(f"  Loss:              {cfg.LOSS_TYPE} (no label_smooth, no focal)")
print(f"  EMA:               {cfg.USE_EMA} (decay={cfg.EMA_DECAY})")
print(f"  Best val acc:      {max(history['val_acc']):.4f}")
print(f"  Model saved to:    {cfg.MODEL_DIR}")
print()
print("  +-----------------------------------------------------------+")
print("  |  Script 14 (Cross-Modal):     61.33%  (overfit, gap 31%)  |")
print("  |  Script 15 (KAN, 11 changes): 53.43%  (over-regularized)  |")
print("  |  Script 16 (Unified):         25%     (BROKEN: no learn)  |")
print(f"  |  Script 17 (v2 Bug Fix):      {max(history['val_acc']):.2%}                      |")
print("  +-----------------------------------------------------------+")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════════════
# CELL 14: Zip Outputs for Easy Kaggle Download
# ═══════════════════════════════════════════════════════════════════════

import shutil
from IPython.display import FileLink

zip_filename = "outputs_unified_crossmodal_v2"
shutil.make_archive(zip_filename, 'zip', cfg.OUTPUT_DIR)
print(f"\nAll outputs zipped into: {zip_filename}.zip")

try:
    from IPython.display import display
    display(FileLink(f"{zip_filename}.zip"))
except ImportError:
    print(f"FileLink not available. Please download {zip_filename}.zip from the Kaggle output panel.")
