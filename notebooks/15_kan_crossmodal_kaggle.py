#!/usr/bin/env python3
"""
Script 15: KAN-Regularized Cross-Modal Attention Dual-Branch
Designed for: Kaggle GPU (T4/P100) or Google Colab
Task: 4-class SP1/SP2/SP4/Negative TF-binding classification

BOTTLENECK FIXES vs Script 14 (Val Acc 61.33%, Train Acc 92% → 34% gap):

  BOTTLENECK 1 — Overfitting (Train 92% vs Val 58%):
    Fix A. Freeze more BERT layers: 6→3 unfrozen (halve backbone trainable params)
    Fix B. Aggressive dropout: fusion 0.3→0.5, cross-attn 0.1→0.15
    Fix C. R-Drop regularization (KL-divergence on dual forward passes)
    Fix D. Higher weight decay: 0.1→0.15
    Fix E. Tighter gradient clipping: max_norm 1.0→0.5
    Fix F. Lower backbone LR: 1e-5→5e-6

  BOTTLENECK 2 — MLP classifier too rigid (181 SP4→SP1 misclassifications):
    Fix G. ChebyKAN classifier (Chebyshev polynomial KAN replaces nn.Linear)
           B-spline-like curved decision boundaries instead of hyperplanes

  BOTTLENECK 3 (additional) — Shallow cross-modal communication:
    Fix H. 2 cross-attention layers (was 1) with DropPath stochastic depth
    Fix I. Learnable positional embedding for shape spatial positions
    Fix J. Attention Pooling (learnable query token) for sequence aggregation

  BOTTLENECK 4 (additional) — Loss function treats all errors equally:
    Fix K. Focal Loss (gamma=2): down-weight easy Negative class,
           focus gradient on hard SP1/SP4 confusion cases

ARCHITECTURE (Script 15):
  Branch 1 — DNABERT-2 (3 layers unfrozen):
    Full tokens [B, T, 768] → Linear(768→128) → [B, T, 128]

  Branch 2 — SpatialShapeCNN + Positional Embedding:
    [B, 5, 101] → Conv1D×3 → [B, 25, 128] + pos_embed

  Cross-Modal Attention (2 layers, DropPath):
    Layer 1: seq ↔ shape bidirectional (4 heads, d=128)
    Layer 2: seq ↔ shape bidirectional (4 heads, d=128)

  Pooling:
    Seq: AttentionPooling (learnable [QUERY] token)
    Shape: Mean pooling

  KAN Classifier:
    LayerNorm(256) → ChebyKAN(256→64) → LayerNorm → Dropout(0.5) → ChebyKAN(64→4)

  Loss: Focal Loss + R-Drop KL-divergence
"""

# ═══════════════════════════════════════════════════════════════════════
# CELL 0: Install Dependencies
# ═══════════════════════════════════════════════════════════════════════

import subprocess
import sys

def install_packages():
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
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
          if hasattr(torch.cuda.get_device_properties(0), 'total_mem')
          else f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.backends.cudnn.benchmark = True

# ═══════════════════════════════════════════════════════════════════════
# CELL 2: Configuration
# ═══════════════════════════════════════════════════════════════════════

def find_file(filename, fallback_dir="data/processed"):
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
    resolved_path = find_file(target_file, fallback)
    if resolved_path:
        return os.path.dirname(resolved_path)
    return fallback


class Config:
    """Script 15: KAN-Regularized Cross-Modal Attention Dual-Branch."""

    # ── Paths ──
    FASTA_DIR = auto_detect_dir("sp1_positive_final.fasta", "data/processed")
    SHAPE_DIR = auto_detect_dir("dnashape_sp1.npy", "data/processed")
    OUTPUT_DIR = "outputs_kan_crossmodal"
    FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

    # ── DNABERT-2 ──
    DNABERT_MODEL = "zhihan1996/DNABERT-2-117M"
    EMBEDDING_DIM = 768
    MAX_TOKEN_LENGTH = 512

    # ── Fine-Tuning (FIX A: freeze more layers) ──
    UNFREEZE_LAST_N_LAYERS = 3     # was 6 → halves backbone trainable params
    BACKBONE_LR = 5e-6             # was 1e-5 → more conservative (FIX F)

    # ── Cross-Modal Attention (FIX H: deeper) ──
    CROSS_ATTN_D_MODEL = 128
    CROSS_ATTN_NHEAD = 4
    CROSS_ATTN_LAYERS = 2          # was 1 → deeper cross-modal communication
    CROSS_ATTN_DROPOUT = 0.15      # was 0.1 → more regularization (FIX B)
    DROP_PATH_RATE = 0.1           # NEW: stochastic depth
    CROSS_ATTN_LR = 1.5e-4
    SEQ_PROJ_LR = 1.5e-4

    # ── DNAshape Branch ──
    SHAPE_CHANNELS = 5
    SHAPE_SEQ_LEN = 101
    SHAPE_CONV_CHANNELS = [32, 64, 128]
    SHAPE_LR = 2e-4

    # ── KAN Classifier (FIX G: replaces MLP) ──
    KAN_HIDDEN_DIM = 64            # smaller than MLP's 256 → fewer params
    KAN_DEGREE = 4                 # Chebyshev polynomial degree
    FUSION_DROPOUT = 0.5           # was 0.3 → aggressive regularization (FIX B)
    HEAD_LR = 1e-4                 # KAN needs slightly higher LR
    ATTN_POOL_LR = 1.5e-4         # for attention pooling module

    # ── Loss Function (FIX K) ──
    FOCAL_GAMMA = 2.0              # down-weight easy examples
    LABEL_SMOOTHING = 0.1
    NUM_CLASSES = 4
    WEIGHT_DECAY = 0.15            # was 0.1 → stronger L2 (FIX D)

    # ── R-Drop Regularization (FIX C) ──
    RDROP_ALPHA = 1.0              # KL divergence weight

    # ── Training ──
    BATCH_SIZE = 12                # reduced from 16 for R-Drop VRAM
    GRAD_ACCUM_STEPS = 6           # effective batch = 72 (was 64)
    EPOCHS = 30                    # more epochs since regularization slows learning
    PATIENCE = 12
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 0.5            # was 1.0 → tighter clipping (FIX E)

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
print(f"Architecture: KAN-Regularized Cross-Modal Attention Dual-Branch")
print(f"  * Cross-Attention: {cfg.CROSS_ATTN_LAYERS} layers, {cfg.CROSS_ATTN_NHEAD} heads, d={cfg.CROSS_ATTN_D_MODEL}")
print(f"  * KAN Classifier: ChebyKAN degree={cfg.KAN_DEGREE}, hidden={cfg.KAN_HIDDEN_DIM}")
print(f"  * R-Drop alpha={cfg.RDROP_ALPHA}, Focal gamma={cfg.FOCAL_GAMMA}")
print(f"  * Regularization: unfreeze={cfg.UNFREEZE_LAST_N_LAYERS}, dropout={cfg.FUSION_DROPOUT}, wd={cfg.WEIGHT_DECAY}")
print(f"  * Effective batch: {cfg.BATCH_SIZE} x {cfg.GRAD_ACCUM_STEPS} = {cfg.BATCH_SIZE * cfg.GRAD_ACCUM_STEPS}")

# ═══════════════════════════════════════════════════════════════════════
# CELL 3: Data Loading (Sequences + DNAshape Features)
# ═══════════════════════════════════════════════════════════════════════

def load_fasta(filepath):
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
    all_shapes = []
    neg_shape_path = None
    for cand in ["dnashape_negative_genomic.npy", "dnashape_negative_cpg.npy", "dnashape_negative.npy"]:
        path = find_file(cand, data_dir)
        if path:
            neg_shape_path = path
            break
    if not neg_shape_path:
        raise FileNotFoundError("Could not find any negative DNAshape file.")
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
    print("=" * 60)
    print("LOADING DATASETS (Sequences + DNAshape Features)")
    print("=" * 60)

    neg_fasta_path = None
    for cand in ["negative_genomic_matched.fasta", "negative_promoter_cpg.fasta", "negative_final.fasta"]:
        path = find_file(cand, fasta_dir)
        if path:
            neg_fasta_path = path
            break
    if not neg_fasta_path:
        raise FileNotFoundError("Could not find any negative FASTA file.")
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
    assert len(all_sequences) == all_shapes.shape[0]

    print(f"\n  Total: {len(all_sequences)} sequences, {group_id} groups")
    print(f"  Shape features: {all_shapes.shape}")
    print(f"  Class distribution: {np.bincount(all_labels)}")
    return all_sequences, all_labels, all_groups, all_shapes, all_headers


def split_data(sequences, labels, groups, shapes, headers, test_size=0.2, seed=42):
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
        print(f"  {channel_names[ch]:>5s}: median={median_val:>8.4f}, P1={p1_val:>8.4f}, P99={p99_val:>8.4f}, scale={scale:>8.4f}")

    nan_train = np.isnan(shape_train_norm).sum()
    nan_test = np.isnan(shape_test_norm).sum()
    shape_train_norm = np.nan_to_num(shape_train_norm, nan=0.0)
    shape_test_norm = np.nan_to_num(shape_test_norm, nan=0.0)
    print(f"\n  NaN filled: train={nan_train}, test={nan_test}")
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
    if bias is not None: attn = attn + bias
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
    if bias is not None: attn = attn + bias
    if causal:
        Sq, Sk = q.shape[2], k.shape[2]
        attn = attn.masked_fill(torch.triu(torch.ones(Sq, Sk, device=q.device, dtype=torch.bool), 1), float("-inf"))
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(attn, v).transpose(1, 2).contiguous()

def _pytorch_flash_attn_func(q, k, v, bias=None, causal=False, softmax_scale=None):
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    scale = softmax_scale if softmax_scale is not None else (q.shape[-1] ** -0.5)
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    if bias is not None: attn = attn + bias
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
        if mod is None: continue
        if "flash_attn_triton" in mod_name or "bert_layers" in mod_name:
            for attr_name, replacement in targets.items():
                if hasattr(mod, attr_name):
                    setattr(mod, attr_name, replacement)
                    patched += 1
    print(f"  Patched {patched} flash-attention refs." if patched else "  No flash-attn refs to patch.")

# ═══════════════════════════════════════════════════════════════════════
# CELL 6: Load DNABERT-2 Backbone (FIX A: only 3 layers unfrozen)
# ═══════════════════════════════════════════════════════════════════════

def load_dnabert2(model_name, device, unfreeze_last_n=3):
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
            if param.device == torch.device("meta"): raise RuntimeError(f"{name} on meta")
        for name, buf in model.named_buffers():
            if buf.device == torch.device("meta"): raise RuntimeError(f"{name} on meta")
        print("  Strategy 1 (direct load) OK")
    except Exception as e:
        print(f"  Strategy 1 failed: {e}")
        model = None

    # Strategy 2: Empty init + state_dict
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
                if ck in set(result.missing_keys): raise RuntimeError(f"Core weight '{ck}' missing!")
            model = model.to("cpu")
            print("  Strategy 2 OK")
        except Exception as e:
            print(f"  Strategy 2 failed: {e}")
            model = None

    # Strategy 3: Monkey-patch
    if model is None:
        try:
            print("  Trying Strategy 3 (monkey-patch ALiBi)...")
            _orig = torch.empty
            def _patched(*a, **kw):
                if kw.get("device") == torch.device("meta") or str(kw.get("device","")) == "meta": kw["device"] = "cpu"
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

    # Selective Freezing
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
    print(f"\n  DNABERT-2 on {device}")
    print(f"  Frozen: layers 0-{unfreeze_from-1} | Unfrozen: layers {unfreeze_from}-{total_layers-1} ({unfreeze_last_n})")
    print(f"  Total: {total_params:,} | Trainable: {trainable:,} ({100*trainable/total_params:.1f}%)")
    return model, tokenizer


dnabert_model, tokenizer = load_dnabert2(cfg.DNABERT_MODEL, DEVICE, cfg.UNFREEZE_LAST_N_LAYERS)

# ═══════════════════════════════════════════════════════════════════════
# CELL 7: KAN + Cross-Modal Architecture (ALL BOTTLENECK FIXES)
# ═══════════════════════════════════════════════════════════════════════

# ── FIX H supplement: Stochastic Depth ──
class DropPath(nn.Module):
    """Stochastic depth: randomly drops entire residual branches during training."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x.div(keep) * mask


# ── FIX G: Chebyshev KAN Layer ──
class ChebyKANLinear(nn.Module):
    """
    Kolmogorov-Arnold Network layer using Chebyshev polynomial basis.

    Instead of y = Wx + b (hyperplane), KAN learns:
      y = W_base * x + sum_d(coeff_d * T_d(tanh(x)))
    where T_d are Chebyshev polynomials of degree d.

    This provides CURVED decision boundaries that can slice through
    the entangled SP1/SP4 manifold, where linear MLP cannot.
    """
    def __init__(self, in_features, out_features, degree=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree

        # Chebyshev coefficients: learnable non-linear edge activations
        self.cheby_coeffs = nn.Parameter(
            torch.randn(out_features, in_features, degree + 1)
            * (1.0 / math.sqrt(in_features * (degree + 1)))
        )

        # Residual linear path (like skip connection for stability)
        self.base_linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        """
        Args:
            x: [batch, in_features]
        Returns:
            [batch, out_features]
        """
        # Residual linear: standard affine transform
        base_output = self.base_linear(x)

        # Normalize to [-1, 1] for Chebyshev domain
        x_norm = torch.tanh(x)

        # Compute Chebyshev polynomials via stable 3-term recurrence:
        #   T_0(x) = 1, T_1(x) = x, T_{n+1}(x) = 2x*T_n(x) - T_{n-1}(x)
        T = [torch.ones_like(x_norm)]
        if self.degree >= 1:
            T.append(x_norm)
        for n in range(2, self.degree + 1):
            T.append(2.0 * x_norm * T[-1] - T[-2])

        # Stack basis: [batch, in_features, degree+1]
        cheby_basis = torch.stack(T, dim=-1)

        # Weighted sum over all edges: output[b,o] = sum_{i,d} basis[b,i,d] * coeffs[o,i,d]
        kan_output = torch.einsum('bid,oid->bo', cheby_basis, self.cheby_coeffs)

        return base_output + kan_output


# ── FIX I: SpatialShapeCNN with positional embedding ──
class SpatialShapeCNN(nn.Module):
    """
    Conv1D on DNAshape features, preserving spatial dimension.
    Now includes learnable positional embedding (FIX I).

    Input:  [B, 5, 101]
    Output: [B, 25, d_model] + positional embedding
    """
    def __init__(self, in_channels=5, conv_channels=None, d_model=128, max_positions=26):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 128]

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels[0], kernel_size=7, padding=3),
            nn.BatchNorm1d(conv_channels[0]),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels[1]),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels[2]),
            nn.GELU(),
        )

        if conv_channels[-1] != d_model:
            self.proj = nn.Linear(conv_channels[-1], d_model)
        else:
            self.proj = nn.Identity()

        # FIX I: Learnable positional embedding for spatial positions
        self.pos_embed = nn.Parameter(torch.randn(1, max_positions, d_model) * 0.02)

    def forward(self, x):
        x = self.conv_block1(x)   # [B, 32, 50]
        x = self.conv_block2(x)   # [B, 64, 25]
        x = self.conv_block3(x)   # [B, 128, 25]
        x = x.transpose(1, 2)    # [B, 25, 128]
        x = self.proj(x)         # [B, 25, d_model]
        L = x.size(1)
        x = x + self.pos_embed[:, :L, :]  # Add positional info
        return x


class FeedForward(nn.Module):
    """Post-attention FFN with residual + DropPath."""
    def __init__(self, d_model, expansion=4, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        return x + self.drop_path(self.net(x))


# ── FIX H: Cross-Modal Attention with DropPath ──
class CrossModalAttentionLayer(nn.Module):
    """
    Bidirectional Cross-Modal Attention with DropPath (stochastic depth).
    Same structure as Script 14 but with DropPath on residual connections.
    """
    def __init__(self, d_model=128, nhead=4, dropout=0.1, drop_path=0.0):
        super().__init__()
        # 1D → 3D
        self.cross_attn_seq2shape = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm_seq = nn.LayerNorm(d_model)
        self.drop_path_seq = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ffn_seq = FeedForward(d_model, expansion=4, dropout=dropout, drop_path=drop_path)

        # 3D → 1D
        self.cross_attn_shape2seq = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm_shape = nn.LayerNorm(d_model)
        self.drop_path_shape = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ffn_shape = FeedForward(d_model, expansion=4, dropout=dropout, drop_path=drop_path)

    def forward(self, seq_features, shape_features, seq_key_padding_mask=None):
        # 1D → 3D: sequence queries structure
        attended_seq, _ = self.cross_attn_seq2shape(
            query=seq_features, key=shape_features, value=shape_features
        )
        seq_out = self.norm_seq(seq_features + self.drop_path_seq(attended_seq))
        seq_out = self.ffn_seq(seq_out)

        # 3D → 1D: structure queries sequence
        attended_shape, _ = self.cross_attn_shape2seq(
            query=shape_features, key=seq_features, value=seq_features,
            key_padding_mask=seq_key_padding_mask
        )
        shape_out = self.norm_shape(shape_features + self.drop_path_shape(attended_shape))
        shape_out = self.ffn_shape(shape_out)

        return seq_out, shape_out


# ── FIX J: Attention Pooling ──
class AttentionPooling(nn.Module):
    """
    Learnable query token that attends over sequence to produce a single vector.
    Better than mean pooling: learns to focus on the most informative positions.
    """
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x: [B, L, d_model]
            key_padding_mask: [B, L] bool, True for padding
        Returns:
            [B, d_model]
        """
        B = x.size(0)
        q = self.query.expand(B, -1, -1)  # [B, 1, d_model]
        out, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask)
        return self.norm(out.squeeze(1))  # [B, d_model]


# ── Full Model: CrossModalKANClassifier ──
class CrossModalKANClassifier(nn.Module):
    """
    Script 15: KAN-Regularized Cross-Modal Attention Dual-Branch.

    Fixes all 4 identified bottlenecks:
      1. Overfitting → aggressive regularization (DropPath, R-Drop, fewer unfrozen layers)
      2. MLP rigidity → ChebyKAN curved decision boundaries
      3. Shallow attention → 2-layer cross-attention
      4. Equal error treatment → Focal Loss (external)

    Additional improvements:
      - Positional embedding for shape spatial positions
      - Attention pooling for sequence aggregation
      - DropPath stochastic depth in cross-attention
    """
    def __init__(self, backbone, embedding_dim=768,
                 shape_in_channels=5, shape_conv_channels=None,
                 d_model=128, nhead=4, num_cross_layers=2, cross_dropout=0.15,
                 drop_path_rate=0.1,
                 kan_hidden_dim=64, kan_degree=4, num_classes=4,
                 fusion_dropout=0.5):
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

        # ── Branch 2: SpatialShapeCNN with positional embedding ──
        if shape_conv_channels is None:
            shape_conv_channels = [32, 64, 128]
        self.shape_cnn = SpatialShapeCNN(
            in_channels=shape_in_channels,
            conv_channels=shape_conv_channels,
            d_model=d_model,
        )

        # ── Cross-Modal Attention (2 layers with increasing DropPath) ──
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_cross_layers)]
        self.cross_attention_layers = nn.ModuleList([
            CrossModalAttentionLayer(
                d_model=d_model, nhead=nhead,
                dropout=cross_dropout, drop_path=dpr[i]
            )
            for i in range(num_cross_layers)
        ])

        # ── Attention Pooling for sequence (FIX J) ──
        self.seq_attn_pool = AttentionPooling(d_model, nhead=nhead, dropout=cross_dropout)

        # ── KAN Classifier (FIX G: replaces MLP) ──
        fusion_dim = d_model * 2  # 128 + 128 = 256
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            ChebyKANLinear(fusion_dim, kan_hidden_dim, degree=kan_degree),
            nn.LayerNorm(kan_hidden_dim),
            nn.Dropout(p=fusion_dropout),
            ChebyKANLinear(kan_hidden_dim, num_classes, degree=kan_degree),
        )

    def forward(self, input_ids, attention_mask, shape_features):
        # ── Branch 1: DNABERT-2 full token outputs ──
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]  # [B, T, 768]
        seq_features = self.seq_projection(hidden_states)  # [B, T, d_model]

        # ── Branch 2: SpatialShapeCNN + positional embedding ──
        shape_feats = self.shape_cnn(shape_features)  # [B, 25, d_model]

        # ── Cross-Modal Attention (2 layers) ──
        seq_key_padding_mask = (attention_mask == 0)
        for cross_layer in self.cross_attention_layers:
            seq_features, shape_feats = cross_layer(
                seq_features, shape_feats,
                seq_key_padding_mask=seq_key_padding_mask
            )

        # ── Pooling (AFTER cross-attention) ──
        # Seq: Attention Pooling (learnable query, handles padding)
        seq_pooled = self.seq_attn_pool(seq_features, key_padding_mask=seq_key_padding_mask)

        # Shape: Mean pooling (dense, short sequence)
        shape_pooled = shape_feats.mean(dim=1)

        # ── Fusion + KAN Classification ──
        fused = torch.cat([seq_pooled, shape_pooled], dim=1)  # [B, 2*d_model]
        return self.classifier(fused)


# ═══════════════════════════════════════════════════════════════════════
# CELL 8: DualBranchDataset & DataLoaders
# ═══════════════════════════════════════════════════════════════════════

class DualBranchDataset(Dataset):
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


train_dataset = DualBranchDataset(seq_train, y_train, shape_train_norm, tokenizer, cfg.MAX_TOKEN_LENGTH)
test_dataset = DualBranchDataset(seq_test, y_test, shape_test_norm, tokenizer, cfg.MAX_TOKEN_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)

print(f"DataLoaders: {len(train_loader)} train, {len(test_loader)} test batches")
print(f"Grad accum: {cfg.GRAD_ACCUM_STEPS} -> effective batch = {cfg.BATCH_SIZE * cfg.GRAD_ACCUM_STEPS}")

# ═══════════════════════════════════════════════════════════════════════
# CELL 9: Build Model & Optimizer (6-Group Differential LR)
# ═══════════════════════════════════════════════════════════════════════

model = CrossModalKANClassifier(
    backbone=dnabert_model,
    embedding_dim=cfg.EMBEDDING_DIM,
    shape_in_channels=cfg.SHAPE_CHANNELS,
    shape_conv_channels=cfg.SHAPE_CONV_CHANNELS,
    d_model=cfg.CROSS_ATTN_D_MODEL,
    nhead=cfg.CROSS_ATTN_NHEAD,
    num_cross_layers=cfg.CROSS_ATTN_LAYERS,
    cross_dropout=cfg.CROSS_ATTN_DROPOUT,
    drop_path_rate=cfg.DROP_PATH_RATE,
    kan_hidden_dim=cfg.KAN_HIDDEN_DIM,
    kan_degree=cfg.KAN_DEGREE,
    num_classes=cfg.NUM_CLASSES,
    fusion_dropout=cfg.FUSION_DROPOUT,
)
model = model.to(DEVICE)

# ── 6-Group Parameter Separation ──
backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
seq_proj_params = list(model.seq_projection.parameters())
shape_params = list(model.shape_cnn.parameters())
cross_attn_params = list(model.cross_attention_layers.parameters())
attn_pool_params = list(model.seq_attn_pool.parameters())
head_params = list(model.classifier.parameters())

optimizer = optim.AdamW([
    {"params": backbone_params,   "lr": cfg.BACKBONE_LR,   "weight_decay": cfg.WEIGHT_DECAY},
    {"params": seq_proj_params,   "lr": cfg.SEQ_PROJ_LR,   "weight_decay": cfg.WEIGHT_DECAY},
    {"params": shape_params,      "lr": cfg.SHAPE_LR,      "weight_decay": cfg.WEIGHT_DECAY},
    {"params": cross_attn_params, "lr": cfg.CROSS_ATTN_LR, "weight_decay": cfg.WEIGHT_DECAY},
    {"params": attn_pool_params,  "lr": cfg.ATTN_POOL_LR,  "weight_decay": cfg.WEIGHT_DECAY},
    {"params": head_params,       "lr": cfg.HEAD_LR,       "weight_decay": cfg.WEIGHT_DECAY},
])

total_steps = (len(train_loader) // cfg.GRAD_ACCUM_STEPS) * cfg.EPOCHS
warmup_steps = int(total_steps * cfg.WARMUP_RATIO)

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Class weights
class_counts = np.bincount(y_train)
class_weights = len(y_train) / (len(class_counts) * class_counts.astype(np.float32))
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
print(f"  Class weights: " + ", ".join([f"{cfg.CLASS_NAMES[i]}={class_weights[i]:.4f}" for i in range(len(class_counts))]))

# ── FIX K: Focal Loss ──
class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)
    gamma > 0 down-weights easy examples, focusing on hard SP1/SP4 confusion.
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.alpha,
            reduction='none', label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


criterion = FocalLoss(
    alpha=class_weights_tensor,
    gamma=cfg.FOCAL_GAMMA,
    label_smoothing=cfg.LABEL_SMOOTHING,
)
scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

# Print model summary
groups = {
    "DNABERT-2 backbone": (backbone_params, cfg.BACKBONE_LR),
    "Seq Projection":     (seq_proj_params, cfg.SEQ_PROJ_LR),
    "SpatialShapeCNN":    (shape_params, cfg.SHAPE_LR),
    "CrossModalAttn(x2)": (cross_attn_params, cfg.CROSS_ATTN_LR),
    "AttentionPooling":   (attn_pool_params, cfg.ATTN_POOL_LR),
    "KAN Classifier":     (head_params, cfg.HEAD_LR),
}
total_trainable = 0
print(f"\n{'='*60}")
print("MODEL SUMMARY -- KAN-Regularized Cross-Modal Attention")
print(f"{'='*60}")
for name, (params, lr) in groups.items():
    n = sum(p.numel() for p in params)
    total_trainable += n
    print(f"  {name:25s}: {n:>10,} params (LR={lr})")
print(f"  {'─'*50}")
print(f"  {'Total trainable':25s}: {total_trainable:>10,} params")
print(f"  Training steps: {total_steps} | Warmup: {warmup_steps}")
print(f"\n  FIXES vs Script 14:")
print(f"    [A] Unfrozen layers:  6 -> {cfg.UNFREEZE_LAST_N_LAYERS}")
print(f"    [B] Fusion dropout:   0.3 -> {cfg.FUSION_DROPOUT}")
print(f"    [C] R-Drop alpha:     {cfg.RDROP_ALPHA}")
print(f"    [D] Weight decay:     0.1 -> {cfg.WEIGHT_DECAY}")
print(f"    [E] Grad clip:        1.0 -> {cfg.MAX_GRAD_NORM}")
print(f"    [F] Backbone LR:      1e-5 -> {cfg.BACKBONE_LR}")
print(f"    [G] Classifier:       MLP -> ChebyKAN(degree={cfg.KAN_DEGREE})")
print(f"    [H] Cross-Attn:       1 -> {cfg.CROSS_ATTN_LAYERS} layers + DropPath({cfg.DROP_PATH_RATE})")
print(f"    [I] Shape pos embed:  NEW")
print(f"    [J] Attn pooling:     NEW (replaces mean pool for seq)")
print(f"    [K] Loss:             CE -> Focal(gamma={cfg.FOCAL_GAMMA})")

# ═══════════════════════════════════════════════════════════════════════
# CELL 10: Training Loop with R-Drop & Focal Loss
# ═══════════════════════════════════════════════════════════════════════

def compute_rdrop_kl(logits1, logits2):
    """Symmetric KL divergence for R-Drop regularization."""
    p1 = F.log_softmax(logits1, dim=-1)
    p2 = F.log_softmax(logits2, dim=-1)
    kl1 = F.kl_div(p1, p2.detach().exp(), reduction='batchmean')
    kl2 = F.kl_div(p2, p1.detach().exp(), reduction='batchmean')
    return (kl1 + kl2) / 2


def train_one_epoch(model, train_loader, optimizer, scheduler, criterion,
                    scaler, device, grad_accum_steps, rdrop_alpha=1.0, max_grad_norm=0.5):
    """Train with R-Drop regularization (dual forward pass + KL divergence)."""
    model.train()
    running_loss = 0.0
    running_ce = 0.0
    running_kl = 0.0
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
            # R-Drop: Two forward passes with different dropout masks
            logits1 = model(input_ids, attention_mask, shape_features)
            logits2 = model(input_ids, attention_mask, shape_features)

            # Focal Loss on both outputs
            loss1 = criterion(logits1, labels)
            loss2 = criterion(logits2, labels)
            ce_loss = (loss1 + loss2) / 2

            # KL divergence between the two distributions
            kl_loss = compute_rdrop_kl(logits1, logits2)

            # Total loss
            loss = ce_loss + rdrop_alpha * kl_loss
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        batch_loss = loss.item() * grad_accum_steps
        running_loss += batch_loss * labels.size(0)
        running_ce += ce_loss.item() * labels.size(0)
        running_kl += kl_loss.item() * labels.size(0)

        # Use logits1 for accuracy tracking
        _, predicted = logits1.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            "loss": f"{running_loss/total:.4f}",
            "acc": f"{correct/total:.4f}",
            "kl": f"{running_kl/total:.4f}",
        })

    return running_loss / total, correct / total, running_ce / total, running_kl / total


@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
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
    print("\n" + "=" * 60)
    print("TRAINING -- KAN-Regularized Cross-Modal Attention")
    print("=" * 60)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "ce_loss": [], "kl_loss": []}
    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(cfg.EPOCHS):
        t0 = time.time()

        train_loss, train_acc, ce_loss, kl_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            scaler, device, cfg.GRAD_ACCUM_STEPS,
            rdrop_alpha=cfg.RDROP_ALPHA, max_grad_norm=cfg.MAX_GRAD_NORM,
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["ce_loss"].append(ce_loss)
        history["kl_loss"].append(kl_loss)

        gap = train_acc - val_acc

        print(
            f"Epoch {epoch+1:02d}/{cfg.EPOCHS} | "
            f"Train: {train_loss:.4f}/{train_acc:.4f} | "
            f"Val: {val_loss:.4f}/{val_acc:.4f} | "
            f"Gap: {gap:.2%} | KL: {kl_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.1e} | {elapsed:.0f}s"
        )

        if val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, os.path.join(cfg.MODEL_DIR, "best_kan_crossmodal.pt"))
            print(f"  -> Saved best (val_acc={val_acc:.4f}, gap={gap:.2%})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1} (patience={cfg.PATIENCE})")
                break

    print(f"\nTraining complete. Best val_acc: {best_val_acc:.4f}")
    return history


history = train_model(model, train_loader, test_loader, optimizer, scheduler, criterion, scaler, cfg, DEVICE)

# ═══════════════════════════════════════════════════════════════════════
# CELL 11: Load Best Model & Full Evaluation
# ═══════════════════════════════════════════════════════════════════════

best_path = os.path.join(cfg.MODEL_DIR, "best_kan_crossmodal.pt")
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
        f.write("CLASSIFICATION REPORT -- KAN-Regularized Cross-Modal Attention\n")
        f.write("=" * 60 + "\n")
        f.write(report_str)
    print(f"  -> Report saved: {report_path}")

    acc = accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average="macro")
    print(f"Overall Accuracy:  {acc:.4f}")
    print(f"Macro F1-Score:    {f1_macro:.4f}")
    return all_preds, all_targets, all_probs


all_preds, all_targets, all_probs = full_evaluation(model, test_loader, cfg.CLASS_NAMES, DEVICE)

# BED export
def parse_header_to_bed(header):
    try:
        clean = header.split("_")[0]
        chrom, coords = clean.split(":")
        start, end = coords.split("-")
        return chrom, start, end
    except Exception:
        return None

def export_igv_bed_files(headers_test, predictions, targets, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    beds = {"True_SP1": [], "True_SP4": [], "Confused_SP4_as_SP1": []}
    for idx, (pred, target) in enumerate(zip(predictions, targets)):
        fields = parse_header_to_bed(headers_test[idx])
        if not fields: continue
        line = f"{fields[0]}\t{fields[1]}\t{fields[2]}\n"
        if target == 0 and pred == 0: beds["True_SP1"].append(line)
        elif target == 2 and pred == 2: beds["True_SP4"].append(line)
        elif target == 2 and pred == 0: beds["Confused_SP4_as_SP1"].append(line)
    for name, lines in beds.items():
        with open(os.path.join(output_dir, f"{name}.bed"), "w") as f:
            f.writelines(lines)
        print(f"    {name}.bed: {len(lines)} lines")

print("\n  BED files:")
export_igv_bed_files(headers_test, all_preds, all_targets, cfg.OUTPUT_DIR)

# ═══════════════════════════════════════════════════════════════════════
# CELL 12: Generate All Performance Figures
# ═══════════════════════════════════════════════════════════════════════

TITLE_PREFIX = "KAN Cross-Modal Attention (DNABERT-2 + DNAshape)"

def plot_training_curves(history, save_dir):
    epochs = len(history["train_loss"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(range(1, epochs+1), history["train_loss"], label="Train", color="#2196F3", lw=2)
    axes[0, 0].plot(range(1, epochs+1), history["val_loss"], label="Val", color="#FF5722", lw=2)
    axes[0, 0].set_title("Total Loss", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(); axes[0, 0].grid(True, ls="--", alpha=0.4)

    # Accuracy
    axes[0, 1].plot(range(1, epochs+1), history["train_acc"], label="Train", color="#4CAF50", lw=2)
    axes[0, 1].plot(range(1, epochs+1), history["val_acc"], label="Val", color="#E91E63", lw=2)
    axes[0, 1].axhline(y=0.25, color="gray", ls="--", alpha=0.5, label="Random")
    axes[0, 1].set_title("Accuracy", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Acc")
    axes[0, 1].legend(); axes[0, 1].grid(True, ls="--", alpha=0.4)

    # Overfitting Gap
    gaps = [t - v for t, v in zip(history["train_acc"], history["val_acc"])]
    axes[1, 0].plot(range(1, epochs+1), gaps, color="#9C27B0", lw=2)
    axes[1, 0].axhline(y=0, color="green", ls="--", alpha=0.5)
    axes[1, 0].fill_between(range(1, epochs+1), gaps, alpha=0.15, color="#9C27B0")
    axes[1, 0].set_title("Overfitting Gap (Train - Val)", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Gap")
    axes[1, 0].grid(True, ls="--", alpha=0.4)

    # R-Drop KL
    axes[1, 1].plot(range(1, epochs+1), history["kl_loss"], color="#FF9800", lw=2)
    axes[1, 1].set_title("R-Drop KL Divergence", fontweight="bold")
    axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("KL")
    axes[1, 1].grid(True, ls="--", alpha=0.4)

    plt.suptitle(f"{TITLE_PREFIX} -- Training Progress", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "kan_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  -> Training curves: {path}")


def plot_confusion_matrix(all_targets, all_preds, class_names, save_dir):
    cm = confusion_matrix(all_targets, all_preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, data, title, fmt in [(ax1, cm, "Counts", "d"), (ax2, cm_norm, "Normalized", ".2%")]:
        im = ax.imshow(data, cmap="Blues", interpolation="nearest")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        thresh = data.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if data[i, j] > thresh else "black", fontsize=12, fontweight="bold")
    plt.suptitle(TITLE_PREFIX, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "kan_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  -> Confusion matrix: {path}")


def plot_roc_curves(all_targets, all_probs, class_names, save_dir):
    n_classes = len(class_names)
    y_bin = label_binarize(all_targets, classes=list(range(n_classes)))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    plt.figure(figsize=(9, 7))
    fpr_d, tpr_d, auc_d = {}, {}, {}
    for i in range(n_classes):
        fpr_d[i], tpr_d[i], _ = roc_curve(y_bin[:, i], all_probs[:, i])
        auc_d[i] = auc(fpr_d[i], tpr_d[i])
        plt.plot(fpr_d[i], tpr_d[i], color=colors[i], lw=2, label=f"{class_names[i]} (AUC={auc_d[i]:.4f})")
    all_fpr = np.unique(np.concatenate([fpr_d[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_d[i], tpr_d[i])
    mean_tpr /= n_classes
    plt.plot(all_fpr, mean_tpr, color="#9C27B0", lw=2.5, ls="--", label=f"Macro (AUC={auc(all_fpr, mean_tpr):.4f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4); plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC -- {TITLE_PREFIX}", fontweight="bold"); plt.legend(loc="lower right"); plt.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "kan_roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  -> ROC curves: {path}")


def plot_pr_curves(all_targets, all_probs, class_names, save_dir):
    n_classes = len(class_names)
    y_bin = label_binarize(all_targets, classes=list(range(n_classes)))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    plt.figure(figsize=(9, 7))
    for i in range(n_classes):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], all_probs[:, i])
        ap = average_precision_score(y_bin[:, i], all_probs[:, i])
        plt.plot(rec, prec, color=colors[i], lw=2, label=f"{class_names[i]} (AP={ap:.4f})")
    plt.axhline(y=0.25, color="gray", ls="--", alpha=0.5)
    plt.xlim([0, 1]); plt.ylim([0, 1.05]); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR -- {TITLE_PREFIX}", fontweight="bold"); plt.legend(loc="lower left"); plt.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "kan_pr_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  -> PR curves: {path}")


def plot_per_class_bar(all_targets, all_preds, class_names, save_dir):
    prec, rec, f1, support = precision_recall_fscore_support(all_targets, all_preds, average=None)
    x = np.arange(len(class_names)); w = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (data, label, color) in enumerate([(prec, "Precision", "#2196F3"), (rec, "Recall", "#4CAF50"), (f1, "F1", "#FF9800")]):
        bars = ax.bar(x + (i-1)*w, data, w, label=label, color=color, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x()+w/2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([f"{c}\n(n={s})" for c, s in zip(class_names, support)])
    ax.set_ylabel("Score"); ax.set_title(f"Per-Class -- {TITLE_PREFIX}", fontweight="bold")
    ax.set_ylim(0, 1.15); ax.legend(); ax.grid(True, ls="--", alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(save_dir, "kan_per_class_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  -> Per-class: {path}")


print("\n" + "=" * 60)
print("GENERATING FIGURES")
print("=" * 60)
plot_training_curves(history, cfg.FIG_DIR)
plot_confusion_matrix(all_targets, all_preds, cfg.CLASS_NAMES, cfg.FIG_DIR)
plot_roc_curves(all_targets, all_probs, cfg.CLASS_NAMES, cfg.FIG_DIR)
plot_pr_curves(all_targets, all_probs, cfg.CLASS_NAMES, cfg.FIG_DIR)
plot_per_class_bar(all_targets, all_preds, cfg.CLASS_NAMES, cfg.FIG_DIR)

# ═══════════════════════════════════════════════════════════════════════
# CELL 13: Summary & Comparison
# ═══════════════════════════════════════════════════════════════════════

best_gap = history["train_acc"][checkpoint["epoch"]-1] - checkpoint["val_acc"] if checkpoint["epoch"] <= len(history["train_acc"]) else 0
final_gap = history["train_acc"][-1] - history["val_acc"][-1]

print("\n" + "=" * 60)
print("PIPELINE SUMMARY -- KAN-Regularized Cross-Modal Attention")
print("=" * 60)
print(f"  Architecture:      KAN Cross-Modal Dual-Branch")
print(f"  DNABERT-2:         Last {cfg.UNFREEZE_LAST_N_LAYERS} layers (was 6)")
print(f"  Cross-Attention:   {cfg.CROSS_ATTN_LAYERS} layers, DropPath={cfg.DROP_PATH_RATE}")
print(f"  Classifier:        ChebyKAN(degree={cfg.KAN_DEGREE}, hidden={cfg.KAN_HIDDEN_DIM})")
print(f"  Regularization:    R-Drop(a={cfg.RDROP_ALPHA}) + Focal(g={cfg.FOCAL_GAMMA}) + WD={cfg.WEIGHT_DECAY}")
print(f"  Best val_acc:      {checkpoint['val_acc']:.4f}")
print(f"  Overfitting gap:   {final_gap:.2%} (was 34% in Script 14)")
print()
print("  +-----------------------------------------------------------+")
print("  |  COMPARISON                                               |")
print("  +-----------------------------------------------------------+")
print("  |  Sc.12: Balanced Dual-Branch (torch.cat)                  |")
print("  |    -> Val Acc: ~50.33%   Gap: ~30%                        |")
print("  |                                                           |")
print("  |  Sc.14: Cross-Modal Attention                             |")
print("  |    -> Val Acc: 61.33%    Gap: ~34%                        |")
print("  |                                                           |")
print("  |  Sc.15: KAN + Anti-Overfitting Cross-Modal                |")
print(f"  |    -> Val Acc: {checkpoint['val_acc']:.2%}    Gap: {final_gap:.0%}{'':20s}|")
print("  +-----------------------------------------------------------+")
print("=" * 60)
