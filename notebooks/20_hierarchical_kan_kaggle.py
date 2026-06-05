#!/usr/bin/env python3
"""
Script 19: G-CMAB Safe Core — Incremental Improvements on Script 18 Baseline
Designed for: Kaggle GPU (1×T4 or 2×T4 via HuggingFace Accelerate)
Task: 4-class SP1/SP2/SP4/Negative TF-binding classification

BUILD PHILOSOPHY (Lessons from Scripts 15→18):
  - Script 16 collapsed to 25% by changing 11 things at once.
  - Each improvement here is an independent flag — test one at a time.
  - Keep Script 18 core (d_model=128, 1 cross-attn, weighted-CE, no pos-embed/EMA).

SAFE IMPROVEMENTS (4 flags):
  1. USE_STRIDED_CONV = True   — Strided Conv replaces MaxPool in shape CNN
  2. USE_GROUPNORM   = True   — GroupNorm replaces BatchNorm (multi-GPU safe)
  3. USE_LAYER_ATTN  = True   — Scalar-mix over multiple BERT layers (ELMo-style)
  4. USE_MULTI_POOL  = True   — seq: mean‖max pooling, shape: K-Max(k=4) pooling

Multi-GPU via HuggingFace Accelerate (automatic, no code change needed for 1-GPU).
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
import copy
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
)
from sklearn.preprocessing import label_binarize
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()

# ── HuggingFace Accelerate ──
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

if torch.cuda.is_available():
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
        # Recursive fallback search
        for root, _, files in os.walk(fallback_dir):
            if filename in files:
                return os.path.join(root, filename)
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
    """Script 19: G-CMAB Safe Core config."""

    # ── Paths ──
    FASTA_DIR = auto_detect_dir("sp1_positive_final.fasta", "data/processed")
    SHAPE_DIR = auto_detect_dir("dnashape_sp1.npy", "data/processed")
    OUTPUT_DIR = "outputs_gcmab_safe"
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
    BACKBONE_LR = 2e-5
    USE_LAYERWISE_LR_DECAY = False

    # ── Cross-Modal Attention (kept from Script 18) ──
    CROSS_ATTN_D_MODEL = 128
    CROSS_ATTN_NHEAD = 4
    CROSS_ATTN_LAYERS = 1
    CROSS_ATTN_DROPOUT = 0.1
    CROSS_ATTN_LR = 2e-4
    SEQ_PROJ_LR = 2e-4

    # ── DNAshape Branch ──
    SHAPE_CHANNELS = 5
    SHAPE_SEQ_LEN = 101
    SHAPE_CONV_CHANNELS = [32, 64, 128]
    SHAPE_LR = 3e-4

    # ══════════════════════════════════════════════════════════════════
    # SCRIPT 19 FLAGS — each toggleable independently for ablation
    # ══════════════════════════════════════════════════════════════════

    # Flag 1: Strided Conv replaces MaxPool in shape CNN
    USE_STRIDED_CONV = True

    # Flag 2: GroupNorm replaces BatchNorm (multi-GPU safe, stable on small batch)
    USE_GROUPNORM = True
    GROUPNORM_GROUPS = 16

    # Flag 3: Layer-Attention (scalar-mix) over BERT hidden states
    USE_LAYER_ATTN = True
    LAYER_ATTN_N = 6  # Number of last BERT layers to mix

    # Flag 4: Multi-statistic pooling (seq: mean‖max, shape: K-Max)
    USE_MULTI_POOL = True
    KMAX_K = 4

    # ══════════════════════════════════════════════════════════════════

    # ── Classifier Head ──
    HEAD_TYPE = "mlp"
    HIDDEN_DIM = 256
    FUSION_DROPOUT = 0.5
    HEAD_LR = 2e-4

    # ── Loss ──
    LOSS_TYPE = "weighted_ce"
    LABEL_SMOOTHING = 0.0
    NUM_CLASSES = 4
    WEIGHT_DECAY = 0.1

    # ── Training ──
    BATCH_SIZE = 16
    # With 2×T4 Accelerate: effective batch = 16×2 = 32 → no grad accum needed
    # With 1×T4: keep grad_accum=4 for effective 64
    GRAD_ACCUM_STEPS = 1  # Accelerate handles multi-GPU; set >1 for single GPU if desired
    EPOCHS = 30
    PATIENCE = 12
    MAX_OVERFITTING_GAP = 30.0  # Max train-val gap (%) to prevent severe overfitting
    WARMUP_RATIO = 0.15
    MAX_GRAD_NORM = 0.5

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

    # ── Layer-Attention LR ──
    LAYER_ATTN_LR = 1e-3  # Scalar-mix weights learn fast

cfg = Config()

for d in [cfg.OUTPUT_DIR, cfg.FIG_DIR, cfg.MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

random.seed(cfg.RANDOM_SEED)
np.random.seed(cfg.RANDOM_SEED)
torch.manual_seed(cfg.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.RANDOM_SEED)

# ── Initialize Accelerator ──
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=cfg.GRAD_ACCUM_STEPS,
    kwargs_handlers=[ddp_kwargs],
)
DEVICE = accelerator.device

# Redefine print globally to suppress non-main process logging in DDP/Multi-GPU
if not accelerator.is_main_process:
    import builtins
    builtins.print = lambda *args, **kwargs: None

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    try:
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    except AttributeError:
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    n_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {n_gpus}")

print(f"\nUsing device: {DEVICE}")
print(f"Num processes: {accelerator.num_processes}")
print(f"Architecture: G-CMAB Safe Core (Script 19)")
print(f"  Base: Script 18 (d_model={cfg.CROSS_ATTN_D_MODEL}, 1 cross-attn, weighted-CE)")
print(f"  Flag 1 — Strided Conv:   {cfg.USE_STRIDED_CONV}")
print(f"  Flag 2 — GroupNorm:       {cfg.USE_GROUPNORM}")
print(f"  Flag 3 — Layer-Attention: {cfg.USE_LAYER_ATTN} (N={cfg.LAYER_ATTN_N})")
print(f"  Flag 4 — Multi-Pool:      {cfg.USE_MULTI_POOL} (K-Max k={cfg.KMAX_K})")
print("=" * 60)

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
    if accelerator.is_main_process:
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
        if accelerator.is_main_process:
            print(f"  {cls_name} shape: {shape_data.shape} ({os.path.basename(fpath)})")
        all_shapes.append(shape_data)

    return np.concatenate(all_shapes, axis=0)


def load_all_data(fasta_dir, shape_dir, shape_files):
    """Load all 4 classes: sequences + shape features + group-aware labels."""
    if accelerator.is_main_process:
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
    if accelerator.is_main_process:
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
        if accelerator.is_main_process:
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

    if accelerator.is_main_process:
        print("\n  Loading DNAshape features...")
    all_shapes = load_shape_features(shape_dir, shape_files)

    assert len(all_sequences) == all_shapes.shape[0], (
        f"Sequence count ({len(all_sequences)}) != shape count ({all_shapes.shape[0]})"
    )

    if accelerator.is_main_process:
        print(f"\n  Total: {len(all_sequences)} sequences, {group_id} groups")
        print(f"  Shape features: {all_shapes.shape}")
        print(f"  Class distribution: {np.bincount(all_labels)}")
    return all_sequences, all_labels, all_groups, all_shapes, all_headers


def split_data(sequences, labels, groups, shapes, headers, test_size=0.2, seed=42):
    """Group-aware train/test split (no revcomp leakage)."""
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
    if accelerator.is_main_process:
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
        if accelerator.is_main_process:
            print(f"  {channel_names[ch]:>5s}: median={median_val:>8.4f}, "
                  f"P1={p1_val:>8.4f}, P99={p99_val:>8.4f}, scale={scale:>8.4f}")

    nan_train = np.isnan(shape_train_norm).sum()
    nan_test = np.isnan(shape_test_norm).sum()
    shape_train_norm = np.nan_to_num(shape_train_norm, nan=0.0)
    shape_test_norm = np.nan_to_num(shape_test_norm, nan=0.0)
    if accelerator.is_main_process:
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
    if accelerator.is_main_process:
        print(f"  Patched {patched} flash-attention refs -> pure PyTorch."
              if patched else "  No flash-attn refs to patch.")

# ═══════════════════════════════════════════════════════════════════════
# CELL 6: Load DNABERT-2 Backbone (with Selective Unfreezing)
# ═══════════════════════════════════════════════════════════════════════

def load_dnabert2(model_name, unfreeze_last_n=6):
    """Load DNABERT-2 with selective layer unfreezing.
    Returns model on CPU — Accelerate will handle device placement.
    """
    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("LOADING DNABERT-2 BACKBONE (Selective Fine-Tuning)")
        print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 3

    # Enable hidden_states output for Layer-Attention
    if cfg.USE_LAYER_ATTN:
        config.output_hidden_states = True

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
        if accelerator.is_main_process:
            print("  Strategy 1 (direct load) OK")
    except Exception as e:
        if accelerator.is_main_process:
            print(f"  Strategy 1 failed: {e}")
        model = None

    # Strategy 2: Empty init + manual state_dict
    if model is None:
        try:
            if accelerator.is_main_process:
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
            if accelerator.is_main_process:
                print("  Strategy 2 OK")
        except Exception as e:
            if accelerator.is_main_process:
                print(f"  Strategy 2 failed: {e}")
            model = None

    # Strategy 3: Monkey-patch torch.empty meta->cpu
    if model is None:
        try:
            if accelerator.is_main_process:
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
            if accelerator.is_main_process:
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

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if accelerator.is_main_process:
        print(f"\n  DNABERT-2 loaded")
        print(f"  Total encoder layers: {total_layers}")
        print(f"  Frozen: layers 0-{unfreeze_from-1} | Unfrozen: layers {unfreeze_from}-{total_layers-1} ({unfreeze_last_n})")
        print(f"  Total: {total_params:,} | Trainable: {trainable:,} ({100*trainable/total_params:.1f}%)")
        if cfg.USE_LAYER_ATTN:
            print(f"  Layer-Attention: mixing last {cfg.LAYER_ATTN_N} hidden states")
    return model, tokenizer


dnabert_model, tokenizer = load_dnabert2(cfg.DNABERT_MODEL, cfg.UNFREEZE_LAST_N_LAYERS)

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
    chosen = int(max(floor, min(cap, p_val + 2)))
    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("AUTO MAX TOKEN LENGTH")
        print("=" * 60)
        print(f"  Token length (sample n={len(sample)}): "
              f"min={min(lengths)}, mean={np.mean(lengths):.1f}, "
              f"p{percentile}={p_val}, max={max(lengths)}")
        print(f"  Chosen MAX_TOKEN_LENGTH = {chosen}")
    return chosen

if cfg.AUTO_MAX_LENGTH:
    MAX_LENGTH = compute_max_token_length(
        seq_train, tokenizer,
        floor=cfg.MAX_LENGTH_FLOOR, cap=cfg.MAX_LENGTH_CAP,
    )
else:
    MAX_LENGTH = cfg.MAX_TOKEN_LENGTH
    if accelerator.is_main_process:
        print(f"\n  Using fixed MAX_TOKEN_LENGTH = {MAX_LENGTH}")

# ═══════════════════════════════════════════════════════════════════════
# CELL 7: Architecture — G-CMAB Safe Core
# ═══════════════════════════════════════════════════════════════════════

# ---------- Component 1: Layer-Attention (Scalar-Mix) ----------

class LayerAttention(nn.Module):
    """
    ELMo-style scalar-mix: learns a weighted combination of BERT hidden states.
    Much cheaper than keeping 768 dimensions — only N+1 parameters.
    Motif-related features often live in middle layers, not just the last.
    """
    def __init__(self, n_layers):
        super().__init__()
        # Learnable weights for each layer + global scalar
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, hidden_states_list):
        """
        Args:
            hidden_states_list: list of [B, T, D] tensors (one per layer)
        Returns:
            mixed: [B, T, D] — weighted combination
        """
        weights = F.softmax(self.layer_weights, dim=0)
        mixed = torch.zeros_like(hidden_states_list[0])
        for w, h in zip(weights, hidden_states_list):
            mixed = mixed + w * h
        return self.gamma * mixed


# ---------- Component 2: SpatialShapeCNN (with Strided Conv + GroupNorm flags) ----------

class SpatialShapeCNN(nn.Module):
    """
    Conv1D on DNAshape features, PRESERVING spatial dimension (no GlobalAvgPool).

    Script 19 changes (flagged):
      - USE_STRIDED_CONV: Conv1d(stride=2) replaces MaxPool1d — learns downsampling
      - USE_GROUPNORM: GroupNorm replaces BatchNorm — stable on small/split batches

    Input:  [B, 5, 101]  → Output: [B, ~26, d_model]
    """
    def __init__(self, in_channels=5, conv_channels=None, d_model=128,
                 use_strided=True, use_groupnorm=True, gn_groups=8):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 128]

        def make_norm(ch):
            if use_groupnorm:
                return nn.GroupNorm(min(gn_groups, ch), ch)
            else:
                return nn.BatchNorm1d(ch)

        if use_strided:
            # Strided conv replaces MaxPool — learns how to downsample
            self.conv_block1 = nn.Sequential(
                nn.Conv1d(in_channels, conv_channels[0], kernel_size=7, padding=3, stride=2),
                make_norm(conv_channels[0]),
                nn.GELU(),
            )  # [B, 32, 51]
            self.conv_block2 = nn.Sequential(
                nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=5, padding=2, stride=2),
                make_norm(conv_channels[1]),
                nn.GELU(),
            )  # [B, 64, 26]
        else:
            # Original Script 18: Conv + MaxPool
            self.conv_block1 = nn.Sequential(
                nn.Conv1d(in_channels, conv_channels[0], kernel_size=7, padding=3),
                make_norm(conv_channels[0]),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2),
            )  # [B, 32, 50]
            self.conv_block2 = nn.Sequential(
                nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=5, padding=2),
                make_norm(conv_channels[1]),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2),
            )  # [B, 64, 25]

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=3, padding=1),
            make_norm(conv_channels[2]),
            nn.GELU(),
        )  # [B, 128, 25 or 26]

        self.proj = nn.Linear(conv_channels[-1], d_model)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.transpose(1, 2)    # [B, L, conv_channels[-1]]
        x = self.proj(x)         # [B, L, d_model]
        return x


# ---------- Component 3: Cross-Modal Attention (unchanged from Script 18) ----------

class FeedForward(nn.Module):
    """Post-attention FFN with residual connection."""
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
      1D->3D: Sequence queries structural features (DNAshape)
      3D->1D: Structure queries sequence features (DNABERT-2)
    """
    def __init__(self, d_model=128, nhead=4, dropout=0.1):
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


# ---------- Component 4: Main Classifier ----------

class GCMABSafeClassifier(nn.Module):
    """
    Script 19: G-CMAB Safe Core Classifier.
    Builds on Script 18 with 4 independently togglable improvements.
    """
    def __init__(self, backbone, cfg):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        d_model = cfg.CROSS_ATTN_D_MODEL

        # ── Flag 3: Layer-Attention ──
        self.use_layer_attn = cfg.USE_LAYER_ATTN
        if self.use_layer_attn:
            self.layer_attention = LayerAttention(n_layers=cfg.LAYER_ATTN_N)
            self._layer_attn_fallback = False  # set to True if backbone doesn't support hidden_states
        else:
            self.layer_attention = None

        # Branch 1: DNABERT-2 token projection (768 -> d_model)
        self.seq_projection = nn.Sequential(
            nn.Linear(cfg.EMBEDDING_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Branch 2: SpatialShapeCNN (flags 1 & 2)
        self.shape_cnn = SpatialShapeCNN(
            in_channels=cfg.SHAPE_CHANNELS,
            conv_channels=cfg.SHAPE_CONV_CHANNELS,
            d_model=d_model,
            use_strided=cfg.USE_STRIDED_CONV,
            use_groupnorm=cfg.USE_GROUPNORM,
            gn_groups=cfg.GROUPNORM_GROUPS,
        )

        # Cross-Modal Attention stack (unchanged from Script 18)
        self.cross_attention_layers = nn.ModuleList([
            CrossModalAttentionLayer(d_model=d_model, nhead=cfg.CROSS_ATTN_NHEAD, dropout=cfg.CROSS_ATTN_DROPOUT)
            for _ in range(cfg.CROSS_ATTN_LAYERS)
        ])

        # ── Flag 4: Multi-statistic Pooling ──
        self.use_multi_pool = cfg.USE_MULTI_POOL
        self.kmax_k = cfg.KMAX_K

        if self.use_multi_pool:
            # seq: mean‖max → 2*d_model; shape: kmax → d_model
            fusion_dim = d_model * 3  # 2*d_model (seq) + d_model (shape)
        else:
            # Original Script 18: mean + mean → 2*d_model
            fusion_dim = d_model * 2

        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, cfg.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(p=cfg.FUSION_DROPOUT),
            nn.Linear(cfg.HIDDEN_DIM, cfg.NUM_CLASSES),
        )

    def _get_bert_features(self, input_ids, attention_mask):
        """Extract BERT features, optionally using Layer-Attention."""
        if self.use_layer_attn and not getattr(self, '_layer_attn_fallback', False):
            try:
                outputs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                # Try to get hidden_states
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    all_hidden = outputs.hidden_states  # tuple of (n_layers+1) tensors
                    # Take last N layers (skip embedding layer at index 0)
                    n = min(self.cfg.LAYER_ATTN_N, len(all_hidden) - 1)
                    selected = list(all_hidden[-n:])
                    return self.layer_attention(selected)
                else:
                    # Fallback: hidden_states not available in output
                    if accelerator.is_main_process:
                        print("  [LayerAttn] WARNING: hidden_states not in output, using hook fallback...")
                    self._layer_attn_fallback = True
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"  [LayerAttn] WARNING: output_hidden_states failed ({e}), using hook fallback...")
                self._layer_attn_fallback = True

        if self.use_layer_attn and getattr(self, '_layer_attn_fallback', False):
            # Hook-based fallback: register hooks on encoder layers
            hidden_states_collected = []
            hooks = []
            n = min(self.cfg.LAYER_ATTN_N, len(self.backbone.encoder.layer))
            start_layer = len(self.backbone.encoder.layer) - n

            def make_hook(storage):
                def hook_fn(module, input, output):
                    # output is typically (hidden_states, ...) or just hidden_states
                    if isinstance(output, tuple):
                        storage.append(output[0])
                    else:
                        storage.append(output)
                return hook_fn

            for i in range(start_layer, len(self.backbone.encoder.layer)):
                h = self.backbone.encoder.layer[i].register_forward_hook(make_hook(hidden_states_collected))
                hooks.append(h)

            _ = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

            for h in hooks:
                h.remove()

            mixed = self.layer_attention(hidden_states_collected)
            if mixed.dim() == 2:
                B = attention_mask.size(0)
                T = attention_mask.size(1)
                D = mixed.size(-1)
                padded = torch.zeros(B, T, D, dtype=mixed.dtype, device=mixed.device)
                padded[attention_mask.bool()] = mixed
                mixed = padded
            return mixed

        # No Layer-Attention: use last hidden state (Script 18 default)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return outputs[0]

    def _masked_mean_pool(self, features, attention_mask):
        """Masked mean pooling over sequence dimension."""
        mask_expanded = attention_mask.unsqueeze(-1).float()
        return (features * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)

    def _masked_max_pool(self, features, attention_mask):
        """Masked max pooling over sequence dimension."""
        mask_expanded = attention_mask.unsqueeze(-1).float()
        # Set padding positions to very large negative number
        features_masked = features.clone()
        features_masked[mask_expanded.squeeze(-1) == 0] = -1e9
        return features_masked.max(dim=1)[0]

    def _kmax_pool(self, features, k=4):
        """K-Max pooling: average of top-k values along sequence dim, per feature.
        More robust than pure Max (which is k=1) — captures top structural peaks
        without being dominated by a single outlier.
        """
        # features: [B, L, D]
        k_actual = min(k, features.size(1))
        # topk along dim=1
        topk_vals, _ = features.topk(k_actual, dim=1)  # [B, k, D]
        return topk_vals.mean(dim=1)  # [B, D]

    def forward(self, input_ids, attention_mask, shape_features):
        # ── BERT feature extraction (with optional Layer-Attention) ──
        hidden_states = self._get_bert_features(input_ids, attention_mask)  # [B, T, 768]
        seq_features = self.seq_projection(hidden_states)  # [B, T, d_model]

        # ── Shape CNN ──
        shape_feats = self.shape_cnn(shape_features)  # [B, L_shape, d_model]

        # ── Cross-Modal Attention ──
        seq_key_padding_mask = (attention_mask == 0)
        for cross_layer in self.cross_attention_layers:
            seq_features, shape_feats = cross_layer(
                seq_features, shape_feats, seq_key_padding_mask=seq_key_padding_mask
            )

        # ── Pooling (Flag 4) ──
        if self.use_multi_pool:
            # Seq: concat[masked_mean, masked_max] → 2*d_model
            seq_mean = self._masked_mean_pool(seq_features, attention_mask)
            seq_max = self._masked_max_pool(seq_features, attention_mask)
            seq_pooled = torch.cat([seq_mean, seq_max], dim=1)

            # Shape: K-Max pooling → d_model
            shape_pooled = self._kmax_pool(shape_feats, k=self.kmax_k)
        else:
            # Script 18 default
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

if accelerator.is_main_process:
    print(f"\nDataLoaders ready: {len(train_loader)} train batches, {len(test_loader)} test batches")
    print(f"Sequence max_length = {MAX_LENGTH}")

# ═══════════════════════════════════════════════════════════════════════
# CELL 9: Build Model, Optimizer, Loss
# ═══════════════════════════════════════════════════════════════════════

model = GCMABSafeClassifier(backbone=dnabert_model, cfg=cfg)

# ── Optimizer with component-specific learning rates ──
param_groups = []

backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
param_groups.append({"params": backbone_params, "lr": cfg.BACKBONE_LR, "weight_decay": cfg.WEIGHT_DECAY})

param_groups.append({"params": list(model.seq_projection.parameters()), "lr": cfg.SEQ_PROJ_LR, "weight_decay": cfg.WEIGHT_DECAY})
param_groups.append({"params": list(model.shape_cnn.parameters()), "lr": cfg.SHAPE_LR, "weight_decay": cfg.WEIGHT_DECAY})
param_groups.append({"params": list(model.cross_attention_layers.parameters()), "lr": cfg.CROSS_ATTN_LR, "weight_decay": cfg.WEIGHT_DECAY})
param_groups.append({"params": list(model.classifier.parameters()), "lr": cfg.HEAD_LR, "weight_decay": cfg.WEIGHT_DECAY})

# Layer-Attention parameters (scalar-mix learns fast → higher LR)
if cfg.USE_LAYER_ATTN and model.layer_attention is not None:
    param_groups.append({"params": list(model.layer_attention.parameters()), "lr": cfg.LAYER_ATTN_LR, "weight_decay": 0.0})

optimizer = optim.AdamW(param_groups)

total_steps = (len(train_loader) // max(cfg.GRAD_ACCUM_STEPS, 1)) * cfg.EPOCHS
warmup_steps = int(total_steps * cfg.WARMUP_RATIO)

def lr_lambda(current_step):
    """Linear warmup then cosine decay."""
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Class weights for CrossEntropy Loss
class_counts = np.bincount(y_train)
class_weights = len(y_train) / (len(class_counts) * class_counts.astype(np.float32))
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=accelerator.device)

if accelerator.is_main_process:
    print(f"  Class weights: " + ", ".join([f"{cfg.CLASS_NAMES[i]}={class_weights[i]:.4f}" for i in range(len(class_counts))]))

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=cfg.LABEL_SMOOTHING)

# ── Prepare with Accelerate (handles DDP, AMP, device placement) ──
model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, test_loader, scheduler
)

# Model summary
trainable_total = sum(p.numel() for g in param_groups for p in g["params"])
if accelerator.is_main_process:
    print(f"\n{'='*60}")
    print("MODEL SUMMARY -- G-CMAB Safe Core (Script 19)")
    print(f"{'='*60}")
    print(f"  Trainable params:    {trainable_total:,}")
    print(f"  d_model:             {cfg.CROSS_ATTN_D_MODEL}")
    print(f"  Backbone LR:         {cfg.BACKBONE_LR}")
    print(f"  Head LR:             {cfg.HEAD_LR}")
    print(f"  Flag 1 Strided Conv: {cfg.USE_STRIDED_CONV}")
    print(f"  Flag 2 GroupNorm:    {cfg.USE_GROUPNORM}")
    print(f"  Flag 3 LayerAttn:   {cfg.USE_LAYER_ATTN} (N={cfg.LAYER_ATTN_N}, LR={cfg.LAYER_ATTN_LR})")
    print(f"  Flag 4 MultiPool:   {cfg.USE_MULTI_POOL} (kmax_k={cfg.KMAX_K})")
    print(f"  Effective batch:     {cfg.BATCH_SIZE}×{accelerator.num_processes} = {cfg.BATCH_SIZE * accelerator.num_processes}")
    print(f"  Grad accum steps:    {cfg.GRAD_ACCUM_STEPS}")
    print(f"  Total / warmup:      {total_steps} / {warmup_steps}")

# ═══════════════════════════════════════════════════════════════════════
# CELL 10: Training Loop with Early Stopping + Gradient Monitoring
# ═══════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, scheduler, criterion, accelerator_obj,
                    grad_accum_steps, max_grad_norm=1.0, epoch_num=0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    grad_norms = []

    pbar = tqdm(loader, desc="  Training", leave=False, disable=not accelerator_obj.is_main_process)
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        shape_features = batch["shape_features"]
        labels = batch["labels"]

        with accelerator_obj.accumulate(model):
            with accelerator_obj.autocast():
                logits = model(input_ids, attention_mask, shape_features)
                loss = criterion(logits, labels)

            accelerator_obj.backward(loss)

            if accelerator_obj.sync_gradients:
                total_norm = accelerator_obj.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                if isinstance(total_norm, torch.Tensor):
                    grad_norms.append(total_norm.item())
                else:
                    grad_norms.append(float(total_norm))

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        running_loss += loss.item() * labels.size(0)
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        if accelerator_obj.is_main_process:
            pbar.set_postfix({"loss": f"{running_loss/total:.4f}", "acc": f"{correct/total:.4f}"})

    # Print gradient diagnostics for first few epochs
    if epoch_num < 5 and grad_norms and accelerator_obj.is_main_process:
        avg_norm = np.mean(grad_norms)
        max_norm_val = np.max(grad_norms)
        print(f"  [Grad Monitor] avg_norm={avg_norm:.4f}, max_norm={max_norm_val:.4f}, steps={len(grad_norms)}")
        if avg_norm < 1e-6 or np.isnan(avg_norm):
            print("  ⚠️  WARNING: Gradients near zero or NaN! Model training might be unstable.")

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, accelerator_obj):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        shape_features = batch["shape_features"]
        labels = batch["labels"]
        with accelerator_obj.autocast():
            logits = model(input_ids, attention_mask, shape_features)
            loss = criterion(logits, labels)

        # Gather predictions across GPUs for correct accuracy
        preds = logits.argmax(dim=1)
        preds, labels_gathered = accelerator_obj.gather_for_metrics((preds, labels))
        loss_gathered = accelerator_obj.gather_for_metrics(loss.repeat(labels.size(0)))

        running_loss += loss_gathered.sum().item()
        total += labels_gathered.size(0)
        correct += preds.eq(labels_gathered).sum().item()

    return running_loss / max(total, 1), correct / max(total, 1)


def train_model(model, train_loader, test_loader, optimizer, scheduler,
                criterion, accelerator_obj, cfg):
    if accelerator_obj.is_main_process:
        print("\n" + "=" * 60)
        print("TRAINING -- G-CMAB Safe Core (Script 19)")
        print("=" * 60)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(cfg.EPOCHS):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, accelerator_obj,
            cfg.GRAD_ACCUM_STEPS, max_grad_norm=cfg.MAX_GRAD_NORM, epoch_num=epoch,
        )

        val_loss, val_acc = evaluate(model, test_loader, criterion, accelerator_obj)

        elapsed = time.time() - t0
        gap = train_acc - val_acc

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
        accelerator_obj.wait_for_everyone()

        backbone_lr = optimizer.param_groups[0]["lr"]
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

        if epoch >= 3 and max(history["val_acc"]) < 0.30:
            print("  ⚠️  WARNING: Val accuracy still near random chance after 3 epochs!")

        if val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save checkpoint only on the main process
            if accelerator_obj.is_main_process:
                unwrapped = accelerator_obj.unwrap_model(model)
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": unwrapped.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "flags": {
                        "USE_STRIDED_CONV": cfg.USE_STRIDED_CONV,
                        "USE_GROUPNORM": cfg.USE_GROUPNORM,
                        "USE_LAYER_ATTN": cfg.USE_LAYER_ATTN,
                        "USE_MULTI_POOL": cfg.USE_MULTI_POOL,
                    },
                }, os.path.join(cfg.MODEL_DIR, "best_gcmab_safe.pt"))
                print(f"  -> Saved best model (val_loss={val_loss:.4f}, val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1} (patience={cfg.PATIENCE})")
                break

        # Synchronize after checkpoint operations
        accelerator_obj.wait_for_everyone()

    if accelerator_obj.is_main_process:
        print(f"\nTraining complete. Best val_acc: {best_val_acc:.4f}")
    return history


history = train_model(model, train_loader, test_loader, optimizer, scheduler,
                      criterion, accelerator, cfg)

# ═══════════════════════════════════════════════════════════════════════
# CELL 11: Load Best Model & Full Evaluation (main process only)
# ═══════════════════════════════════════════════════════════════════════

best_path = os.path.join(cfg.MODEL_DIR, "best_gcmab_safe.pt")
checkpoint = torch.load(best_path, map_location=DEVICE)
unwrapped = accelerator.unwrap_model(model)
unwrapped.load_state_dict(checkpoint["model_state_dict"])
print(f"Loaded best model from epoch {checkpoint['epoch']} "
      f"(val_loss={checkpoint['val_loss']:.4f}, val_acc={checkpoint['val_acc']:.4f})")


@torch.no_grad()
def full_evaluation(model, test_loader, class_names, accelerator_obj):
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    for batch in tqdm(test_loader, desc="  Evaluating", disable=not accelerator_obj.is_main_process):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        shape_features = batch["shape_features"]
        labels = batch["labels"]
        with accelerator_obj.autocast():
            logits = model(input_ids, attention_mask, shape_features)
        probs = torch.softmax(logits.float(), dim=1)
        preds = logits.argmax(dim=1)

        preds_g, labels_g, probs_g = accelerator_obj.gather_for_metrics((preds, labels, probs))
        all_preds.extend(preds_g.cpu().numpy())
        all_targets.extend(labels_g.cpu().numpy())
        all_probs.extend(probs_g.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    if accelerator_obj.is_main_process:
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        report_str = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
        print(report_str)

        report_path = os.path.join(cfg.OUTPUT_DIR, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write("CLASSIFICATION REPORT -- G-CMAB Safe Core (Script 19)\n")
            f.write("=" * 60 + "\n")
            f.write(report_str)
        print(f"  -> Report saved: {report_path}")

        acc = accuracy_score(all_targets, all_preds)
        f1_macro = f1_score(all_targets, all_preds, average="macro")
        print(f"Overall Accuracy:  {acc:.4f}")
        print(f"Macro F1-Score:    {f1_macro:.4f}")

    return all_preds, all_targets, all_probs


all_preds, all_targets, all_probs = full_evaluation(model, test_loader, cfg.CLASS_NAMES, accelerator)

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


if accelerator.is_main_process:
    print("\n  BED files for IGV:")
    export_igv_bed_files(headers_test, all_preds, all_targets, cfg.OUTPUT_DIR)

# ═══════════════════════════════════════════════════════════════════════
# CELL 12: Generate All Performance Figures (main process only)
# ═══════════════════════════════════════════════════════════════════════

TITLE_PREFIX = "G-CMAB Safe Core (DNABERT-2 + DNAshape)"

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
    path = os.path.join(save_dir, "gcmab_training_curves.png")
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
    path = os.path.join(save_dir, "gcmab_confusion_matrix.png")
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
    path = os.path.join(save_dir, "gcmab_roc_curves.png")
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
    path = os.path.join(save_dir, "gcmab_pr_curves.png")
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
    path = os.path.join(save_dir, "gcmab_per_class_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
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
# CELL 13: Summary
# ═══════════════════════════════════════════════════════════════════════

if accelerator.is_main_process:
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY -- G-CMAB Safe Core (Script 19)")
    print("=" * 60)
    print(f"  DNABERT-2:          Last {cfg.UNFREEZE_LAST_N_LAYERS} layers fine-tuned")
    print(f"  Max token length:   {MAX_LENGTH}")
    print(f"  d_model:            {cfg.CROSS_ATTN_D_MODEL}")
    print(f"  Cross-Attention:    {cfg.CROSS_ATTN_LAYERS} layer(s), {cfg.CROSS_ATTN_NHEAD} heads")
    print(f"  Loss:               {cfg.LOSS_TYPE}")
    print(f"  ── Script 19 Flags ──")
    print(f"  Strided Conv:       {cfg.USE_STRIDED_CONV}")
    print(f"  GroupNorm:          {cfg.USE_GROUPNORM} (groups={cfg.GROUPNORM_GROUPS})")
    print(f"  Layer-Attention:    {cfg.USE_LAYER_ATTN} (N={cfg.LAYER_ATTN_N})")
    print(f"  Multi-Pool:         {cfg.USE_MULTI_POOL} (seq=mean‖max, shape=kmax(k={cfg.KMAX_K}))")
    print(f"  Accelerate GPUs:    {accelerator.num_processes}")
    print(f"  Best val acc:       {max(history['val_acc']):.4f}")
    print(f"  Model saved to:     {cfg.MODEL_DIR}")
    print()
    print("  +-----------------------------------------------------------+")
    print("  |  Script 14 (Cross-Modal):     61.33%  (overfit, gap 31%)  |")
    print("  |  Script 15 (KAN, 11 changes): 53.43%  (over-regularized)  |")
    print("  |  Script 16 (Unified):         25%     (BROKEN: no learn)  |")
    print("  |  Script 17 (v2 Bug Fix):      34.21%  (EMA lag & pos_emb) |")
    print("  |  Script 18 (Simplified Core): 60.81%  (gap 25.57%)        |")
    print(f"  |  Script 19 (G-CMAB Safe):    {max(history['val_acc']):.2%}                      |")
    print("  +-----------------------------------------------------------+")
    print("=" * 60)

# ═══════════════════════════════════════════════════════════════════════
# CELL 13.5: SHAP Interpretability Analysis (FAD Style)
# ═══════════════════════════════════════════════════════════════════════

if accelerator.is_main_process:
    print("\n" + "=" * 60)
    print("RUNNING SHAP INTERPRETABILITY ANALYSIS")
    print("=" * 60)
    try:
        import shap
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import re
        
        # Run predictions on test loader (main process only)
        model.eval()
        all_preds = []
        all_targets = []
        device = accelerator.device
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                shape_features = batch["shape_features"].to(device)
                labels = batch["labels"].to(device)
                
                logits = model(input_ids, attention_mask, shape_features)
                preds = logits.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Filter indices for Subset A and Subset B
        # Class index 2 represents "SP4"
        sp4_class_idx = 2
        
        subset_A_indices = []  # Correct SP4 predictions
        subset_B_indices = []  # Confused SP4 (predicted as SP1 or SP2)
        
        for idx in range(len(all_targets)):
            if all_targets[idx] == sp4_class_idx:
                pred = all_preds[idx]
                if pred == sp4_class_idx:
                    subset_A_indices.append(idx)
                elif pred in [0, 1]:  # SP1 (0) or SP2 (1)
                    subset_B_indices.append(idx)
                    
        print(f"  Subset A (Correct SP4): {len(subset_A_indices)} samples")
        print(f"  Subset B (Confused SP4): {len(subset_B_indices)} samples")
        
        if len(subset_A_indices) > 0 and len(subset_B_indices) > 0:
            # Extract background dataset from test set (bg_size = 50)
            bg_size = 50
            np.random.seed(cfg.RANDOM_SEED)
            bg_indices = np.random.choice(len(test_dataset), min(bg_size, len(test_dataset)), replace=False)
            
            bg_input_ids = torch.stack([test_dataset[i]["input_ids"] for i in bg_indices]).to(device)
            bg_attention_mask = torch.stack([test_dataset[i]["attention_mask"] for i in bg_indices]).to(device)
            bg_shapes = torch.stack([test_dataset[i]["shape_features"] for i in bg_indices]).to(device)
            
            unwrapped_model = accelerator.unwrap_model(model)
            with torch.no_grad():
                bg_embeddings = unwrapped_model._get_bert_features(bg_input_ids, bg_attention_mask)
                
            # Define wrapper model that accepts differentiable embeddings and shapes
            class ModelEmbeddingWrapper(nn.Module):
                def __init__(self, gcmab_model):
                    super().__init__()
                    self.gcmab_model = gcmab_model
                    
                def forward(self, seq_embeddings, shape_features):
                    B, T, _ = seq_embeddings.shape
                    device = seq_embeddings.device
                    attention_mask = torch.ones(B, T, dtype=torch.long, device=device)
                    
                    seq_features = self.gcmab_model.seq_projection(seq_embeddings)
                    shape_feats = self.gcmab_model.shape_cnn(shape_features)
                    
                    seq_key_padding_mask = (attention_mask == 0)
                    for cross_layer in self.gcmab_model.cross_attention_layers:
                        seq_features, shape_feats = cross_layer(
                            seq_features, shape_feats, seq_key_padding_mask=seq_key_padding_mask
                        )
                        
                    if self.gcmab_model.use_multi_pool:
                        seq_mean = self.gcmab_model._masked_mean_pool(seq_features, attention_mask)
                        seq_max = self.gcmab_model._masked_max_pool(seq_features, attention_mask)
                        seq_pooled = torch.cat([seq_mean, seq_max], dim=1)
                        shape_pooled = self.gcmab_model._kmax_pool(shape_feats, k=self.gcmab_model.kmax_k)
                    else:
                        mask_expanded = attention_mask.unsqueeze(-1).float()
                        seq_pooled = (seq_features * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
                        shape_pooled = shape_feats.mean(dim=1)
                        
                    fused = torch.cat([seq_pooled, shape_pooled], dim=1)
                    logits = self.gcmab_model.classifier(fused)
                    return logits
                    
            wrapper_model = ModelEmbeddingWrapper(unwrapped_model)
            
            # Select test samples to explain (up to 20 per subset)
            num_explain = 20
            explain_A_indices = subset_A_indices[:num_explain]
            explain_B_indices = subset_B_indices[:num_explain]
            
            # Extract inputs to explain
            test_A_input_ids = torch.stack([test_dataset[i]["input_ids"] for i in explain_A_indices]).to(device)
            test_A_attention_mask = torch.stack([test_dataset[i]["attention_mask"] for i in explain_A_indices]).to(device)
            test_A_shapes = torch.stack([test_dataset[i]["shape_features"] for i in explain_A_indices]).to(device)
            
            test_B_input_ids = torch.stack([test_dataset[i]["input_ids"] for i in explain_B_indices]).to(device)
            test_B_attention_mask = torch.stack([test_dataset[i]["attention_mask"] for i in explain_B_indices]).to(device)
            test_B_shapes = torch.stack([test_dataset[i]["shape_features"] for i in explain_B_indices]).to(device)
            
            with torch.no_grad():
                test_A_embeddings = unwrapped_model._get_bert_features(test_A_input_ids, test_A_attention_mask)
                test_B_embeddings = unwrapped_model._get_bert_features(test_B_input_ids, test_B_attention_mask)
                
            # Run GradientExplainer
            print("  Running GradientExplainer on differentiable branches...")
            explainer = shap.GradientExplainer(wrapper_model, [bg_embeddings, bg_shapes])
            
            shap_values_A = explainer.shap_values([test_A_embeddings, test_A_shapes])
            shap_values_B = explainer.shap_values([test_B_embeddings, test_B_shapes])
            
            # Extract SHAP values for target class SP4 (index 2)
            seq_shap_A = shap_values_A[0][..., sp4_class_idx]    # [N, T, 768]
            shape_shap_A = shap_values_A[1][..., sp4_class_idx]  # [N, 5, 101]
            
            seq_shap_B = shap_values_B[0][..., sp4_class_idx]    # [N, T, 768]
            shape_shap_B = shap_values_B[1][..., sp4_class_idx]  # [N, 5, 101]
            
            # ──────────────────────────────────────────────────────────
            # Goal A: DNAshape Feature Importance Bar Chart
            # ──────────────────────────────────────────────────────────
            importance_shape_A = np.abs(shape_shap_A).sum(axis=2).mean(axis=0)
            importance_shape_B = np.abs(shape_shap_B).sum(axis=2).mean(axis=0)
            
            features = ["MGW", "ProT", "Roll", "HelT", "EP"]
            x = np.arange(len(features))
            width = 0.35
            
            plt.figure(figsize=(8, 5))
            plt.bar(x - width/2, importance_shape_A, width, label="Subset A (Correct SP4)", color="#4CAF50")
            plt.bar(x + width/2, importance_shape_B, width, label="Subset B (Confused)", color="#F44336")
            plt.xticks(x, features)
            plt.ylabel("Mean Absolute SHAP Value")
            plt.title("DNAshape Feature Importance (Subset A vs Subset B)")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout()
            
            shape_bar_path = os.path.join(cfg.FIG_DIR, "gcmab_shap_dnashape_bar.png")
            plt.savefig(shape_bar_path, dpi=150)
            plt.close()
            print(f"  -> DNAshape SHAP bar chart saved to: {shape_bar_path}")
            
            # ──────────────────────────────────────────────────────────
            # Goal B: Sequence Context Heatmap with RegEx Alignment
            # ──────────────────────────────────────────────────────────
            def get_char_shap_for_batch(explain_indices, seq_shap):
                char_shaps = []
                for idx_in_batch, original_idx in enumerate(explain_indices):
                    seq = seq_test[original_idx]
                    tokens = tokenizer.convert_ids_to_tokens(test_dataset[original_idx]["input_ids"])
                    token_importance = np.linalg.norm(seq_shap[idx_in_batch], axis=1)
                    
                    char_shap = np.zeros(len(seq))
                    char_counts = np.zeros(len(seq))
                    
                    current_char_idx = 0
                    for t_idx, token in enumerate(tokens):
                        if token in ["[CLS]", "[SEP]", "[PAD]", "<pad>", "<s>", "</s>", "<unk>"]:
                            continue
                        clean_token = token.replace("##", "").replace("Ġ", "").replace(" ", "")
                        if not clean_token:
                            continue
                        
                        pos = seq.find(clean_token, current_char_idx)
                        if pos != -1:
                            length = len(clean_token)
                            char_shap[pos:pos+length] += token_importance[t_idx]
                            char_counts[pos:pos+length] += 1
                            current_char_idx = pos + length
                            
                    char_shap = np.where(char_counts > 0, char_shap / char_counts, 0.0)
                    char_shaps.append(char_shap)
                return char_shaps
                
            char_shap_A = get_char_shap_for_batch(explain_A_indices, seq_shap_A)
            char_shap_B = get_char_shap_for_batch(explain_B_indices, seq_shap_B)
            
            seq_A_list = [seq_test[i] for i in explain_A_indices]
            seq_B_list = [seq_test[i] for i in explain_B_indices]
            
            def align_and_average_shap(seq_list, char_shaps_list, window_size=15):
                aligned_list = []
                for seq, char_s in zip(seq_list, char_shaps_list):
                    match = re.search(r'GGGCGG|CCGCCC', seq, re.IGNORECASE)
                    if not match:
                        continue
                    start, end = match.span()
                    center = (start + end) // 2
                    
                    aligned = np.zeros(2 * window_size + 1)
                    for i, offset in enumerate(range(-window_size, window_size + 1)):
                        abs_pos = center + offset
                        if 0 <= abs_pos < len(char_s):
                            aligned[i] = char_s[abs_pos]
                    aligned_list.append(aligned)
                if len(aligned_list) == 0:
                    return None
                return np.mean(aligned_list, axis=0)
                
            window_size = 15
            avg_aligned_A = align_and_average_shap(seq_A_list, char_shap_A, window_size)
            avg_aligned_B = align_and_average_shap(seq_B_list, char_shap_B, window_size)
            
            if avg_aligned_A is not None and avg_aligned_B is not None:
                heatmap_data = np.vstack([avg_aligned_A, avg_aligned_B])
                
                plt.figure(figsize=(12, 3))
                x_labels = [str(x) for x in range(-window_size, window_size + 1)]
                sns.heatmap(heatmap_data, annot=False, cmap="viridis",
                            yticklabels=["Subset A (Correct)", "Subset B (Confused)"],
                            xticklabels=x_labels)
                plt.xlabel("Position Relative to GC-box Center (bp)")
                plt.title("Sequence Context SHAP Importance Alignment (GC-box Flank Analysis)")
                plt.tight_layout()
                
                heatmap_path = os.path.join(cfg.FIG_DIR, "gcmab_shap_sequence_heatmap.png")
                plt.savefig(heatmap_path, dpi=150)
                plt.close()
                print(f"  -> Sequence SHAP heatmap saved to: {heatmap_path}")
                
                plt.figure(figsize=(10, 4))
                offsets = np.arange(-window_size, window_size + 1)
                plt.plot(offsets, avg_aligned_A, label="Subset A (Correct)", color="#4CAF50", linewidth=2)
                plt.plot(offsets, avg_aligned_B, label="Subset B (Confused)", color="#F44336", linewidth=2)
                plt.axvline(x=0, color="gray", linestyle="--", alpha=0.7, label="GC-box Center")
                plt.xlabel("Position Relative to GC-box Center (bp)")
                plt.ylabel("Average SHAP Importance (L2 Norm)")
                plt.title("Sequence Context SHAP Importance (Flanking Regions)")
                plt.legend()
                plt.grid(True, linestyle="--", alpha=0.3)
                plt.tight_layout()
                
                line_path = os.path.join(cfg.FIG_DIR, "gcmab_shap_sequence_line.png")
                plt.savefig(line_path, dpi=150)
                plt.close()
                print(f"  -> Sequence SHAP line chart saved to: {line_path}")
            else:
                print("  ⚠️ WARNING: Could not find any GC-box consensus motif in explains.")
        else:
            print("  ⚠️ WARNING: Subset A or Subset B is empty!")
            
    except Exception as shap_err:
        print(f"  ⚠️ Error running SHAP analysis: {shap_err}")
        import traceback
        traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════════
# CELL 14: Zip Outputs for Easy Kaggle Download
# ═══════════════════════════════════════════════════════════════════════

if accelerator.is_main_process:
    import shutil
    try:
        from IPython.display import FileLink, display as ipy_display
    except ImportError:
        FileLink = None
        ipy_display = None

    zip_filename = "outputs_gcmab_safe"
    shutil.make_archive(zip_filename, 'zip', cfg.OUTPUT_DIR)
    print(f"\nAll outputs zipped into: {zip_filename}.zip")

    if FileLink and ipy_display:
        try:
            ipy_display(FileLink(f"{zip_filename}.zip"))
        except Exception:
            print(f"Download {zip_filename}.zip from the Kaggle output panel.")
    else:
        print(f"Download {zip_filename}.zip from the Kaggle output panel.")
