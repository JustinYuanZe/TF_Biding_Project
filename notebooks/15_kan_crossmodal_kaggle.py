#!/usr/bin/env python3
"""
Auto-refactored script: 15_kan_crossmodal_kaggle.py
Refactored to align with project standards.
"""
"""
Script 15: KAN-Regularized Cross-Modal Attention Dual-Branch
Designed for: Kaggle GPU (T4/P100) or Google Colab
Task: 4-class SP1/SP2/SP4/Negative TF-binding classification
"""

# ═══════════════════════════════════════════════════════════════════════
# CELL 0: Install Dependencies
# ═══════════════════════════════════════════════════════════════════════

import subprocess
import sys
import os

# Suppress Hugging Face warnings and tokenizers parallelism messages
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def install_packages():
    packages = [
        "transformers>=4.37.0",
        "einops>=0.7.0",
        "datasets>=2.16.0",
        "accelerate>=0.25.0",
        "safetensors>=0.4.0",
    ]
    # Only run pip install on local rank 0 to avoid race conditions
    if os.environ.get("LOCAL_RANK", "0") == "0":
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
)
from sklearn.preprocessing import label_binarize
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True

# ═══════════════════════════════════════════════════════════════════════
# CELL 2: Configuration
# ═══════════════════════════════════════════════════════════════════════

def find_file(filename: str, fallback_dir: str = "data/processed") -> str | None:
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


def auto_detect_dir(target_file: str, fallback: str = "data/processed") -> str:
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

    # ── Fine-Tuning ──
    UNFREEZE_LAST_N_LAYERS = 3     # freeze more layers
    BACKBONE_LR = 5e-6             # conservative backbone LR

    # ── Cross-Modal Attention ──
    CROSS_ATTN_D_MODEL = 128
    CROSS_ATTN_NHEAD = 4
    CROSS_ATTN_LAYERS = 2
    CROSS_ATTN_DROPOUT = 0.15
    DROP_PATH_RATE = 0.1
    CROSS_ATTN_LR = 1.5e-4
    SEQ_PROJ_LR = 1.5e-4

    # ── DNAshape Branch ──
    SHAPE_CHANNELS = 5
    SHAPE_SEQ_LEN = 101
    SHAPE_CONV_CHANNELS = [32, 64, 128]
    SHAPE_LR = 2e-4

    # ── KAN Classifier ──
    KAN_HIDDEN_DIM = 64
    KAN_DEGREE = 4
    FUSION_DROPOUT = 0.5
    HEAD_LR = 1e-4
    ATTN_POOL_LR = 1.5e-4

    # ── Loss Function ──
    FOCAL_GAMMA = 2.0
    LABEL_SMOOTHING = 0.1
    NUM_CLASSES = 4
    WEIGHT_DECAY = 0.15

    # ── R-Drop Regularization ──
    RDROP_ALPHA = 1.0

    # ── Training ──
    BATCH_SIZE = 12
    GRAD_ACCUM_STEPS = 6
    EPOCHS = 30
    PATIENCE = 12
    MAX_OVERFITTING_GAP = 30.0  # Max train-val gap (%) to prevent severe overfitting
    WARMUP_RATIO = 0.1
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

cfg = Config()

for d in [cfg.OUTPUT_DIR, cfg.FIG_DIR, cfg.MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

random.seed(cfg.RANDOM_SEED)
np.random.seed(cfg.RANDOM_SEED)
torch.manual_seed(cfg.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.RANDOM_SEED)

# Initialize Accelerator
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=cfg.GRAD_ACCUM_STEPS,
    kwargs_handlers=[ddp_kwargs],
)
DEVICE = accelerator.device

if not accelerator.is_main_process:
    import builtins
    builtins.print = lambda *args, **kwargs: None

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
# CELL 6: Load DNABERT-2 Backbone (Selective Fine-Tuning)
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


class ChebyKANLinear(nn.Module):
    """Kolmogorov-Arnold Network layer using Chebyshev polynomial basis."""
    def __init__(self, in_features, out_features, degree=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(
            torch.randn(out_features, in_features, degree + 1)
            * (1.0 / math.sqrt(in_features * (degree + 1)))
        )
        self.base_linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        base_output = self.base_linear(x)
        x_norm = torch.tanh(x)
        T = [torch.ones_like(x_norm)]
        if self.degree >= 1:
            T.append(x_norm)
        for n in range(2, self.degree + 1):
            T.append(2.0 * x_norm * T[-1] - T[-2])

        cheby_basis = torch.stack(T, dim=-1)
        kan_output = torch.einsum('bid,oid->bo', cheby_basis, self.cheby_coeffs)
        return base_output + kan_output


class SpatialShapeCNN(nn.Module):
    """Conv1D on DNAshape features, preserving spatial dimension."""
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

        self.pos_embed = nn.Parameter(torch.randn(1, max_positions, d_model) * 0.02)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.transpose(1, 2)
        x = self.proj(x)
        L = x.size(1)
        x = x + self.pos_embed[:, :L, :]
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


class CrossModalAttentionLayer(nn.Module):
    """Bidirectional Cross-Modal Attention with DropPath."""
    def __init__(self, d_model=128, nhead=4, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.cross_attn_seq2shape = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm_seq = nn.LayerNorm(d_model)
        self.drop_path_seq = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ffn_seq = FeedForward(d_model, expansion=4, dropout=dropout, drop_path=drop_path)

        self.cross_attn_shape2seq = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm_shape = nn.LayerNorm(d_model)
        self.drop_path_shape = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ffn_shape = FeedForward(d_model, expansion=4, dropout=dropout, drop_path=drop_path)

    def forward(self, seq_features, shape_features, seq_key_padding_mask=None):
        attended_seq, _ = self.cross_attn_seq2shape(
            query=seq_features, key=shape_features, value=shape_features
        )
        seq_out = self.norm_seq(seq_features + self.drop_path_seq(attended_seq))
        seq_out = self.ffn_seq(seq_out)

        attended_shape, _ = self.cross_attn_shape2seq(
            query=shape_features, key=seq_features, value=seq_features,
            key_padding_mask=seq_key_padding_mask
        )
        shape_out = self.norm_shape(shape_features + self.drop_path_shape(attended_shape))
        shape_out = self.ffn_shape(shape_out)

        return seq_out, shape_out


class AttentionPooling(nn.Module):
    """Learnable query token that attends over sequence to produce a single vector."""
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


class CrossModalKANClassifier(nn.Module):
    """Full Model: CrossModalKANClassifier."""
    def __init__(self, backbone, embedding_dim=768,
                 shape_in_channels=5, shape_conv_channels=None,
                 d_model=128, nhead=4, num_cross_layers=2, cross_dropout=0.15,
                 drop_path_rate=0.1,
                 kan_hidden_dim=64, kan_degree=4, num_classes=4,
                 fusion_dropout=0.5):
        super().__init__()
        self.backbone = backbone
        self.d_model = d_model

        self.seq_projection = nn.Sequential(
            nn.Linear(embedding_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        if shape_conv_channels is None:
            shape_conv_channels = [32, 64, 128]
        self.shape_cnn = SpatialShapeCNN(
            in_channels=shape_in_channels,
            conv_channels=shape_conv_channels,
            d_model=d_model,
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_cross_layers)]
        self.cross_attention_layers = nn.ModuleList([
            CrossModalAttentionLayer(
                d_model=d_model, nhead=nhead,
                dropout=cross_dropout, drop_path=dpr[i]
            )
            for i in range(num_cross_layers)
        ])

        self.seq_attn_pool = AttentionPooling(d_model, nhead=nhead, dropout=cross_dropout)

        fusion_dim = d_model * 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            ChebyKANLinear(fusion_dim, kan_hidden_dim, degree=kan_degree),
            nn.LayerNorm(kan_hidden_dim),
            nn.Dropout(p=fusion_dropout),
            ChebyKANLinear(kan_hidden_dim, num_classes, degree=kan_degree),
        )

    def forward(self, input_ids, attention_mask, shape_features):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]
        seq_features = self.seq_projection(hidden_states)

        shape_feats = self.shape_cnn(shape_features)

        seq_key_padding_mask = (attention_mask == 0)
        for cross_layer in self.cross_attention_layers:
            seq_features, shape_feats = cross_layer(
                seq_features, shape_feats,
                seq_key_padding_mask=seq_key_padding_mask
            )

        seq_pooled = self.seq_attn_pool(seq_features, key_padding_mask=seq_key_padding_mask)
        shape_pooled = shape_feats.mean(dim=1)

        fused = torch.cat([seq_pooled, shape_pooled], dim=1)
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
# CELL 9: Build Model & Optimizer
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

# Prepare components
model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, test_loader, scheduler
)

# Class weights
class_counts = np.bincount(y_train)
class_weights = len(y_train) / (len(class_counts) * class_counts.astype(np.float32))
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
print(f"  Class weights: " + ", ".join([f"{cfg.CLASS_NAMES[i]}={class_weights[i]:.4f}" for i in range(len(class_counts))]))

# ── Focal Loss ──
class FocalLoss(nn.Module):
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

# ═══════════════════════════════════════════════════════════════════════
# CELL 10: Training Loop with R-Drop & Focal Loss
# ═══════════════════════════════════════════════════════════════════════

def compute_rdrop_kl(logits1, logits2):
    p1 = F.log_softmax(logits1, dim=-1)
    p2 = F.log_softmax(logits2, dim=-1)
    kl1 = F.kl_div(p1, p2.detach().exp(), reduction='batchmean')
    kl2 = F.kl_div(p2, p1.detach().exp(), reduction='batchmean')
    return (kl1 + kl2) / 2


def train_one_epoch(model, train_loader, optimizer, scheduler, criterion,
                    device, grad_accum_steps, rdrop_alpha=1.0, max_grad_norm=0.5):
    model.train()
    running_loss = 0.0
    running_ce = 0.0
    running_kl = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc="  Training", leave=False, disable=not accelerator.is_main_process)
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        shape_features = batch["shape_features"]
        labels = batch["labels"]

        with accelerator.accumulate(model):
            with accelerator.autocast():
                logits1 = model(input_ids, attention_mask, shape_features)
                logits2 = model(input_ids, attention_mask, shape_features)

                loss1 = criterion(logits1, labels)
                loss2 = criterion(logits2, labels)
                ce_loss = (loss1 + loss2) / 2

                kl_loss = compute_rdrop_kl(logits1, logits2)
                loss = ce_loss + rdrop_alpha * kl_loss

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Gather metrics across processes for accurate training log
        with torch.no_grad():
            t_loss = torch.tensor([loss.item() * grad_accum_steps], device=device)
            t_ce = torch.tensor([ce_loss.item()], device=device)
            t_kl = torch.tensor([kl_loss.item()], device=device)
            logits_g, labels_g, loss_g, ce_g, kl_g = accelerator.gather_for_metrics(
                (logits1, labels, t_loss, t_ce, t_kl)
            )

            running_loss += loss_g.mean().item() * labels_g.size(0)
            running_ce += ce_g.mean().item() * labels_g.size(0)
            running_kl += kl_g.mean().item() * labels_g.size(0)

            _, predicted = logits_g.max(1)
            total += labels_g.size(0)
            correct += predicted.eq(labels_g).sum().item()

        if accelerator.is_main_process:
            pbar.set_postfix({
                "loss": f"{running_loss/total:.4f}",
                "acc": f"{correct/total:.4f}",
                "kl": f"{running_kl/total:.4f}",
            })

    return running_loss / total, correct / total, running_ce / total, running_kl / total


@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    model.eval()
    all_logits = []
    all_labels = []
    for batch in test_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        shape_features = batch["shape_features"]
        labels = batch["labels"]

        with accelerator.autocast():
            logits = model(input_ids, attention_mask, shape_features)

        logits_gathered, labels_gathered = accelerator.gather_for_metrics((logits, labels))
        all_logits.append(logits_gathered.cpu())
        all_labels.append(labels_gathered.cpu())

    all_logits = torch.cat(all_logits, dim=0).to(device)
    all_labels = torch.cat(all_labels, dim=0).to(device)

    val_loss = criterion(all_logits, all_labels).item()
    preds = all_logits.argmax(dim=1)
    val_acc = (preds == all_labels).float().mean().item()
    return val_loss, val_acc


def train_model(model, train_loader, test_loader, optimizer, scheduler,
                criterion, cfg, device):
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
            device, cfg.GRAD_ACCUM_STEPS,
            rdrop_alpha=cfg.RDROP_ALPHA, max_grad_norm=cfg.MAX_GRAD_NORM,
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
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
        history["ce_loss"].append(ce_loss)
        history["kl_loss"].append(kl_loss)

        gap_percent = (train_acc - val_acc) * 100

        print(
            f"Epoch {epoch+1:02d}/{cfg.EPOCHS} | "
            f"Train: {train_loss:.4f}/{train_acc:.4f} | "
            f"Val: {val_loss:.4f}/{val_acc:.4f} | "
            f"Gap: {gap_percent:+.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.1e} | {elapsed:.0f}s"
        )

        if gap_percent >= cfg.MAX_OVERFITTING_GAP:
            print(f"\n  ⏹ Early stopping at epoch {epoch+1} due to severe overfitting (Gap: {gap_percent:+.2f}% >= {cfg.MAX_OVERFITTING_GAP}%)")
            break

        if val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": unwrapped.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }, os.path.join(cfg.MODEL_DIR, "best_kan_crossmodal.pt"))
                print(f"  -> Saved best (val_acc={val_acc:.4f}, gap={gap_percent/100:.2%})")
            accelerator.wait_for_everyone()
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1} (patience={cfg.PATIENCE})")
                break

    print(f"\nTraining complete. Best val_acc: {best_val_acc:.4f}")
    return history


history = train_model(model, train_loader, test_loader, optimizer, scheduler, criterion, cfg, DEVICE)

# ═══════════════════════════════════════════════════════════════════════
# CELL 11: Load Best Model & Full Evaluation
# ═══════════════════════════════════════════════════════════════════════

best_path = os.path.join(cfg.MODEL_DIR, "best_kan_crossmodal.pt")
checkpoint = torch.load(best_path, map_location=DEVICE)
unwrapped = accelerator.unwrap_model(model)
unwrapped.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Loaded best model from epoch {checkpoint['epoch']} "
      f"(val_loss={checkpoint['val_loss']:.4f}, val_acc={checkpoint['val_acc']:.4f})")


@torch.no_grad()
def full_evaluation(model, test_loader, class_names, device):
    all_preds, all_targets, all_probs = [], [], []
    for batch in tqdm(test_loader, desc="  Evaluating", disable=not accelerator.is_main_process):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        shape_features = batch["shape_features"]
        labels = batch["labels"]

        with accelerator.autocast():
            logits = model(input_ids, attention_mask, shape_features)
        probs = torch.softmax(logits.float(), dim=1)
        _, preds = logits.max(1)

        # Gather
        preds_g, labels_g, probs_g = accelerator.gather_for_metrics((preds, labels, probs))

        all_preds.extend(preds_g.cpu().numpy())
        all_targets.extend(labels_g.cpu().numpy())
        all_probs.extend(probs_g.cpu().numpy())

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

if accelerator.is_main_process:
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


if accelerator.is_main_process:
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
print(f"  DNABERT-2:         Last {cfg.UNFREEZE_LAST_N_LAYERS} layers")
print(f"  Cross-Attention:   {cfg.CROSS_ATTN_LAYERS} layers, DropPath={cfg.DROP_PATH_RATE}")
print(f"  Classifier:        ChebyKAN(degree={cfg.KAN_DEGREE}, hidden={cfg.KAN_HIDDEN_DIM})")
print(f"  Regularization:    R-Drop(a={cfg.RDROP_ALPHA}) + Focal(g={cfg.FOCAL_GAMMA}) + WD={cfg.WEIGHT_DECAY}")
print(f"  Best val_acc:      {checkpoint['val_acc']:.4f}")
print(f"  Overfitting gap:   {final_gap:.2%}")
print()
