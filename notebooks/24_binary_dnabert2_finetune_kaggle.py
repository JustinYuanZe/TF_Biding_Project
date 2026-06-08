#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  DNABERT-2 Fine-Tuning — End-to-End Training Script                ║
║  Designed for: Kaggle GPU (T4/P100) or Google Colab                ║
║  Task: Binary SP_Positive/Negative TF-binding classification     ║
╚══════════════════════════════════════════════════════════════════════╝

IMPROVEMENT over Script 09 (frozen DNABERT-2 + mCNN):
  Script 09 froze DNABERT-2 entirely and used a heavy mCNN classifier
  (2.4M params), which resulted in severe overfitting (train 99.98%
  but val stuck at ~40%). The root cause: frozen embeddings couldn't
  capture the subtle motif differences between SP1/SP2/SP4 (same
  Sp/KLF family, nearly identical GC-rich binding motifs).

  This script fixes the problem by:
  1. UNFREEZING the last 3 encoder layers of DNABERT-2, allowing the
     Transformer attention to learn task-specific motif discrimination.
  2. Using a SIMPLE classifier head ([CLS] + MeanPool → Linear) with
     only ~100K params instead of 2.4M, dramatically reducing overfitting.
  3. DIFFERENTIAL learning rates: backbone (2e-5) vs head (1e-3).
  4. Linear warmup + cosine annealing scheduler for stable fine-tuning.
  5. Label smoothing (0.1) for better generalization.
  6. Gradient accumulation (4 steps) to achieve effective batch_size=64
     while only using batch_size=16 on GPU (T4-friendly).

Architecture:
  DNABERT-2 (zhihan1996/DNABERT-2-117M)
    ├── Layers 0-8:  FROZEN (general DNA language knowledge)
    └── Layers 9-11: UNFROZEN (fine-tuned for SP binding discrimination)
  ↓
  Pooling: concat([CLS] token, MeanPool over all tokens) → 1536-dim
  ↓
  Classifier Head:
    ├── LayerNorm(1536)
    ├── Linear(1536 → 256) + GELU + Dropout(0.3)
    └── Linear(256 → 4)

Usage on Kaggle:
  1. Upload data/processed/*.fasta files as a Kaggle Dataset
  2. Add this script as a Kaggle Notebook
  3. Enable GPU accelerator (T4 recommended)
  4. Run all cells

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

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()

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

def auto_detect_dir(target_file, fallback="data/processed"):
    """Search for the directory containing target_file in Kaggle input or local path."""
    if os.path.exists(fallback) and os.path.exists(os.path.join(fallback, target_file)):
        return fallback
    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if target_file in files:
                print(f"  [Auto-detect] Found {target_file} at {root}")
                return root
    return fallback

class Config:
    """Central configuration for the fine-tuning pipeline."""

    # ── Paths ──
    DATA_DIR = auto_detect_dir("sp1_positive_final.fasta", "data/processed")
    OUTPUT_DIR = "outputs_finetune"
    FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

    # ── DNABERT-2 ──
    DNABERT_MODEL = "zhihan1996/DNABERT-2-117M"
    EMBEDDING_DIM = 768
    MAX_TOKEN_LENGTH = 512

    # ── Fine-Tuning Strategy ──
    UNFREEZE_LAST_N_LAYERS = 3    # Unfreeze last 3 of 12 encoder layers
    BACKBONE_LR = 1.5e-5          # Reduced slightly from 2e-5
    HEAD_LR = 1e-4                # Reduced from 1e-3 to 1e-4 (slower head learning)
    WEIGHT_DECAY = 0.1            # Increased from 0.01 to 0.1 (stronger regularization)

    # ── Classifier Head ──
    HIDDEN_DIM = 256
    DROPOUT_RATE = 0.3
    LABEL_SMOOTHING = 0.1
    NUM_CLASSES = 1

    # ── Training ──
    BATCH_SIZE = 16               # Per-GPU batch size (T4-friendly)
    GRAD_ACCUM_STEPS = 4          # Effective batch = 16 * 4 = 64
    EPOCHS = 15
    PATIENCE = 5                  # Early stopping patience
    MAX_OVERFITTING_GAP = 30.0  # Max train-val gap (%) to prevent severe overfitting
    WARMUP_RATIO = 0.1            # 10% of total steps for warmup

    # ── Data Split ──
    TEST_SIZE = 0.2
    RANDOM_SEED = 42

    # ── Class Names ──
    CLASS_NAMES = ["Negative", "SP_Positive"]

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
print(f"Fine-tuning strategy: Unfreeze last {cfg.UNFREEZE_LAST_N_LAYERS} layers")
print(f"Effective batch size: {cfg.BATCH_SIZE} × {cfg.GRAD_ACCUM_STEPS} = {cfg.BATCH_SIZE * cfg.GRAD_ACCUM_STEPS}")

# ═══════════════════════════════════════════════════════════════════════
# CELL 3: Data Loading (with Group-Aware Splitting)
# ═══════════════════════════════════════════════════════════════════════

def load_fasta(filepath):
    """Load DNA sequences from a FASTA file."""
    sequences = []
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing dataset: {filepath}")
    with open(filepath, "r") as f:
        seq_lines = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_lines:
                    sequences.append("".join(seq_lines).upper())
                    seq_lines = []
            else:
                seq_lines.append(line)
        if seq_lines:
            sequences.append("".join(seq_lines).upper())
    return sequences


def load_all_data(data_dir):
    """Load all 4 classes and build group-aware labels."""
    print("=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)

    # Detect negative FASTA file
    neg_fasta = "negative_final.fasta"
    fasta_candidates = ["negative_genomic_matched.fasta", "negative_promoter_cpg.fasta", "negative_final.fasta"]
    for cand in fasta_candidates:
        if os.path.exists(os.path.join(data_dir, cand)):
            neg_fasta = cand
            break

    print(f"  [Auto-detect] Using negative FASTA file: {neg_fasta}")

    files = {
        "SP1": os.path.join(data_dir, "sp1_positive_final.fasta"),
        "SP2": os.path.join(data_dir, "sp2_positive_final.fasta"),
        "SP4": os.path.join(data_dir, "sp4_positive_final.fasta"),
        "Negative": os.path.join(data_dir, neg_fasta),
    }

    all_sequences = []
    all_labels = []
    all_groups = []
    group_id = 0

    for cls_idx, (cls_name, fpath) in enumerate(files.items()):
        seqs = load_fasta(fpath)
        print(f"  {cls_name}: {len(seqs)} sequences")
        all_sequences.extend(seqs)
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

    print(f"\n  Total: {len(all_sequences)} sequences, {group_id} groups")
    print(f"  Class distribution: {np.bincount(all_labels)}")

    return all_sequences, all_labels, all_groups


def split_data(sequences, labels, groups, test_size=0.2, seed=42):
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

    print(f"  Train: {len(seq_train)} sequences")
    print(f"  Test:  {len(seq_test)} sequences")
    print(f"  Train class dist: {np.bincount(y_train)}")
    print(f"  Test  class dist: {np.bincount(y_test)}")

    return seq_train, seq_test, y_train, y_test


# Load and split
all_sequences, all_labels, all_groups = load_all_data(cfg.DATA_DIR)
seq_train, seq_test, y_train, y_test = split_data(
    all_sequences, all_labels, all_groups,
    test_size=cfg.TEST_SIZE, seed=cfg.RANDOM_SEED,
)

# Convert multiclass labels SP1(0), SP2(1), SP4(2) -> Positive(1), and Negative(3) -> Negative(0)
y_train = np.array([1 if l < 3 else 0 for l in y_train])
y_test = np.array([1 if l < 3 else 0 for l in y_test])

# ═══════════════════════════════════════════════════════════════════════
# CELL 4: DNABERT-2 Flash Attention Patch (Pure PyTorch, No Triton)
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
# CELL 5: Load DNABERT-2 Backbone (with Selective Unfreezing)
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
# CELL 6: Classifier Head Definition
# ═══════════════════════════════════════════════════════════════════════

class DNABERT2Classifier(nn.Module):
    """
    End-to-end DNABERT-2 + simple classifier for fine-tuning.

    Pooling strategy: Concatenate [CLS] token embedding with mean-pooled
    token embeddings → 2 × 768 = 1536-dimensional feature vector.
    This captures both the global [CLS] summary and the average signal
    across all positions in the sequence.

    Classifier: LayerNorm → Linear(1536→256) → GELU → Dropout → Linear(256→4)
    Only ~400K parameters in the head (vs 2.4M in mCNN).
    """
    def __init__(self, backbone, embedding_dim=768, hidden_dim=256,
                 num_classes=4, dropout_rate=0.3):
        super().__init__()
        self.backbone = backbone
        pooled_dim = embedding_dim * 2  # [CLS] + MeanPool concatenation

        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]  # (batch, seq_len, 768)

        # [CLS] token (first token)
        cls_token = hidden_states[:, 0, :]  # (batch, 768)

        # Mean pooling (excluding padding tokens)
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)  # (batch, 768)
        count = mask_expanded.sum(dim=1).clamp(min=1e-9)  # (batch, 1)
        mean_pooled = sum_hidden / count  # (batch, 768)

        # Concatenate [CLS] + MeanPool
        pooled = torch.cat([cls_token, mean_pooled], dim=1)  # (batch, 1536)

        logits = self.classifier(pooled)
        return logits.squeeze(-1)

# ═══════════════════════════════════════════════════════════════════════
# CELL 7: PyTorch Dataset & DataLoaders
# ═══════════════════════════════════════════════════════════════════════

class DNASequenceDataset(Dataset):
    """Dataset that tokenizes DNA sequences on-the-fly."""
    def __init__(self, sequences, labels, tokenizer, max_length=512):
        self.sequences = sequences
        self.labels = labels
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
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


train_dataset = DNASequenceDataset(seq_train, y_train, tokenizer, cfg.MAX_TOKEN_LENGTH)
test_dataset = DNASequenceDataset(seq_test, y_test, tokenizer, cfg.MAX_TOKEN_LENGTH)

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
# CELL 8: Build Model & Optimizer with Differential LR
# ═══════════════════════════════════════════════════════════════════════

model = DNABERT2Classifier(
    backbone=dnabert_model,
    embedding_dim=cfg.EMBEDDING_DIM,
    hidden_dim=cfg.HIDDEN_DIM,
    num_classes=cfg.NUM_CLASSES,
    dropout_rate=cfg.DROPOUT_RATE,
)

# Separate parameter groups for differential learning rates
backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
head_params = list(model.classifier.parameters())

optimizer = optim.AdamW([
    {"params": backbone_params, "lr": cfg.BACKBONE_LR, "weight_decay": cfg.WEIGHT_DECAY},
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

# Pos weight for BCEWithLogitsLoss
y_train_arr = np.array(y_train)
pos_count = np.sum(y_train_arr == 1)
neg_count = np.sum(y_train_arr == 0)
pos_weight_val = float(neg_count) / float(pos_count) if pos_count > 0 else 1.0
pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=DEVICE)
print(f"  BCE pos_weight: {pos_weight_val:.4f} (Negative: {neg_count}, Positive: {pos_count})")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, test_loader, scheduler
)

# Print model summary
backbone_trainable = sum(p.numel() for p in backbone_params)
head_trainable = sum(p.numel() for p in head_params)
total_trainable = backbone_trainable + head_trainable

print(f"\n{'='*60}")
print("MODEL SUMMARY")
print(f"{'='*60}")
print(f"  Backbone trainable params: {backbone_trainable:>10,} (LR={cfg.BACKBONE_LR})")
print(f"  Head trainable params:     {head_trainable:>10,} (LR={cfg.HEAD_LR})")
print(f"  Total trainable params:    {total_trainable:>10,}")
print(f"  Total training steps:      {total_steps}")
print(f"  Warmup steps:              {warmup_steps}")
print(f"  Label smoothing:           {cfg.LABEL_SMOOTHING}")

# ═══════════════════════════════════════════════════════════════════════
# CELL 9: Training Loop with Gradient Accumulation & Early Stopping
# ═══════════════════════════════════════════════════════════════════════

def train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device):
    """Train for one epoch with gradient accumulation."""
    model.train()
    train_loss_sum, train_total = 0.0, 0
    all_train_preds, all_train_targets = [], []

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with accelerator.accumulate(model):
            with accelerator.autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels.float())

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Gather metrics
        predicted = (logits > 0).long()
        preds_gathered, targets_gathered = accelerator.gather_for_metrics((predicted, labels))
        loss_gathered = accelerator.gather(loss).mean().item()

        train_loss_sum += loss_gathered * targets_gathered.size(0)
        train_total += targets_gathered.size(0)
        all_train_preds.extend(preds_gathered.cpu().numpy())
        all_train_targets.extend(targets_gathered.cpu().numpy())

    train_loss = train_loss_sum / train_total
    train_acc = accuracy_score(all_train_targets, all_train_preds)
    return train_loss, train_acc


@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    """Evaluate on validation/test set."""
    model.eval()
    val_loss_sum, val_total = 0.0, 0
    all_val_preds, all_val_targets = [], []

    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with accelerator.autocast():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels.float())

        # Gather metrics
        predicted = (logits > 0).long()
        preds_gathered, targets_gathered = accelerator.gather_for_metrics((predicted, labels))
        loss_gathered = accelerator.gather(loss).mean().item()

        val_loss_sum += loss_gathered * targets_gathered.size(0)
        val_total += targets_gathered.size(0)
        all_val_preds.extend(preds_gathered.cpu().numpy())
        all_val_targets.extend(targets_gathered.cpu().numpy())

    val_loss = val_loss_sum / val_total
    val_acc = accuracy_score(all_val_targets, all_val_preds)
    return val_loss, val_acc


def train_model(model, train_loader, test_loader, optimizer, scheduler,
                criterion, cfg, device):
    """Full training loop with early stopping."""
    print("\n" + "=" * 60)
    print("TRAINING — End-to-End Fine-Tuning")
    print("=" * 60)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(cfg.EPOCHS):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
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

        # Checkpoint based on val_acc (using the requested val_acc-based early stopping logic)
        accelerator.wait_for_everyone()
        if val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            ckpt_path = os.path.join(cfg.MODEL_DIR, "best_dnabert2_finetune.pt")
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": unwrapped.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }, ckpt_path)
                print(f"  → Saved best model (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                print(f"\n  ⏹ Early stopping at epoch {epoch+1} (patience={cfg.PATIENCE})")
                break
        accelerator.wait_for_everyone()

    print(f"\nTraining complete. Best val_acc: {best_val_acc:.4f}")
    return history


history = train_model(
    model, train_loader, test_loader, optimizer, scheduler,
    criterion, cfg, DEVICE,
)

# ═══════════════════════════════════════════════════════════════════════
# CELL 10: Load Best Model & Full Evaluation
# ═══════════════════════════════════════════════════════════════════════

best_path = os.path.join(cfg.MODEL_DIR, "best_dnabert2_finetune.pt")
checkpoint = torch.load(best_path, map_location=DEVICE)
unwrapped = accelerator.unwrap_model(model)
unwrapped.load_state_dict(checkpoint["model_state_dict"])
model = unwrapped
model.eval()
print(f"Loaded best model from epoch {checkpoint['epoch']} "
      f"(val_loss={checkpoint['val_loss']:.4f}, val_acc={checkpoint['val_acc']:.4f})")


@torch.no_grad()
def full_evaluation(model, test_loader, class_names, device):
    """Run full evaluation and return predictions/probabilities."""
    all_preds, all_targets, all_probs = [], [], []

    for batch in tqdm(test_loader, desc="  Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with accelerator.autocast():
            logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits.float())
        preds = (logits > 0).long()

        # Gather metrics across GPUs
        preds_gathered, targets_gathered = accelerator.gather_for_metrics((preds, labels))
        probs_gathered = accelerator.gather_for_metrics(probs)

        all_preds.extend(preds_gathered.cpu().numpy())
        all_targets.extend(targets_gathered.cpu().numpy())
        all_probs.extend(probs_gathered.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4))

    acc = accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average="binary")
    print(f"Overall Accuracy:  {acc:.4f}")
    print(f"Macro F1-Score:    {f1_macro:.4f}")

    return all_preds, all_targets, all_probs


all_preds, all_targets, all_probs = full_evaluation(model, test_loader, cfg.CLASS_NAMES, DEVICE)

# ═══════════════════════════════════════════════════════════════════════
# CELL 11: Generate All Performance Figures
# ═══════════════════════════════════════════════════════════════════════

TITLE_PREFIX = "DNABERT-2 Fine-Tuned"

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
    ax2.axhline(y=0.50, color="gray", linestyle="--", alpha=0.5, label="Random Guess")
    ax2.set_title("Accuracy Performance", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle(f"{TITLE_PREFIX} — Training Progress", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "finetune_training_curves.png")
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
    path = os.path.join(save_dir, "finetune_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Confusion matrix: {path}")


def plot_roc_curves(all_targets, all_probs, class_names, save_dir):
    """Plot ROC curves."""
    plt.figure(figsize=(9, 7))
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    if len(class_names) == 2:
        fpr, tpr, _ = roc_curve(all_targets, all_probs)
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, color="#2196F3", linewidth=2.5,
                 label=f"{class_names[1]} vs {class_names[0]} (AUC = {auc_val:.4f})")
    else:
        n_classes = len(class_names)
        y_bin = label_binarize(all_targets, classes=list(range(n_classes)))
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
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
    path = os.path.join(save_dir, "finetune_roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → ROC curves: {path}")


def plot_precision_recall_curves(all_targets, all_probs, class_names, save_dir):
    """Plot Precision-Recall curves."""
    plt.figure(figsize=(9, 7))
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    if len(class_names) == 2:
        precision, recall, _ = precision_recall_curve(all_targets, all_probs)
        ap = average_precision_score(all_targets, all_probs)
        plt.plot(recall, precision, color="#2196F3", linewidth=2.5,
                 label=f"{class_names[1]} vs {class_names[0]} (AP = {ap:.4f})")
        pos_ratio = np.sum(all_targets == 1) / len(all_targets) if len(all_targets) > 0 else 0.5
        plt.axhline(y=pos_ratio, color="gray", linestyle="--", alpha=0.5, label=f"Random Baseline ({pos_ratio:.2f})")
    else:
        n_classes = len(class_names)
        y_bin = label_binarize(all_targets, classes=list(range(n_classes)))
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
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
    path = os.path.join(save_dir, "finetune_pr_curves.png")
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
    path = os.path.join(save_dir, "finetune_per_class_metrics.png")
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
# CELL 12: Summary & Comparison with Script 09
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PIPELINE SUMMARY")
print("=" * 60)
print(f"  Model:             DNABERT-2 (last {cfg.UNFREEZE_LAST_N_LAYERS} layers fine-tuned)")
print(f"  Classifier:        [CLS]+MeanPool → Linear({cfg.HIDDEN_DIM}) → {cfg.NUM_CLASSES}")
print(f"  Tokenizer:         BPE (Byte Pair Encoding)")
print(f"  Attention:         ALiBi (patched to pure PyTorch)")
print(f"  Data split:        GroupShuffleSplit (revcomp-aware)")
print(f"  Training samples:  {len(seq_train)}")
print(f"  Test samples:      {len(seq_test)}")
print(f"  Backbone LR:       {cfg.BACKBONE_LR}")
print(f"  Head LR:           {cfg.HEAD_LR}")
print(f"  Label smoothing:   {cfg.LABEL_SMOOTHING}")
print(f"  Grad accumulation: {cfg.GRAD_ACCUM_STEPS} steps")
print(f"  Best val loss:     {min(history['val_loss']):.4f}")
print(f"  Best val acc:      {max(history['val_acc']):.4f}")
print(f"  Figures saved to:  {cfg.FIG_DIR}")
print(f"  Model saved to:    {cfg.MODEL_DIR}")
print()
print("  ┌─────────────────────────────────────────────────────┐")
print("  │  COMPARISON with Script 09 (Frozen + mCNN)         │")
print("  ├─────────────────────────────────────────────────────┤")
print("  │  Script 09: Frozen DNABERT-2 + mCNN (2.4M params)  │")
print("  │    → Train Acc: 99.98%, Val Acc: ~40% (OVERFIT)    │")
print("  │                                                     │")
print("  │  Script 10: Fine-Tuned DNABERT-2 + Simple Head     │")
print(f"  │    → Best Val Acc: {max(history['val_acc']):.2%}                      │")
print("  └─────────────────────────────────────────────────────┘")
print("=" * 60)
