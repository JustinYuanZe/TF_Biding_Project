#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  DNABERT-2 + Multi-Scale CNN (mCNN) — End-to-End Training Script   ║
║  Designed for: Kaggle GPU (T4/P100) or Google Colab                ║
║  Task: 4-class SP1/SP2/SP4/Negative TF-binding classification     ║
╚══════════════════════════════════════════════════════════════════════╝

Architecture:
  DNABERT-2 (zhihan1996/DNABERT-2-117M)
    ├── BPE Tokenizer (Byte Pair Encoding, NOT k-mer)
    ├── ALiBi Attention (Attention with Linear Biases)
    └── 117M parameters (frozen backbone → feature extractor)
  ↓
  mCNN Head (Multi-Scale Convolutional Neural Network)
    ├── Parallel Conv1D branches (kernel sizes 3, 5, 7, 9)
    ├── GlobalMaxPool per branch
    └── FC classification head (512 → 128 → 4)

Key Design Choices:
  1. GroupShuffleSplit: orig/revcomp pairs ALWAYS stay in same split
     → prevents reverse-complement data leakage
  2. Pure-PyTorch flash attention fallback (no Triton required)
     → compatible with Kaggle T4/P100 & Colab environments
  3. Mixed-precision (AMP) training for speed on GPU
  4. Embedding extraction + mCNN training in single script

Usage on Kaggle:
  1. Upload your data/processed/*.fasta files as a Kaggle Dataset
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

class Config:
    """Central configuration for the entire pipeline."""

    # ── Paths ──
    # Kaggle: /kaggle/input/<dataset-name>/data/processed/
    # Local:  data/processed/
    # Adjust DATA_DIR to match your Kaggle dataset mount path.
    DATA_DIR = "data/processed"
    OUTPUT_DIR = "outputs"
    FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

    # ── DNABERT-2 ──
    DNABERT_MODEL = "zhihan1996/DNABERT-2-117M"
    EMBEDDING_DIM = 768
    MAX_TOKEN_LENGTH = 512   # BPE tokenizer max length
    EMBED_BATCH_SIZE = 32    # Batch size for embedding extraction
    EMBED_DTYPE = np.float16 # float16 saves 50% RAM

    # ── mCNN Head ──
    BRANCH_CHANNELS = 128
    KERNEL_SIZES = [3, 5, 7, 9]
    DROPOUT_RATE = 0.5
    NUM_CLASSES = 4

    # ── Training ──
    TRAIN_BATCH_SIZE = 64
    VAL_BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 30
    PATIENCE = 7           # Early stopping patience
    MAX_OVERFITTING_GAP = 30.0  # Max train-val gap (%) to prevent severe overfitting
    LR_PATIENCE = 3        # ReduceLROnPlateau patience
    LR_FACTOR = 0.5
    GRAD_ACCUM_STEPS = 1   # Set to 1 for consistency since script 09 doesn't accumulate gradients

    # ── Data Split ──
    TEST_SIZE = 0.2
    RANDOM_SEED = 42

    # ── Class Names ──
    CLASS_NAMES = ["SP1", "SP2", "SP4", "Negative"]

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

    files = {
        "SP1": os.path.join(data_dir, "sp1_positive_final.fasta"),
        "SP2": os.path.join(data_dir, "sp2_positive_final.fasta"),
        "SP4": os.path.join(data_dir, "sp4_positive_final.fasta"),
        "Negative": os.path.join(data_dir, "negative_final.fasta"),
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
        # Positive classes: sequences are in orig/revcomp PAIRS (idx 0,1 = pair; 2,3 = pair; ...)
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
# CELL 5: Load DNABERT-2 Backbone
# ═══════════════════════════════════════════════════════════════════════

def load_dnabert2(model_name, device):
    """
    Load DNABERT-2 with robust multi-strategy loading.
    Handles ALiBi meta-device issues and Triton incompatibilities.
    """
    print("\n" + "=" * 60)
    print("LOADING DNABERT-2 BACKBONE")
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
        # Verify no meta-device tensors
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

    model = model.to(device)
    model.eval()

    # Freeze all backbone parameters (feature extractor only)
    for param in model.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  DNABERT-2 loaded on {device}")
    print(f"  Total parameters: {total_params:,} (all frozen)")

    return model, tokenizer


dnabert_model, tokenizer = load_dnabert2(cfg.DNABERT_MODEL, DEVICE)

# ═══════════════════════════════════════════════════════════════════════
# CELL 6: Extract Embeddings
# ═══════════════════════════════════════════════════════════════════════

def extract_embeddings(model, tokenizer, sequences, batch_size=32,
                       max_length=512, device=DEVICE, dtype=np.float16):
    """
    Extract token-level embeddings from DNABERT-2.
    Returns: np.ndarray of shape (n_samples, token_seq_len, 768)
    Uses padding='max_length' to ensure consistent tensor dimensions.
    """
    print(f"\nExtracting embeddings for {len(sequences)} sequences...")
    embeddings_list = []

    # Determine actual max token length from a sample
    sample_enc = tokenizer(
        sequences[:10], return_tensors="pt",
        padding="max_length", truncation=True, max_length=max_length
    )
    actual_token_len = sample_enc["input_ids"].shape[1]
    print(f"  Token sequence length: {actual_token_len}")

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="  Embedding"):
            batch_seqs = sequences[i:i + batch_size]

            inputs = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            ).to(device)

            outputs = model(**inputs)
            batch_emb = outputs[0].cpu().numpy().astype(dtype)
            embeddings_list.append(batch_emb)

            del inputs, outputs
            if device.type == "cuda" and i % (batch_size * 5) == 0:
                torch.cuda.empty_cache()

    result = np.concatenate(embeddings_list, axis=0)
    del embeddings_list
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"  Embeddings shape: {result.shape} (dtype={result.dtype})")
    return result


t_embed_start = time.time()
X_train_emb = extract_embeddings(
    dnabert_model, tokenizer, seq_train,
    batch_size=cfg.EMBED_BATCH_SIZE,
    max_length=cfg.MAX_TOKEN_LENGTH,
    dtype=cfg.EMBED_DTYPE,
)
X_test_emb = extract_embeddings(
    dnabert_model, tokenizer, seq_test,
    batch_size=cfg.EMBED_BATCH_SIZE,
    max_length=cfg.MAX_TOKEN_LENGTH,
    dtype=cfg.EMBED_DTYPE,
)
t_embed_end = time.time()
print(f"\nTotal embedding extraction time: {t_embed_end - t_embed_start:.1f}s")

# Free DNABERT-2 backbone from GPU memory
del dnabert_model
gc.collect()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
print("DNABERT-2 backbone freed from GPU memory.")

# ═══════════════════════════════════════════════════════════════════════
# CELL 7: mCNN Model Definition
# ═══════════════════════════════════════════════════════════════════════

class Conv1DBranch(nn.Module):
    """Conv1D → BatchNorm → ReLU → GlobalMaxPool branch."""
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x)))).squeeze(-1)


class MultiScaleCNN(nn.Module):
    """
    Multi-Scale CNN classifier on top of DNABERT-2 embeddings.
    Input: (batch, seq_len, 768) → Output: (batch, num_classes)
    """
    def __init__(self, embedding_dim=768, branch_channels=128,
                 kernel_sizes=[3, 5, 7, 9], num_classes=4, dropout_rate=0.5):
        super().__init__()

        self.branches = nn.ModuleList([
            Conv1DBranch(embedding_dim, branch_channels, k)
            for k in kernel_sizes
        ])

        concat_dim = len(kernel_sizes) * branch_channels

        self.fc_head = nn.Sequential(
            nn.Linear(concat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, embed_dim) → Conv1D needs (batch, embed_dim, seq_len)
        x = x.transpose(1, 2)
        branch_out = [branch(x) for branch in self.branches]
        feat = torch.cat(branch_out, dim=1)
        return self.fc_head(feat)

# ═══════════════════════════════════════════════════════════════════════
# CELL 8: PyTorch Dataset & DataLoaders
# ═══════════════════════════════════════════════════════════════════════

class DNAEmbeddingDataset(Dataset):
    """Dataset wrapping pre-extracted DNABERT-2 embeddings."""
    def __init__(self, embeddings, labels):
        self.embeddings = torch.from_numpy(embeddings) if isinstance(embeddings, np.ndarray) else embeddings
        self.labels = torch.from_numpy(labels) if isinstance(labels, np.ndarray) else labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx].float(), self.labels[idx].long()


train_dataset = DNAEmbeddingDataset(X_train_emb, y_train)
test_dataset = DNAEmbeddingDataset(X_test_emb, y_test)

train_loader = DataLoader(
    train_dataset, batch_size=cfg.TRAIN_BATCH_SIZE, shuffle=True,
    num_workers=2, pin_memory=True, drop_last=False,
)
test_loader = DataLoader(
    test_dataset, batch_size=cfg.VAL_BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=True,
)

print(f"DataLoaders ready: {len(train_loader)} train batches, {len(test_loader)} test batches")

# ═══════════════════════════════════════════════════════════════════════
# CELL 9: Training Loop with Early Stopping & AMP
# ═══════════════════════════════════════════════════════════════════════

def train_mcnn(model, train_loader, val_loader, cfg, device):
    """
    Train the mCNN head with:
    - Mixed-precision training (AMP)
    - ReduceLROnPlateau scheduler
    - Early stopping
    - Best model checkpoint saving
    """
    print("\n" + "=" * 60)
    print("TRAINING mCNN HEAD")
    print("=" * 60)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=cfg.LR_PATIENCE, factor=cfg.LR_FACTOR
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(cfg.EPOCHS):
        t0 = time.time()

        # ── Training ──
        model.train()
        train_loss_sum, train_total = 0.0, 0
        all_train_preds, all_train_targets = [], []

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            _, predicted = outputs.max(1)
            preds_gathered, targets_gathered = accelerator.gather_for_metrics((predicted, targets))
            loss_gathered = accelerator.gather(loss).mean().item()

            train_loss_sum += loss_gathered * targets_gathered.size(0)
            train_total += targets_gathered.size(0)
            all_train_preds.extend(preds_gathered.cpu().numpy())
            all_train_targets.extend(targets_gathered.cpu().numpy())

        train_loss = train_loss_sum / train_total
        train_acc = accuracy_score(all_train_targets, all_train_preds)

        # ── Validation ──
        model.eval()
        val_loss_sum, val_total = 0.0, 0
        all_val_preds, all_val_targets = [], []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with accelerator.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                _, predicted = outputs.max(1)
                preds_gathered, targets_gathered = accelerator.gather_for_metrics((predicted, targets))
                loss_gathered = accelerator.gather(loss).mean().item()

                val_loss_sum += loss_gathered * targets_gathered.size(0)
                val_total += targets_gathered.size(0)
                all_val_preds.extend(preds_gathered.cpu().numpy())
                all_val_targets.extend(targets_gathered.cpu().numpy())

        val_loss = val_loss_sum / val_total
        val_acc = accuracy_score(all_val_targets, all_val_preds)

        scheduler.step(val_loss)
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

        # ── Checkpointing & Early Stopping ──
        accelerator.wait_for_everyone()
        if val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            ckpt_path = os.path.join(cfg.MODEL_DIR, "best_dnabert2_mcnn.pt")
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


# Initialize & train
mcnn = MultiScaleCNN(
    embedding_dim=cfg.EMBEDDING_DIM,
    branch_channels=cfg.BRANCH_CHANNELS,
    kernel_sizes=cfg.KERNEL_SIZES,
    num_classes=cfg.NUM_CLASSES,
    dropout_rate=cfg.DROPOUT_RATE,
)

total_params = sum(p.numel() for p in mcnn.parameters())
trainable_params = sum(p.numel() for p in mcnn.parameters() if p.requires_grad)
print(f"mCNN Parameters: {total_params:,} total, {trainable_params:,} trainable")

history = train_mcnn(mcnn, train_loader, test_loader, cfg, DEVICE)

# ═══════════════════════════════════════════════════════════════════════
# CELL 10: Load Best Model & Full Evaluation
# ═══════════════════════════════════════════════════════════════════════

# Load best checkpoint
best_path = os.path.join(cfg.MODEL_DIR, "best_dnabert2_mcnn.pt")
checkpoint = torch.load(best_path, map_location=DEVICE)
unwrapped = accelerator.unwrap_model(mcnn)
unwrapped.load_state_dict(checkpoint["model_state_dict"])
mcnn = unwrapped
mcnn.eval()
print(f"Loaded best model from {best_path}")


def full_evaluation(model, test_loader, class_names, device):
    """Run full evaluation and return predictions/probabilities."""
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            with accelerator.autocast():
                outputs = model(inputs)
            probs = torch.softmax(outputs.float(), dim=1)
            _, preds = outputs.max(1)

            # Gather predictions, labels, and probabilities across processes
            preds_gathered, targets_gathered = accelerator.gather_for_metrics((preds, targets))
            probs_gathered = accelerator.gather_for_metrics(probs)

            all_preds.extend(preds_gathered.cpu().numpy())
            all_targets.extend(targets_gathered.cpu().numpy())
            all_probs.extend(probs_gathered.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4))

    acc = accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average="macro")
    print(f"Overall Accuracy:  {acc:.4f}")
    print(f"Macro F1-Score:    {f1_macro:.4f}")

    return all_preds, all_targets, all_probs


all_preds, all_targets, all_probs = full_evaluation(mcnn, test_loader, cfg.CLASS_NAMES, DEVICE)

# ═══════════════════════════════════════════════════════════════════════
# CELL 11: Generate All Performance Figures
# ═══════════════════════════════════════════════════════════════════════

def plot_training_curves(history, save_dir):
    """Plot loss and accuracy curves."""
    epochs = len(history["train_loss"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(range(1, epochs+1), history["train_loss"], label="Train Loss",
             color="#2196F3", linewidth=2)
    ax1.plot(range(1, epochs+1), history["val_loss"], label="Val Loss",
             color="#FF5722", linewidth=2)
    ax1.set_title("Loss Convergence", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Accuracy
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

    plt.suptitle("DNABERT-2 + mCNN Training Progress", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "dnabert2_mcnn_training_curves.png")
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

    plt.suptitle("DNABERT-2 + mCNN", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "dnabert2_mcnn_confusion_matrix.png")
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

    # Macro-average
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
    plt.title("ROC Curves — DNABERT-2 + mCNN", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "dnabert2_mcnn_roc_curves.png")
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
    plt.title("Precision-Recall Curves — DNABERT-2 + mCNN", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "dnabert2_mcnn_pr_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → PR curves: {path}")


def plot_per_class_metrics_bar(all_targets, all_preds, class_names, save_dir):
    """Per-class bar chart of Precision, Recall, F1."""
    from sklearn.metrics import precision_recall_fscore_support

    prec, rec, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, average=None
    )

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, prec, width, label="Precision", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x, rec, width, label="Recall", color="#4CAF50", alpha=0.85)
    bars3 = ax.bar(x + width, f1, width, label="F1-Score", color="#FF9800", alpha=0.85)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{cn}\n(n={s})" for cn, s in zip(class_names, support)])
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Performance — DNABERT-2 + mCNN", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(save_dir, "dnabert2_mcnn_per_class_metrics.png")
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
# CELL 12: Summary
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PIPELINE SUMMARY")
print("=" * 60)
print(f"  Model:            DNABERT-2 (117M, frozen) + mCNN head")
print(f"  Tokenizer:        BPE (Byte Pair Encoding)")
print(f"  Attention:        ALiBi (patched to pure PyTorch)")
print(f"  Data split:       GroupShuffleSplit (revcomp-aware)")
print(f"  Training samples: {len(seq_train)}")
print(f"  Test samples:     {len(seq_test)}")
print(f"  Best val loss:    {min(history['val_loss']):.4f}")
print(f"  Best val acc:     {max(history['val_acc']):.4f}")
print(f"  Figures saved to: {cfg.FIG_DIR}")
print(f"  Model saved to:   {cfg.MODEL_DIR}")
print("=" * 60)
