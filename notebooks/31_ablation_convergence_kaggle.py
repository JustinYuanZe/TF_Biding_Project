#!/usr/bin/env python3
"""
Script 31: Ablation CONVERGENCE Study (Binary) -- did the no-bio variants merely
           UNDER-CONVERGE at 15 epochs, or is the bio-branch gain real?
Designed for: Kaggle GPU (1xT4 or 2xT4 via HuggingFace Accelerate)

WHY THIS SCRIPT EXISTS
  Script 29's controlled ablation found that adding the Bio branch jumps accuracy
  A3 (seq+shape+attn) 77.7% -> A4 (full tri-branch) 89.6% (+11.9pp), with Negative
  recall 0.277 -> 0.666. A natural objection: the no-bio variants (A1-A3) were only
  trained for 15 epochs -- maybe they were simply STILL IMPROVING (under-converged),
  and the bio variant didn't get a real architectural advantage, it just happened to
  sit higher on a slower learning curve.

  This script re-runs the INCREMENTAL ladder A1 -> A2 -> A3 -> A4 with the SAME
  proven machinery as script 29, but:
    * EPOCHS = 30 (double the original budget; configurable below)
    * PATIENCE = 8 (more generous early-stopping so slow learners get to plateau)
    * For each variant it explicitly logs:
        - best val accuracy and the EPOCH it was reached,
        - whether the best epoch was at/near the cap (=> still improving => under-converged),
        - the val-accuracy slope over the final few epochs,
        - a still_improving boolean flag.
    * It writes outputs_ablation_convergence/convergence_results.csv and an
      OVERLAID val-accuracy-vs-epoch plot for A1-A4 on one axis, so that
      under-convergence (a curve still rising at the cap) is visually obvious.

  Interpretation guide written to CSV/plot:
    - If A1-A3 plateau well before epoch 30 and STILL sit far below A4 -> the bio
      gain is REAL (not an under-convergence artifact).
    - If A1-A3 are still climbing at epoch 30 and approach A4 -> the original
      15-epoch comparison was confounded by under-convergence.

ARCHITECTURE / DATA / TRAINING are copied faithfully from script 29 (this notebook
is self-contained; we do NOT import across notebooks). The ConfigurableGCMAB, data
loading, backbone snapshot/reset, train/eval loops, and SHAP are identical to 29.
"""

# ═══════════════════════════════════════════════════════════════════════
# CELL 0: Dependencies
# ═══════════════════════════════════════════════════════════════════════
import subprocess
import sys

def install_packages():
    for pkg in ["transformers>=4.37.0", "einops>=0.7.0", "datasets>=2.16.0",
                "accelerate>=0.25.0", "safetensors>=0.4.0", "shap>=0.44.0", "seaborn>=0.12.0"]:
        try:
            __import__(pkg.split(">=")[0].split("==")[0])
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
import csv
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
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, accuracy_score,
    f1_score, precision_recall_fscore_support,
)

import builtins
orig_import = builtins.__import__

def custom_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'triton' or name.startswith('triton.'):
        try:
            frame = sys._getframe(1)
            while frame:
                if 'flash_attn_triton' in frame.f_code.co_filename or 'bert_layers' in frame.f_code.co_filename:
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
# CELL 2: Config + Ablation variants
# ═══════════════════════════════════════════════════════════════════════
def find_file(filename, fallback_dir="data/processed"):
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if filename in files:
                return os.path.join(root, filename)
    if fallback_dir and os.path.exists(fallback_dir):
        p1 = os.path.join(fallback_dir, filename)
        if os.path.exists(p1):
            return p1
        for root, _, files in os.walk(fallback_dir):
            if filename in files:
                return os.path.join(root, filename)
    return filename if os.path.exists(filename) else None


def auto_detect_dir(target_file, fallback="data/processed"):
    r = find_file(target_file, fallback)
    return os.path.dirname(r) if r else fallback


class Config:
    FASTA_DIR = auto_detect_dir("sp1_positive_final.fasta", "data/processed")
    SHAPE_DIR = auto_detect_dir("dnashape_sp1.npy", "data/processed")
    OUTPUT_DIR = "outputs_ablation_convergence"
    DNABERT_MODEL = "zhihan1996/DNABERT-2-117M"
    EMBEDDING_DIM = 768

    AUTO_MAX_LENGTH = True
    MAX_TOKEN_LENGTH = 48
    MAX_LENGTH_CAP = 96
    MAX_LENGTH_FLOOR = 32

    UNFREEZE_LAST_N_LAYERS = 6
    BACKBONE_LR = 2e-5
    CROSS_ATTN_D_MODEL = 128
    CROSS_ATTN_NHEAD = 4
    CROSS_ATTN_LAYERS = 1
    CROSS_ATTN_DROPOUT = 0.1
    CROSS_ATTN_LR = 2e-4
    SEQ_PROJ_LR = 2e-4
    SHAPE_CHANNELS = 5
    SHAPE_SEQ_LEN = 101
    SHAPE_LR = 3e-4
    MSCNN_OUT_CHANNELS = 256
    SEQ_MSCNN_KERNELS = [7, 9, 11, 15]
    SHAPE_MSCNN_KERNELS = [4, 8, 12, 16]
    GROUPNORM_GROUPS = 16
    LAYER_ATTN_N = 6
    LAYER_ATTN_LR = 1e-3
    HIDDEN_DIM = 256
    FUSION_DROPOUT = 0.7
    HEAD_LR = 2e-4
    WEIGHT_DECAY = 0.1

    # ───────────────────────────────────────────────────────────────────
    # CONVERGENCE STUDY KNOBS (the only training-budget changes vs script 29)
    #   EPOCHS doubled to 30 to let slow learners plateau; PATIENCE raised to 8.
    #   ===>> EDIT THESE TWO LINES to change the convergence budget. <<===
    # ───────────────────────────────────────────────────────────────────
    EPOCHS = 30        # CONVERGENCE: was 15 in script 29; doubled here.
    PATIENCE = 8       # CONVERGENCE: was 6 in script 29; more generous here.
    # Number of trailing epochs used to estimate the late-stage val-acc slope
    # (used to decide whether a variant was "still improving" at the cap).
    CONVERGENCE_SLOPE_WINDOW = 5
    # If best epoch is within this many epochs of the cap AND late slope > 0,
    # we flag the variant as "still improving" (=> under-converged).
    CONVERGENCE_TAIL_MARGIN = 3
    CONVERGENCE_SLOPE_EPS = 1e-4   # min slope (acc/epoch) to call "still improving"

    BATCH_SIZE = 64
    GRAD_ACCUM_STEPS = 1
    MAX_OVERFITTING_GAP = 30.0
    WARMUP_RATIO = 0.15
    MAX_GRAD_NORM = 0.5

    TEST_SIZE = 0.2
    RANDOM_SEED = 42
    CLASS_NAMES = ["Negative", "SP_Positive"]
    SHAPE_FILES = {"SP1": "dnashape_sp1.npy", "SP2": "dnashape_sp2.npy",
                   "SP4": "dnashape_sp4.npy", "Negative": "dnashape_negative.npy"}

    SHAP_BG_SIZE = 50
    SHAP_NUM_EXPLAIN = 20
    SHAP_WINDOW = 15

cfg = Config()

# Default variant flags; each variant overrides a subset.
DEFAULT_FLAGS = dict(use_shape=True, use_bio=True, use_cross_attn=True,
                     use_mscnn=True, use_layer_attn=True, use_groupnorm=True)

ABLATION_VARIANTS = [
    {"name": "A1_seq_only",         **{**DEFAULT_FLAGS, "use_shape": False, "use_bio": False, "use_cross_attn": False}},
    {"name": "A2_seq_shape_noattn", **{**DEFAULT_FLAGS, "use_bio": False, "use_cross_attn": False}},
    {"name": "A3_seq_shape_attn",   **{**DEFAULT_FLAGS, "use_bio": False}},
    {"name": "A4_full_tribranch",   **DEFAULT_FLAGS},
    {"name": "A5_full_no_mscnn",    **{**DEFAULT_FLAGS, "use_mscnn": False}},
    {"name": "A6_full_no_layerattn",**{**DEFAULT_FLAGS, "use_layer_attn": False}},
    {"name": "A7_full_batchnorm",   **{**DEFAULT_FLAGS, "use_groupnorm": False}},
]
# CONVERGENCE STUDY: restrict to the incremental ladder A1 -> A2 -> A3 -> A4.
# (Component knock-outs A5-A7 are irrelevant to the under-convergence question.)
VARIANTS_TO_RUN = ["A1_seq_only", "A2_seq_shape_noattn", "A3_seq_shape_attn", "A4_full_tribranch"]

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

random.seed(cfg.RANDOM_SEED)
np.random.seed(cfg.RANDOM_SEED)
torch.manual_seed(cfg.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.RANDOM_SEED)

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=cfg.GRAD_ACCUM_STEPS,
                          kwargs_handlers=[ddp_kwargs])
DEVICE = accelerator.device
if not accelerator.is_main_process:
    builtins.print = lambda *a, **k: None

print(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()} | procs {accelerator.num_processes}")
print(f"CONVERGENCE STUDY | variants: {VARIANTS_TO_RUN}")
print(f"Epochs/variant: {cfg.EPOCHS} (was 15 in script 29) | Patience: {cfg.PATIENCE} (was 6)")
print(f"Trains {len(VARIANTS_TO_RUN)} models to test under-convergence of the no-bio ladder.")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════════════
# CELL 3: Data Loading
# ═══════════════════════════════════════════════════════════════════════
def load_fasta(filepath):
    sequences, headers = [], []
    with open(filepath) as f:
        seq_lines, cur = [], None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_lines:
                    sequences.append("".join(seq_lines).upper()); headers.append(cur); seq_lines = []
                cur = line[1:]
            else:
                seq_lines.append(line)
        if seq_lines:
            sequences.append("".join(seq_lines).upper()); headers.append(cur)
    return sequences, headers


def load_shape_features(data_dir, shape_files):
    neg_path = None
    for cand in ["dnashape_negative_genomic.npy", "dnashape_negative_cpg.npy", "dnashape_negative.npy"]:
        p = find_file(cand, data_dir)
        if p:
            neg_path = p; break
    if not neg_path:
        raise FileNotFoundError("No negative DNAshape file found.")
    all_shapes = []
    for cls, fname in shape_files.items():
        p = neg_path if cls == "Negative" else find_file(fname, data_dir)
        if not p:
            raise FileNotFoundError(f"Missing DNAshape for {cls}")
        all_shapes.append(np.load(p))
    return np.concatenate(all_shapes, axis=0)


def load_all_data():
    neg_fa = None
    for cand in ["negative_genomic_matched.fasta", "negative_promoter_cpg.fasta", "negative_final.fasta"]:
        p = find_file(cand, cfg.FASTA_DIR)
        if p:
            neg_fa = p; break
    fasta_files = {"SP1": find_file("sp1_positive_final.fasta", cfg.FASTA_DIR),
                   "SP2": find_file("sp2_positive_final.fasta", cfg.FASTA_DIR),
                   "SP4": find_file("sp4_positive_final.fasta", cfg.FASTA_DIR),
                   "Negative": neg_fa}
    seqs_all, hdrs_all, labels, groups = [], [], [], []
    gid = 0
    for idx, (cls, fp) in enumerate(fasta_files.items()):
        if not fp:
            raise FileNotFoundError(f"Missing FASTA for {cls}")
        seqs, hdrs = load_fasta(fp)
        print(f"  {cls}: {len(seqs)} seqs")
        seqs_all.extend(seqs); hdrs_all.extend(hdrs); labels.extend([idx] * len(seqs))
        if cls != "Negative":
            for _ in range(0, len(seqs), 2):
                groups.extend([gid, gid]); gid += 1
        else:
            for _ in seqs:
                groups.append(gid); gid += 1
    labels, groups = np.array(labels), np.array(groups)
    shapes = load_shape_features(cfg.SHAPE_DIR, cfg.SHAPE_FILES)
    assert len(seqs_all) == shapes.shape[0], (len(seqs_all), shapes.shape)
    print(f"  Total {len(seqs_all)} seqs | shapes {shapes.shape}")
    return seqs_all, labels, groups, shapes, hdrs_all


seqs_all, labels_all, groups_all, shapes_all, hdrs_all = load_all_data()
gss = GroupShuffleSplit(n_splits=1, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_SEED)
tr_idx, te_idx = next(gss.split(seqs_all, labels_all, groups_all))
seq_train = [seqs_all[i] for i in tr_idx]; seq_test = [seqs_all[i] for i in te_idx]
y_train, y_test = labels_all[tr_idx], labels_all[te_idx]
shape_train, shape_test = shapes_all[tr_idx], shapes_all[te_idx]
headers_test = [hdrs_all[i] for i in te_idx]
del seqs_all, labels_all, groups_all, shapes_all, hdrs_all
gc.collect()

# Robust scaling (train stats only)
def robust_normalize(shape_train, shape_test):
    n_ch = shape_train.shape[1]
    tr = np.copy(shape_train).astype(np.float32); te = np.copy(shape_test).astype(np.float32)
    for ch in range(n_ch):
        v = shape_train[:, ch, :].flatten(); v = v[~np.isnan(v)]
        med = np.median(v); p1, p99 = np.percentile(v, 1), np.percentile(v, 99)
        sc = max(p99 - p1, 1e-9)
        tr[:, ch, :] = (tr[:, ch, :] - med) / sc
        te[:, ch, :] = (te[:, ch, :] - med) / sc
    return np.nan_to_num(tr, nan=0.0), np.nan_to_num(te, nan=0.0)

shape_train_norm, shape_test_norm = robust_normalize(shape_train, shape_test)
del shape_train, shape_test
gc.collect()

# Binary mapping
y_train = np.array([1 if l < 3 else 0 for l in y_train])
y_test = np.array([1 if l < 3 else 0 for l in y_test])
print(f"  Train dist (neg,pos): {np.bincount(y_train)} | Test: {np.bincount(y_test)}")

# ═══════════════════════════════════════════════════════════════════════
# CELL 4: Flash-attention patch (pure PyTorch)
# ═══════════════════════════════════════════════════════════════════════
def _fa_qkv(qkv, bias=None, causal=False, softmax_scale=None):
    q, k, v = qkv.unbind(dim=2); q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    sc = softmax_scale or (q.shape[-1] ** -0.5)
    a = torch.matmul(q, k.transpose(-2, -1)) * sc
    if bias is not None:
        a = a + bias
    a = F.softmax(a, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(a, v).transpose(1, 2).contiguous()

def _fa_kv(q, kv, bias=None, causal=False, softmax_scale=None):
    k, v = kv.unbind(dim=2); q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    sc = softmax_scale or (q.shape[-1] ** -0.5)
    a = torch.matmul(q, k.transpose(-2, -1)) * sc
    if bias is not None:
        a = a + bias
    a = F.softmax(a, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(a, v).transpose(1, 2).contiguous()

def _fa(q, k, v, bias=None, causal=False, softmax_scale=None):
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    sc = softmax_scale or (q.shape[-1] ** -0.5)
    a = torch.matmul(q, k.transpose(-2, -1)) * sc
    if bias is not None:
        a = a + bias
    a = F.softmax(a, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(a, v).transpose(1, 2).contiguous()

def patch_flash_attention():
    targets = {"flash_attn_qkvpacked_func": _fa_qkv, "flash_attn_kvpacked_func": _fa, "flash_attn_func": _fa}
    targets["flash_attn_kvpacked_func"] = _fa_kv
    n = 0
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if "flash_attn_triton" in mod_name or "bert_layers" in mod_name:
            for an, rep in targets.items():
                if hasattr(mod, an):
                    setattr(mod, an, rep); n += 1
    print(f"  Patched {n} flash-attn refs." if n else "  No flash-attn refs to patch.")

# ═══════════════════════════════════════════════════════════════════════
# CELL 5: Load DNABERT-2 backbone + snapshot pretrained weights
# ═══════════════════════════════════════════════════════════════════════
def load_dnabert2(model_name, unfreeze_last_n):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if not getattr(config, "pad_token_id", None):
        config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 3
    config.output_hidden_states = True

    model = None
    try:
        model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True, low_cpu_mem_usage=False)
        for _, p in model.named_parameters():
            if p.device == torch.device("meta"):
                raise RuntimeError("meta param")
        print("  Strategy 1 OK")
    except Exception as e:
        print(f"  Strategy 1 failed: {e}"); model = None
    if model is None:
        try:
            _orig = torch.empty
            def _patched(*a, **kw):
                if str(kw.get("device", "")) == "meta":
                    kw["device"] = "cpu"
                return _orig(*a, **kw)
            torch.empty = _patched
            try:
                model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True, low_cpu_mem_usage=False)
            finally:
                torch.empty = _orig
            print("  Strategy 3 (ALiBi patch) OK")
        except Exception as e:
            raise RuntimeError(f"All loading strategies failed: {e}") from e

    patch_flash_attention()
    for p in model.parameters():
        p.requires_grad = False
    total = len(model.encoder.layer)
    for i, layer in enumerate(model.encoder.layer):
        if i >= total - unfreeze_last_n:
            for p in layer.parameters():
                p.requires_grad = True
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  DNABERT-2: {total} layers, last {unfreeze_last_n} unfrozen | trainable {tr:,}")
    return model, tokenizer


dnabert_model, tokenizer = load_dnabert2(cfg.DNABERT_MODEL, cfg.UNFREEZE_LAST_N_LAYERS)
INITIAL_BACKBONE_STATE = {k: v.detach().cpu().clone() for k, v in dnabert_model.state_dict().items()}
print(f"  Snapshotted pretrained backbone state ({len(INITIAL_BACKBONE_STATE)} tensors) for per-variant reset.")


def compute_max_token_length(sequences, tok, n=2000, pct=99, floor=32, cap=96):
    sample = random.sample(sequences, n) if len(sequences) > n else sequences
    lens = [len(tok(s, add_special_tokens=True)["input_ids"]) for s in sample]
    chosen = int(max(floor, min(cap, int(np.percentile(lens, pct)) + 2)))
    print(f"  AUTO MAX_LENGTH: p{pct}={int(np.percentile(lens, pct))} -> {chosen}")
    return chosen

MAX_LENGTH = (compute_max_token_length(seq_train, tokenizer, floor=cfg.MAX_LENGTH_FLOOR, cap=cfg.MAX_LENGTH_CAP)
              if cfg.AUTO_MAX_LENGTH else cfg.MAX_TOKEN_LENGTH)

# ═══════════════════════════════════════════════════════════════════════
# CELL 6: Architecture components + Configurable model
# ═══════════════════════════════════════════════════════════════════════
class LayerAttention(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))
        self.gamma = nn.Parameter(torch.ones(1))
    def forward(self, hs_list):
        w = F.softmax(self.layer_weights, dim=0)
        mixed = torch.zeros_like(hs_list[0])
        for wi, h in zip(w, hs_list):
            mixed = mixed + wi * h
        return self.gamma * mixed


def make_norm(num_channels, use_groupnorm, groups=16):
    return nn.GroupNorm(min(groups, num_channels), num_channels) if use_groupnorm else nn.BatchNorm1d(num_channels)


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, use_groupnorm=True, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1, bias=True)
        self.norm = make_norm(out_ch, use_groupnorm)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.norm(self.pointwise(self.depthwise(x))))


class MSCNNBranchStack(nn.Module):
    def __init__(self, in_ch, out_ch, kernels, use_groupnorm=True):
        super().__init__()
        self.branches = nn.ModuleList([DepthwiseSeparableConv1d(in_ch, out_ch, k, use_groupnorm) for k in kernels])
    def forward(self, x):
        return torch.cat([b(x).max(dim=2)[0] for b in self.branches], dim=1)


class FeedForward(nn.Module):
    def __init__(self, d_model, expansion=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model * expansion), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d_model * expansion, d_model), nn.Dropout(dropout))
    def forward(self, x):
        return x + self.net(x)


class CrossModalAttentionLayer(nn.Module):
    def __init__(self, d_model=128, nhead=4, dropout=0.1):
        super().__init__()
        self.a1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.n1 = nn.LayerNorm(d_model); self.f1 = FeedForward(d_model, 4, dropout)
        self.a2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d_model); self.f2 = FeedForward(d_model, 4, dropout)
    def forward(self, seq, shape, seq_key_padding_mask=None):
        att_s, _ = self.a1(query=seq, key=shape, value=shape)
        seq_out = self.f1(self.n1(seq + att_s))
        att_sh, _ = self.a2(query=shape, key=seq, value=seq, key_padding_mask=seq_key_padding_mask)
        shape_out = self.f2(self.n2(shape + att_sh))
        return seq_out, shape_out


def mean_max_pool(x):  # x: [B, C, L] -> [B, 2C]
    return torch.cat([x.mean(dim=2), x.max(dim=2)[0]], dim=1)


class ConfigurableGCMAB(nn.Module):
    """One architecture, ablatable via flags. Binary single-logit head."""
    def __init__(self, backbone, cfg, flags):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        self.use_shape = flags["use_shape"]
        self.use_bio = flags["use_bio"]
        self.use_cross_attn = flags["use_cross_attn"] and self.use_shape
        self.use_mscnn = flags["use_mscnn"]
        self.use_layer_attn = flags["use_layer_attn"]
        self.use_groupnorm = flags["use_groupnorm"]
        self._layer_attn_fallback = False
        d = cfg.CROSS_ATTN_D_MODEL

        if self.use_layer_attn:
            self.layer_attention = LayerAttention(cfg.LAYER_ATTN_N)

        self.seq_projection = nn.Sequential(nn.Linear(cfg.EMBEDDING_DIM, d), nn.LayerNorm(d), nn.GELU())

        seq_in_ch = d
        if self.use_shape:
            if self.use_cross_attn:
                self.shape_projection = nn.Sequential(
                    nn.Conv1d(cfg.SHAPE_CHANNELS, d, 1), make_norm(d, self.use_groupnorm, cfg.GROUPNORM_GROUPS), nn.GELU())
                self.cross_attention_layers = nn.ModuleList(
                    [CrossModalAttentionLayer(d, cfg.CROSS_ATTN_NHEAD, cfg.CROSS_ATTN_DROPOUT) for _ in range(cfg.CROSS_ATTN_LAYERS)])
                shape_in_ch = d
            else:
                shape_in_ch = cfg.SHAPE_CHANNELS

        if self.use_bio:
            self.bio_branch = nn.Sequential(nn.Linear(3, 16), nn.BatchNorm1d(16), nn.GELU(),
                                            nn.Dropout(0.1), nn.Linear(16, 32), nn.GELU())

        # Aggregation + fusion dim bookkeeping
        if self.use_mscnn:
            self.seq_agg = MSCNNBranchStack(seq_in_ch, cfg.MSCNN_OUT_CHANNELS, cfg.SEQ_MSCNN_KERNELS, self.use_groupnorm)
            seq_dim = len(cfg.SEQ_MSCNN_KERNELS) * cfg.MSCNN_OUT_CHANNELS
            if self.use_shape:
                self.shape_agg = MSCNNBranchStack(shape_in_ch, cfg.MSCNN_OUT_CHANNELS, cfg.SHAPE_MSCNN_KERNELS, self.use_groupnorm)
                shape_dim = len(cfg.SHAPE_MSCNN_KERNELS) * cfg.MSCNN_OUT_CHANNELS
        else:
            seq_dim = seq_in_ch * 2
            shape_dim = shape_in_ch * 2 if self.use_shape else 0

        in_features = seq_dim + (shape_dim if self.use_shape else 0) + (32 if self.use_bio else 0)
        self.fusion_dim = in_features
        self.classifier = nn.Sequential(nn.Linear(in_features, cfg.HIDDEN_DIM), nn.GELU(),
                                        nn.Dropout(cfg.FUSION_DROPOUT), nn.Linear(cfg.HIDDEN_DIM, 1))

    def _get_bert_features(self, input_ids, attention_mask):
        if self.use_layer_attn and not self._layer_attn_fallback:
            try:
                out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                if hasattr(out, "hidden_states") and out.hidden_states is not None:
                    hs = out.hidden_states
                    n = min(self.cfg.LAYER_ATTN_N, len(hs) - 1)
                    return self.layer_attention(list(hs[-n:]))
                elif isinstance(out, tuple) and len(out) > 2 and isinstance(out[2], (tuple, list)):
                    hs = out[2]; n = min(self.cfg.LAYER_ATTN_N, len(hs) - 1)
                    return self.layer_attention(list(hs[-n:]))
                else:
                    self._layer_attn_fallback = True
            except Exception:
                self._layer_attn_fallback = True
        if self.use_layer_attn and self._layer_attn_fallback:
            store, hooks = [], []
            n = min(self.cfg.LAYER_ATTN_N, len(self.backbone.encoder.layer))
            start = len(self.backbone.encoder.layer) - n
            def mk(s):
                def h(m, i, o):
                    s.append(o[0] if isinstance(o, tuple) else o)
                return h
            for i in range(start, len(self.backbone.encoder.layer)):
                hooks.append(self.backbone.encoder.layer[i].register_forward_hook(mk(store)))
            _ = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            for h in hooks:
                h.remove()
            mixed = self.layer_attention(store)
            if mixed.dim() == 2:
                B, T, D = attention_mask.size(0), attention_mask.size(1), mixed.size(-1)
                pad = torch.zeros(B, T, D, dtype=mixed.dtype, device=mixed.device)
                pad[attention_mask.bool()] = mixed
                mixed = pad
            return mixed
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out

    def _fuse_from_embeddings(self, seq_emb, shape_features, bio_features, attention_mask):
        seq_features = self.seq_projection(seq_emb)            # [B,T,d]
        if self.use_shape and self.use_cross_attn:
            shape_feats = self.shape_projection(shape_features).transpose(1, 2)  # [B,101,d]
            mask = (attention_mask == 0)
            for layer in self.cross_attention_layers:
                seq_features, shape_feats = layer(seq_features, shape_feats, mask)
            seq_in = seq_features.transpose(1, 2)               # [B,d,T]
            shape_in = shape_feats.transpose(1, 2)              # [B,d,101]
        else:
            seq_in = seq_features.transpose(1, 2)               # [B,d,T]
            shape_in = shape_features                           # [B,5,101]
        if self.use_mscnn:
            seq_vec = self.seq_agg(seq_in)
            shape_vec = self.shape_agg(shape_in) if self.use_shape else None
        else:
            seq_vec = mean_max_pool(seq_in)
            shape_vec = mean_max_pool(shape_in) if self.use_shape else None
        parts = [seq_vec]
        if self.use_shape:
            parts.append(shape_vec)
        if self.use_bio:
            parts.append(self.bio_branch(bio_features))
        return self.classifier(torch.cat(parts, dim=1))

    def forward(self, input_ids, attention_mask, shape_features, bio_features):
        hs = self._get_bert_features(input_ids, attention_mask)
        return self._fuse_from_embeddings(hs, shape_features, bio_features, attention_mask).squeeze(-1)

# ═══════════════════════════════════════════════════════════════════════
# CELL 7: Dataset
# ═══════════════════════════════════════════════════════════════════════
import re as regex

class TriBranchDataset(Dataset):
    def __init__(self, sequences, labels, shape_features, tokenizer, max_length=48):
        self.sequences = sequences; self.labels = labels; self.shape_features = shape_features
        self.tokenizer = tokenizer; self.max_length = max_length
        self.g4 = regex.compile(r'(G{3,}[ACGTN]{1,7}){3,}G{3,}', regex.IGNORECASE)
        self.bio = self._bio()
    def _bio(self):
        feats = []
        for seq in self.sequences:
            L = len(seq)
            if L == 0:
                feats.append([0.0, 0.0, 0.0]); continue
            c, g, cg = seq.count('C'), seq.count('G'), seq.count('CG')
            cpg_oe = (cg * L) / (c * g) if (c * g) > 0 else 0.0
            feats.append([cpg_oe, (c + g) / L, 1.0 if self.g4.search(seq) else 0.0])
        return np.array(feats, dtype=np.float32)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.sequences[idx], padding="max_length", truncation=True,
                             max_length=self.max_length, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "shape_features": torch.tensor(self.shape_features[idx], dtype=torch.float32),
                "bio_features": torch.tensor(self.bio[idx], dtype=torch.float32),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long)}

train_dataset = TriBranchDataset(seq_train, y_train, shape_train_norm, tokenizer, MAX_LENGTH)
test_dataset = TriBranchDataset(seq_test, y_test, shape_test_norm, tokenizer, MAX_LENGTH)

# ═══════════════════════════════════════════════════════════════════════
# CELL 8: Train / eval / plot / SHAP helpers (reused per variant)
# ═══════════════════════════════════════════════════════════════════════
def build_param_groups(model, cfg):
    pg = []
    bb = [p for p in model.backbone.parameters() if p.requires_grad]
    if bb:
        pg.append({"params": bb, "lr": cfg.BACKBONE_LR, "weight_decay": cfg.WEIGHT_DECAY})
    pg.append({"params": list(model.seq_projection.parameters()), "lr": cfg.SEQ_PROJ_LR, "weight_decay": cfg.WEIGHT_DECAY})
    if hasattr(model, "shape_projection"):
        pg.append({"params": list(model.shape_projection.parameters()), "lr": cfg.SHAPE_LR, "weight_decay": cfg.WEIGHT_DECAY})
    if hasattr(model, "cross_attention_layers"):
        pg.append({"params": list(model.cross_attention_layers.parameters()), "lr": cfg.CROSS_ATTN_LR, "weight_decay": cfg.WEIGHT_DECAY})
    if hasattr(model, "seq_agg"):
        pg.append({"params": list(model.seq_agg.parameters()), "lr": cfg.SHAPE_LR, "weight_decay": cfg.WEIGHT_DECAY})
    if hasattr(model, "shape_agg"):
        pg.append({"params": list(model.shape_agg.parameters()), "lr": cfg.SHAPE_LR, "weight_decay": cfg.WEIGHT_DECAY})
    if hasattr(model, "bio_branch"):
        pg.append({"params": list(model.bio_branch.parameters()), "lr": cfg.SHAPE_LR, "weight_decay": cfg.WEIGHT_DECAY})
    pg.append({"params": list(model.classifier.parameters()), "lr": cfg.HEAD_LR, "weight_decay": cfg.WEIGHT_DECAY})
    if hasattr(model, "layer_attention"):
        pg.append({"params": list(model.layer_attention.parameters()), "lr": cfg.LAYER_ATTN_LR, "weight_decay": 0.0})
    return pg


def train_one_epoch(model, loader, optimizer, scheduler, criterion, acc, max_grad_norm):
    model.train(); rl, correct, total = 0.0, 0, 0
    for batch in tqdm(loader, desc="  Train", leave=False, disable=not acc.is_main_process):
        with acc.accumulate(model):
            with acc.autocast():
                logits = model(batch["input_ids"], batch["attention_mask"], batch["shape_features"], batch["bio_features"])
                loss = criterion(logits, batch["labels"].float())
            acc.backward(loss)
            if acc.sync_gradients:
                acc.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
        rl += loss.item() * batch["labels"].size(0)
        total += batch["labels"].size(0)
        correct += (logits > 0).long().eq(batch["labels"]).sum().item()
    return rl / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, acc):
    model.eval(); rl, correct, total = 0.0, 0, 0
    for batch in loader:
        with acc.autocast():
            logits = model(batch["input_ids"], batch["attention_mask"], batch["shape_features"], batch["bio_features"])
            loss = criterion(logits, batch["labels"].float())
        preds = (logits > 0).long()
        preds, labels_g = acc.gather_for_metrics((preds, batch["labels"]))
        loss_g = acc.gather_for_metrics(loss.repeat(batch["labels"].size(0)))
        rl += loss_g.sum().item(); total += labels_g.size(0); correct += preds.eq(labels_g).sum().item()
    return rl / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def full_evaluation(model, loader, acc):
    model.eval(); P, Tg, Pr = [], [], []
    for batch in loader:
        with acc.autocast():
            logits = model(batch["input_ids"], batch["attention_mask"], batch["shape_features"], batch["bio_features"])
        probs = torch.sigmoid(logits.float()); preds = (logits > 0).long()
        pg, lg, prg = acc.gather_for_metrics((preds, batch["labels"], probs))
        P.extend(pg.cpu().numpy()); Tg.extend(lg.cpu().numpy()); Pr.extend(prg.cpu().numpy())
    return np.array(P), np.array(Tg), np.array(Pr)


def plot_variant_figures(history, targets, preds, probs, out_dir, prefix, class_names):
    ep = len(history["train_loss"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    a1.plot(range(1, ep+1), history["train_loss"], label="Train", color="#2196F3")
    a1.plot(range(1, ep+1), history["val_loss"], label="Val", color="#FF5722")
    a1.set_title("Loss"); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(range(1, ep+1), history["train_acc"], label="Train", color="#4CAF50")
    a2.plot(range(1, ep+1), history["val_acc"], label="Val", color="#E91E63")
    a2.axhline(0.5, color="gray", ls="--", alpha=0.5); a2.set_title("Accuracy"); a2.legend(); a2.grid(alpha=0.3)
    plt.suptitle(prefix); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_training_curves.png"), dpi=130, bbox_inches="tight"); plt.close()

    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(5, 4)); plt.imshow(cm, cmap="Blues")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right"); plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"{prefix} CM"); plt.colorbar(fraction=0.046)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{prefix}_confusion_matrix.png"), dpi=130, bbox_inches="tight"); plt.close()

    plt.figure(figsize=(7, 6))
    fpr, tpr, _ = roc_curve(targets, probs)
    plt.plot(fpr, tpr, color="#2196F3", lw=2, label=f"AUC={auc(fpr, tpr):.4f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"{prefix} ROC"); plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{prefix}_roc_curve.png"), dpi=130, bbox_inches="tight"); plt.close()

    plt.figure(figsize=(7, 6))
    pr, rc, _ = precision_recall_curve(targets, probs)
    plt.plot(rc, pr, color="#2196F3", lw=2, label=f"AP={average_precision_score(targets, probs):.4f}")
    plt.axhline(np.mean(targets == 1), color="gray", ls="--", alpha=0.5)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"{prefix} PR"); plt.legend(loc="lower left"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{prefix}_pr_curve.png"), dpi=130, bbox_inches="tight"); plt.close()


def run_shap(unwrapped, flags, out_dir, prefix, device):
    """Adaptive tri-modal SHAP: sequence always; +DNAshape bar if shape; +Bio bar
    & modality bar if bio. Subset A = correct SP_Positive, B = correct Negative."""
    try:
        import shap
        try:
            import seaborn as sns
            HAVE_SNS = True
        except ImportError:
            HAVE_SNS = False
        import re
        unwrapped.eval()
        use_shape, use_bio = flags["use_shape"], flags["use_bio"]

        sp_preds, sp_tgts = [], []
        loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        with torch.no_grad():
            for b in loader:
                lg = unwrapped(b["input_ids"].to(device), b["attention_mask"].to(device),
                               b["shape_features"].to(device), b["bio_features"].to(device))
                sp_preds.extend((lg > 0).long().cpu().numpy()); sp_tgts.extend(b["labels"].numpy())
        sp_preds, sp_tgts = np.array(sp_preds), np.array(sp_tgts)
        idx_TP = [i for i in range(len(sp_tgts)) if sp_tgts[i] == 1 and sp_preds[i] == 1]
        idx_TN = [i for i in range(len(sp_tgts)) if sp_tgts[i] == 0 and sp_preds[i] == 0]
        if len(idx_TP) < 3 or len(idx_TN) < 3:
            print(f"    [{prefix}] SHAP skipped (TP={len(idx_TP)}, TN={len(idx_TN)})"); return
        A, B = idx_TP[:cfg.SHAP_NUM_EXPLAIN], idx_TN[:cfg.SHAP_NUM_EXPLAIN]

        def stack(indices, field):
            return torch.stack([test_dataset[i][field] for i in indices]).to(device)

        np.random.seed(cfg.RANDOM_SEED)
        bgi = np.random.choice(len(test_dataset), min(cfg.SHAP_BG_SIZE, len(test_dataset)), replace=False)
        with torch.no_grad():
            bg_emb = unwrapped._get_bert_features(stack(bgi, "input_ids"), stack(bgi, "attention_mask")).float()

        def embed(indices):
            with torch.no_grad():
                return unwrapped._get_bert_features(stack(indices, "input_ids"), stack(indices, "attention_mask")).float()

        SHAPE_L = cfg.SHAPE_SEQ_LEN

        class Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__(); self.m = m
            def forward(self, *inp):
                seq_e = inp[0]; k = 1
                Bn, Tn, _ = seq_e.shape
                attn = torch.ones(Bn, Tn, dtype=torch.long, device=seq_e.device)
                if use_shape:
                    shape = inp[k]; k += 1
                else:
                    shape = torch.zeros(Bn, cfg.SHAPE_CHANNELS, SHAPE_L, device=seq_e.device)
                if use_bio:
                    bio = inp[k]; k += 1
                else:
                    bio = torch.zeros(Bn, 3, device=seq_e.device)
                return self.m._fuse_from_embeddings(seq_e, shape, bio, attn)

        wrapper = Wrapper(unwrapped).to(device).eval()
        bg_list = [bg_emb]
        if use_shape:
            bg_list.append(stack(bgi, "shape_features").float())
        if use_bio:
            bg_list.append(stack(bgi, "bio_features").float())

        def make_inputs(indices):
            lst = [embed(indices)]
            if use_shape:
                lst.append(stack(indices, "shape_features").float())
            if use_bio:
                lst.append(stack(indices, "bio_features").float())
            return lst

        explainer = shap.GradientExplainer(wrapper, bg_list)
        sv_A = explainer.shap_values(make_inputs(A))
        sv_B = explainer.shap_values(make_inputs(B))
        # Single-modality variants: GradientExplainer returns a bare array, not a list-of-1.
        if not isinstance(sv_A, (list, tuple)):
            sv_A = [sv_A]
        if not isinstance(sv_B, (list, tuple)):
            sv_B = [sv_B]

        def sq(a, nd):
            a = np.array(a)
            while a.ndim > nd and a.shape[-1] == 1:
                a = a[..., 0]
            return a

        # index map by active modality order
        order = ["seq"] + (["shape"] if use_shape else []) + (["bio"] if use_bio else [])
        def grab(sv, modal, nd):
            return sq(sv[order.index(modal)], nd)

        seq_A = grab(sv_A, "seq", 3); seq_B = grab(sv_B, "seq", 3)
        W = cfg.SHAP_WINDOW

        # DNAshape bar
        if use_shape:
            shp_A = grab(sv_A, "shape", 3); shp_B = grab(sv_B, "shape", 3)
            iA = np.abs(shp_A).sum(axis=2).mean(axis=0); iB = np.abs(shp_B).sum(axis=2).mean(axis=0)
            feats = ["MGW", "ProT", "Roll", "HelT", "EP"]; x = np.arange(5); w = 0.35
            plt.figure(figsize=(8, 5))
            plt.bar(x - w/2, iA, w, label="Correct SP_Positive", color="#4CAF50")
            plt.bar(x + w/2, iB, w, label="Correct Negative", color="#F44336")
            plt.xticks(x, feats); plt.ylabel("Mean |SHAP| (Σ positions)"); plt.title(f"{prefix} DNAshape importance")
            plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{prefix}_shap_dnashape_bar.png"), dpi=130); plt.close()

        # Bio bar + modality bar
        if use_bio:
            bio_A = grab(sv_A, "bio", 2); bio_B = grab(sv_B, "bio", 2)
            iA = np.abs(bio_A).mean(axis=0); iB = np.abs(bio_B).mean(axis=0)
            bf = ["CpG O/E", "GC", "G4"]; x = np.arange(3); w = 0.35
            plt.figure(figsize=(7, 5))
            plt.bar(x - w/2, iA, w, label="Correct SP_Positive", color="#4CAF50")
            plt.bar(x + w/2, iB, w, label="Correct Negative", color="#F44336")
            plt.xticks(x, bf); plt.ylabel("Mean |SHAP|"); plt.title(f"{prefix} Bio-feature importance")
            plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{prefix}_shap_bio_bar.png"), dpi=130); plt.close()

        # Modality-level contribution (over whatever modalities are active)
        def modal_tot(sv):
            vals = []
            for m, nd in [("seq", 3), ("shape", 3), ("bio", 2)]:
                if m in order:
                    a = grab(sv, m, nd)
                    vals.append(np.abs(a).reshape(a.shape[0], -1).sum(1).mean())
            return np.array(vals)
        mA, mB = modal_tot(sv_A), modal_tot(sv_B)
        labels_m = [m.capitalize() for m in order]
        x = np.arange(len(order)); w = 0.35
        plt.figure(figsize=(7, 5))
        plt.bar(x - w/2, mA, w, label="Correct SP_Positive", color="#4CAF50")
        plt.bar(x + w/2, mB, w, label="Correct Negative", color="#F44336")
        plt.xticks(x, labels_m); plt.ylabel("Total |SHAP| / sample"); plt.title(f"{prefix} Modality contribution")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_shap_modality_bar.png"), dpi=130); plt.close()

        # Sequence GC-box-aligned importance
        def char_shap(indices, seq_s):
            outs = []
            for j, oi in enumerate(indices):
                seq = seq_test[oi]; toks = tokenizer.convert_ids_to_tokens(test_dataset[oi]["input_ids"])
                imp = np.linalg.norm(seq_s[j], axis=1); cs = np.zeros(len(seq)); cc = np.zeros(len(seq)); cur = 0
                for ti, tok in enumerate(toks):
                    if tok in ["[CLS]", "[SEP]", "[PAD]", "<pad>", "<s>", "</s>", "<unk>"]:
                        continue
                    ct = tok.replace("##", "").replace("Ġ", "").replace(" ", "")
                    if not ct:
                        continue
                    pos = seq.find(ct, cur)
                    if pos != -1:
                        cs[pos:pos+len(ct)] += imp[ti]; cc[pos:pos+len(ct)] += 1; cur = pos + len(ct)
                outs.append(np.where(cc > 0, cs / cc, 0.0))
            return outs

        def align(indices, css):
            rows = []
            for oi, cs in zip(indices, css):
                m = re.search(r'GGGCGG|CCGCCC', seq_test[oi], re.IGNORECASE)
                if not m:
                    continue
                center = (m.span()[0] + m.span()[1]) // 2
                row = np.zeros(2 * W + 1)
                for i, off in enumerate(range(-W, W + 1)):
                    p = center + off
                    if 0 <= p < len(cs):
                        row[i] = cs[p]
                rows.append(row)
            return np.mean(rows, axis=0) if rows else None

        avg_A = align(A, char_shap(A, seq_A)); avg_B = align(B, char_shap(B, seq_B))
        if avg_A is not None:
            offs = np.arange(-W, W + 1)
            plt.figure(figsize=(10, 4))
            plt.plot(offs, avg_A, label="Correct SP_Positive", color="#4CAF50", lw=2)
            if avg_B is not None:
                plt.plot(offs, avg_B, label="Correct Negative", color="#F44336", lw=2)
            plt.axvline(0, color="gray", ls="--", alpha=0.7, label="GC-box center")
            plt.xlabel("Position rel. GC-box center (bp)"); plt.ylabel("Avg SHAP (L2)")
            plt.title(f"{prefix} sequence SHAP (GC-box flanks)"); plt.legend(); plt.grid(alpha=0.3)
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{prefix}_shap_sequence_line.png"), dpi=130); plt.close()
            if HAVE_SNS:
                rows = [avg_A] + ([avg_B] if avg_B is not None else [])
                yl = ["SP_Positive"] + (["Negative"] if avg_B is not None else [])
                plt.figure(figsize=(12, 2.5))
                sns.heatmap(np.vstack(rows), cmap="viridis", yticklabels=yl,
                            xticklabels=[str(o) for o in range(-W, W + 1)])
                plt.xlabel("Position rel. GC-box center (bp)"); plt.title(f"{prefix} sequence SHAP heatmap")
                plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{prefix}_shap_sequence_heatmap.png"), dpi=130); plt.close()
        print(f"    [{prefix}] SHAP done (modalities: {order}).")
    except Exception as e:
        print(f"    [{prefix}] SHAP error: {e}")
        import traceback; traceback.print_exc()


# ───────────────────────────────────────────────────────────────────────
# CONVERGENCE DIAGNOSTIC: given a full val-accuracy history (one entry per
# epoch actually trained), decide whether the variant was still improving at
# the cap (=> under-converged) and quantify the late-stage slope.
# ───────────────────────────────────────────────────────────────────────
def analyze_convergence(val_acc_hist, epochs_cap, slope_window, tail_margin, slope_eps):
    """Return a dict of convergence diagnostics for one variant.

    best_val_acc        : max val accuracy reached
    best_epoch          : 1-indexed epoch at which best_val_acc occurred
    epochs_trained      : number of epochs actually run (may be < cap if early-stopped)
    reached_cap         : did training run all the way to the configured cap?
    late_slope          : least-squares slope (acc/epoch) over the last `slope_window` epochs
    final_minus_best    : val_acc[-1] - best_val_acc (negative => already declining from peak)
    still_improving     : True if best epoch is within `tail_margin` of the LAST trained
                          epoch AND late_slope > slope_eps  (=> looks under-converged)
    """
    h = list(val_acc_hist)
    n = len(h)
    if n == 0:
        return {"best_val_acc": 0.0, "best_epoch": 0, "epochs_trained": 0, "reached_cap": False,
                "late_slope": 0.0, "final_minus_best": 0.0, "still_improving": False}
    best_idx = int(np.argmax(h))           # 0-indexed
    best_val_acc = float(h[best_idx])
    best_epoch = best_idx + 1              # 1-indexed
    reached_cap = (n >= epochs_cap)
    win = min(slope_window, n)
    if win >= 2:
        ys = np.array(h[-win:], dtype=np.float64)
        xs = np.arange(win, dtype=np.float64)
        late_slope = float(np.polyfit(xs, ys, 1)[0])
    else:
        late_slope = 0.0
    final_minus_best = float(h[-1] - best_val_acc)
    # "Still improving" = the best result sits at/near the END of training AND the
    # late-stage trend is still upward. Both conditions guard against false positives
    # (e.g. a noisy single late spike, or a peak followed by decline).
    near_tail = (best_epoch >= n - tail_margin)
    still_improving = bool(near_tail and (late_slope > slope_eps))
    return {"best_val_acc": round(best_val_acc, 4), "best_epoch": best_epoch,
            "epochs_trained": n, "reached_cap": reached_cap,
            "late_slope": round(late_slope, 6), "final_minus_best": round(final_minus_best, 4),
            "still_improving": still_improving}

# ═══════════════════════════════════════════════════════════════════════
# CELL 9: Convergence ablation loop (A1 -> A2 -> A3 -> A4, EPOCHS=30)
# ═══════════════════════════════════════════════════════════════════════
results = []
# Keep each variant's full val-accuracy history for the overlaid comparison plot.
val_acc_histories = {}

for variant in ABLATION_VARIANTS:
    name = variant["name"]
    if name not in VARIANTS_TO_RUN:
        continue
    flags = {k: variant[k] for k in DEFAULT_FLAGS}
    vdir = os.path.join(cfg.OUTPUT_DIR, name)
    os.makedirs(vdir, exist_ok=True)
    if accelerator.is_main_process:
        print("\n" + "#" * 60)
        print(f"# VARIANT: {name}  flags={flags}")
        print("#" * 60)

    # Reset seeds + backbone to pretrained for a fair, leakage-free comparison
    torch.manual_seed(cfg.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
    dnabert_model.load_state_dict(INITIAL_BACKBONE_STATE)

    model = ConfigurableGCMAB(dnabert_model, cfg, flags)
    if accelerator.is_main_process:
        print(f"  Fusion in_features = {model.fusion_dim} | cross_attn={model.use_cross_attn} mscnn={model.use_mscnn} "
              f"layer_attn={model.use_layer_attn} groupnorm={model.use_groupnorm}")

    optimizer = optim.AdamW(build_param_groups(model, cfg))
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE * 2, shuffle=False, num_workers=8, pin_memory=True)
    total_steps = (len(train_loader) // max(cfg.GRAD_ACCUM_STEPS, 1)) * cfg.EPOCHS
    warmup = int(total_steps * cfg.WARMUP_RATIO)

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * prog)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    pos_w = torch.tensor([float((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)],
                         dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val, best_epoch, patience = 0.0, 0, 0
    ckpt_path = os.path.join(vdir, "best.pt")
    for epoch in range(cfg.EPOCHS):
        t0 = time.time()
        tl, ta = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, accelerator, cfg.MAX_GRAD_NORM)
        vl, va = evaluate(model, test_loader, criterion, accelerator)
        tl = accelerator.gather(torch.tensor(tl, device=DEVICE)).mean().item()
        ta = accelerator.gather(torch.tensor(ta, device=DEVICE)).mean().item()
        history["train_loss"].append(tl); history["train_acc"].append(ta)
        history["val_loss"].append(vl); history["val_acc"].append(va)
        accelerator.wait_for_everyone()
        gap = (ta - va) * 100
        print(f"  [{name}] Ep {epoch+1:02d}/{cfg.EPOCHS} | Train {tl:.4f}/{ta:.4f} | Val {vl:.4f}/{va:.4f} | Gap {gap:+.1f}% | {time.time()-t0:.0f}s")
        if gap >= cfg.MAX_OVERFITTING_GAP:
            print("    early stop (overfitting gap)"); break
        if va > best_val:
            best_val, best_epoch, patience = va, epoch + 1, 0
            if accelerator.is_main_process:
                torch.save({"model_state_dict": accelerator.unwrap_model(model).state_dict(),
                            "val_acc": va, "epoch": epoch + 1, "flags": flags}, ckpt_path)
        else:
            patience += 1
            if patience >= cfg.PATIENCE:
                print("    early stop (patience)"); break
        accelerator.wait_for_everyone()

    # Load best & full eval (collective across processes)
    accelerator.wait_for_everyone()
    sd = torch.load(ckpt_path, map_location=DEVICE)["model_state_dict"]
    accelerator.unwrap_model(model).load_state_dict(sd)
    preds, targets, probs = full_evaluation(model, test_loader, accelerator)

    if accelerator.is_main_process:
        report = classification_report(targets, preds, target_names=cfg.CLASS_NAMES, digits=4)
        acc_v = accuracy_score(targets, preds)
        f1b = f1_score(targets, preds, average="binary")
        try:
            fpr, tpr, _ = roc_curve(targets, probs); roc_auc = auc(fpr, tpr)
        except Exception:
            roc_auc = float("nan")
        pr_auc = average_precision_score(targets, probs)
        prec, rec, f1c, _ = precision_recall_fscore_support(targets, preds, average=None, labels=[0, 1])

        # ── Convergence diagnostics on the full val-accuracy history ──
        conv = analyze_convergence(history["val_acc"], cfg.EPOCHS,
                                   cfg.CONVERGENCE_SLOPE_WINDOW, cfg.CONVERGENCE_TAIL_MARGIN,
                                   cfg.CONVERGENCE_SLOPE_EPS)
        val_acc_histories[name] = list(history["val_acc"])

        print(report)
        print(f"  [{name}] Acc {acc_v:.4f} | F1 {f1b:.4f} | ROC-AUC {roc_auc:.4f} | PR-AUC {pr_auc:.4f} | NegRecall {rec[0]:.4f}")
        print(f"  [{name}] CONVERGENCE: best_val_acc={conv['best_val_acc']:.4f} @ epoch {conv['best_epoch']}"
              f"/{conv['epochs_trained']} (cap {cfg.EPOCHS}) | late_slope={conv['late_slope']:+.5f}"
              f" | still_improving={conv['still_improving']}")
        if conv["still_improving"]:
            print(f"  [{name}] >>> FLAG: appears UNDER-CONVERGED (best at/near final epoch and still rising).")
        else:
            print(f"  [{name}] >>> appears CONVERGED (plateaued before the cap).")

        with open(os.path.join(vdir, "classification_report.txt"), "w") as f:
            f.write(f"CLASSIFICATION REPORT -- Convergence Ablation {name}\nflags={flags}\n" + "=" * 60 + "\n")
            f.write(report + f"\nAcc {acc_v:.4f} | F1 {f1b:.4f} | ROC-AUC {roc_auc:.4f} | PR-AUC {pr_auc:.4f}\n")
            f.write("\nCONVERGENCE DIAGNOSTICS\n")
            for k in ["best_val_acc", "best_epoch", "epochs_trained", "reached_cap",
                      "late_slope", "final_minus_best", "still_improving"]:
                f.write(f"  {k}: {conv[k]}\n")
            f.write(f"  epochs_cap: {cfg.EPOCHS}\n")
            f.write("  val_acc_per_epoch: " + ", ".join(f"{v:.4f}" for v in history["val_acc"]) + "\n")
        plot_variant_figures(history, targets, preds, probs, vdir, name, cfg.CLASS_NAMES)
        results.append({"variant": name, **flags,
                        "best_val_acc": conv["best_val_acc"], "best_epoch": conv["best_epoch"],
                        "epochs_trained": conv["epochs_trained"], "epochs_cap": cfg.EPOCHS,
                        "reached_cap": conv["reached_cap"], "late_slope": conv["late_slope"],
                        "final_minus_best": conv["final_minus_best"], "still_improving": conv["still_improving"],
                        "test_acc": round(acc_v, 4), "f1": round(f1b, 4),
                        "roc_auc": round(float(roc_auc), 4), "pr_auc": round(float(pr_auc), 4),
                        "neg_recall": round(rec[0], 4), "pos_recall": round(rec[1], 4),
                        "neg_precision": round(prec[0], 4),
                        "fusion_dim": model.fusion_dim if not hasattr(model, "module") else accelerator.unwrap_model(model).fusion_dim})
        # SHAP on the unwrapped best model (kept for parity with script 29)
        run_shap(accelerator.unwrap_model(model), flags, vdir, name, DEVICE)

    # Cleanup before next variant
    accelerator.wait_for_everyone()
    model = accelerator.unwrap_model(model)
    del model, optimizer, scheduler, train_loader, test_loader
    accelerator.free_memory()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════════════
# CELL 10: Aggregate convergence comparison (CSV + overlaid val-acc curves)
# ═══════════════════════════════════════════════════════════════════════
if accelerator.is_main_process and results:
    print("\n" + "=" * 60)
    print("CONVERGENCE SUMMARY (did the no-bio ladder under-converge at 15 epochs?)")
    print("=" * 60)
    keys = ["variant", "best_val_acc", "best_epoch", "epochs_trained",
            "still_improving", "late_slope", "test_acc", "neg_recall"]
    print(" | ".join(f"{k:>14s}" for k in keys))
    for r in results:
        print(" | ".join(f"{str(r[k]):>14s}" for k in keys))

    csv_path = os.path.join(cfg.OUTPUT_DIR, "convergence_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    print(f"\n  -> {csv_path}")

    # ── OVERLAID val-accuracy-vs-epoch curves for A1-A4 on ONE axis ──
    # Under-convergence (a curve still rising at the cap) is visually obvious here.
    palette = {"A1_seq_only": "#9C27B0", "A2_seq_shape_noattn": "#2196F3",
               "A3_seq_shape_attn": "#FF9800", "A4_full_tribranch": "#4CAF50"}
    plt.figure(figsize=(11, 6.5))
    for r in results:
        name = r["variant"]
        hist = val_acc_histories.get(name, [])
        if not hist:
            continue
        ep = np.arange(1, len(hist) + 1)
        color = palette.get(name, None)
        plt.plot(ep, hist, marker="o", ms=4, lw=2, color=color,
                 label=f"{name} (best {r['best_val_acc']:.3f} @ ep {r['best_epoch']}"
                       f"{', STILL RISING' if r['still_improving'] else ''})")
        # mark the best epoch
        plt.scatter([r["best_epoch"]], [r["best_val_acc"]], color=color, s=90,
                    edgecolors="black", zorder=5)
    plt.axvline(15, color="red", ls="--", alpha=0.6, label="Original 15-epoch cap (script 29)")
    plt.axhline(0.5, color="gray", ls=":", alpha=0.5)
    plt.xlabel("Epoch"); plt.ylabel("Validation accuracy")
    plt.title("Convergence check: val-accuracy vs epoch for the incremental ladder A1->A4\n"
              "(if no-bio curves are still rising past epoch 15, they UNDER-CONVERGED)")
    plt.legend(loc="lower right", fontsize=9); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, "convergence_val_acc_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {os.path.join(cfg.OUTPUT_DIR, 'convergence_val_acc_curves.png')}")

    # ── Verdict line for the paper ──
    nobio = [r for r in results if not r["use_bio"]]
    bio = [r for r in results if r["use_bio"]]
    any_nobio_under = any(r["still_improving"] for r in nobio)
    if bio and nobio:
        best_nobio = max(r["best_val_acc"] for r in nobio)
        best_bio = max(r["best_val_acc"] for r in bio)
        print("\nVERDICT:")
        print(f"  Best no-bio val acc (A1-A3): {best_nobio:.4f} | Best bio val acc (A4): {best_bio:.4f}"
              f" | gap {best_bio - best_nobio:+.4f}")
        if any_nobio_under:
            print("  At least one no-bio variant was STILL IMPROVING at the 30-epoch cap"
                  " -> the original 15-epoch comparison may have been confounded by under-convergence.")
        else:
            print("  No no-bio variant was still improving at 30 epochs (all plateaued)"
                  " -> the bio-branch gain is NOT merely an under-convergence artifact.")

    import shutil
    shutil.make_archive("outputs_ablation_convergence", "zip", cfg.OUTPUT_DIR)
    print("\nAll convergence outputs zipped into: outputs_ablation_convergence.zip")
