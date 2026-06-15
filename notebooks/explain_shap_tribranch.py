#!/usr/bin/env python3
"""
Auto-refactored script: explain_shap_tribranch.py
Refactored to align with project standards.
"""
"""
Diagnostic Script: Customized SHAP Interpretability Analysis for G-CMAB msCNN.
Allows running SHAP with different window sizes or absolute sequence indexing.
Usage:
    python notebooks/explain_shap.py --window_size 50 --bg_size 50 --num_explain 20
"""

import os
import re
import sys
import argparse
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit

# Use a custom import hook to force DNABERT-2 to fallback to standard PyTorch attention,
# while allowing other components (like torch._dynamo or HF check_imports) to import Triton normally.
import builtins
import sys

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

# ═══════════════════════════════════════════════════════════════════════
# 1. Model Architecture Definitions
# ═══════════════════════════════════════════════════════════════════════

class LayerAttention(nn.Module):
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
    def __init__(self, in_channels, out_channels, kernel_size, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, bias=True
        )
        self.norm = nn.GroupNorm(num_groups=min(16, out_channels), num_channels=out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class MSCNNBranchStack(nn.Module):
    def __init__(self, in_channels, out_channels, kernels):
        super().__init__()
        self.branches = nn.ModuleList([
            DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size=k)
            for k in kernels
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
        attended_seq, _ = self.cross_attn_seq2shape(
            query=seq_features, key=shape_features, value=shape_features
        )
        seq_out = self.norm_seq(seq_features + attended_seq)
        seq_out = self.ffn_seq(seq_out)

        attended_shape, _ = self.cross_attn_shape2seq(
            query=shape_features, key=seq_features, value=seq_features,
            key_padding_mask=seq_key_padding_mask
        )
        shape_out = self.norm_shape(shape_features + attended_shape)
        shape_out = self.ffn_shape(shape_out)
        return seq_out, shape_out

class GCMABmsCNNClassifier(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        d_model = cfg.CROSS_ATTN_D_MODEL
        self.use_cross_attn = getattr(cfg, "USE_CROSS_ATTN", True)

        self.bio_branch = nn.Sequential(
            nn.Linear(3, 16),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(16, 32),
            nn.GELU()
        )

        self.use_layer_attn = cfg.USE_LAYER_ATTN
        if self.use_layer_attn:
            self.layer_attention = LayerAttention(n_layers=cfg.LAYER_ATTN_N)
            self._layer_attn_fallback = False
        else:
            self.layer_attention = None

        self.seq_projection = nn.Sequential(
            nn.Linear(cfg.EMBEDDING_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        if self.use_cross_attn:
            self.shape_projection = nn.Sequential(
                nn.Conv1d(cfg.SHAPE_CHANNELS, d_model, kernel_size=1),
                nn.GroupNorm(min(16, d_model), d_model),
                nn.GELU(),
            )
            self.cross_attention_layers = nn.ModuleList([
                CrossModalAttentionLayer(d_model=d_model, nhead=cfg.CROSS_ATTN_NHEAD, dropout=cfg.CROSS_ATTN_DROPOUT)
                for _ in range(cfg.CROSS_ATTN_LAYERS)
            ])
            self.seq_mscnn = MSCNNBranchStack(
                in_channels=d_model, out_channels=cfg.MSCNN_OUT_CHANNELS, kernels=cfg.SEQ_MSCNN_KERNELS
            )
            self.shape_mscnn = MSCNNBranchStack(
                in_channels=d_model, out_channels=cfg.MSCNN_OUT_CHANNELS, kernels=cfg.SHAPE_MSCNN_KERNELS
            )
        else:
            self.seq_mscnn = MSCNNBranchStack(
                in_channels=d_model, out_channels=cfg.MSCNN_OUT_CHANNELS, kernels=cfg.SEQ_MSCNN_KERNELS
            )
            self.shape_mscnn = MSCNNBranchStack(
                in_channels=cfg.SHAPE_CHANNELS, out_channels=cfg.MSCNN_OUT_CHANNELS, kernels=cfg.SHAPE_MSCNN_KERNELS
            )

        in_features = (len(cfg.SEQ_MSCNN_KERNELS) + len(cfg.SHAPE_MSCNN_KERNELS)) * cfg.MSCNN_OUT_CHANNELS + 32
        self.classifier = nn.Sequential(
            nn.Linear(in_features, cfg.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(cfg.FUSION_DROPOUT),
            nn.Linear(cfg.HIDDEN_DIM, cfg.NUM_CLASSES)
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
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    all_hidden = outputs.hidden_states
                    n = min(self.cfg.LAYER_ATTN_N, len(all_hidden) - 1)
                    selected = list(all_hidden[-n:])
                    return self.layer_attention(selected)
                elif isinstance(outputs, tuple) and len(outputs) > 2:
                    all_hidden = outputs[2]
                    if isinstance(all_hidden, tuple) or isinstance(all_hidden, list):
                        n = min(self.cfg.LAYER_ATTN_N, len(all_hidden) - 1)
                        selected = list(all_hidden[-n:])
                        return self.layer_attention(selected)
                    else:
                        self._layer_attn_fallback = True
                else:
                    print("  [LayerAttn] WARNING: hidden_states not in output, using hook fallback...")
                    self._layer_attn_fallback = True
            except Exception as e:
                print(f"  [LayerAttn] WARNING: output_hidden_states failed ({e}), using hook fallback...")
                self._layer_attn_fallback = True

        if self.use_layer_attn and getattr(self, '_layer_attn_fallback', False):
            hidden_states_collected = []
            hooks = []
            n = min(self.cfg.LAYER_ATTN_N, len(self.backbone.encoder.layer))
            start_layer = len(self.backbone.encoder.layer) - n

            def make_hook(storage):
                def hook_fn(module, input, output):
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

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            return outputs[0]
        elif hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs

    def forward(self, input_ids, attention_mask, shape_features, bio_features):
        hidden_states = self._get_bert_features(input_ids, attention_mask)
        seq_features = self.seq_projection(hidden_states)

        if self.use_cross_attn:
            shape_proj = self.shape_projection(shape_features)
            shape_feats = shape_proj.transpose(1, 2)

            seq_key_padding_mask = (attention_mask == 0)
            for cross_layer in self.cross_attention_layers:
                seq_features, shape_feats = cross_layer(
                    seq_features, shape_feats, seq_key_padding_mask=seq_key_padding_mask
                )
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

class ModelEmbeddingWrapper(nn.Module):
    def __init__(self, gcmab_model):
        super().__init__()
        self.gcmab_model = gcmab_model
        
    def forward(self, seq_embeddings, shape_features, bio_features):
        seq_features = self.gcmab_model.seq_projection(seq_embeddings)
        
        if self.gcmab_model.use_cross_attn:
            shape_proj = self.gcmab_model.shape_projection(shape_features)
            shape_feats = shape_proj.transpose(1, 2)
            
            B, T, _ = seq_embeddings.shape
            device = seq_embeddings.device
            attention_mask = torch.ones(B, T, dtype=torch.long, device=device)
            seq_key_padding_mask = (attention_mask == 0)
            
            for cross_layer in self.gcmab_model.cross_attention_layers:
                seq_features, shape_feats = cross_layer(
                    seq_features, shape_feats, seq_key_padding_mask=seq_key_padding_mask
                )
            
            seq_in = seq_features.transpose(1, 2)
            shape_in = shape_feats.transpose(1, 2)
        else:
            seq_in = seq_features.transpose(1, 2)
            shape_in = shape_features
            
        seq_pooled = self.gcmab_model.seq_mscnn(seq_in)
        shape_pooled = self.gcmab_model.shape_mscnn(shape_in)
        bio_out = self.gcmab_model.bio_branch(bio_features)
        
        fused = torch.cat([seq_pooled, shape_pooled, bio_out], dim=1)
        logits = self.gcmab_model.classifier(fused)
        return logits

# ═══════════════════════════════════════════════════════════════════════
# 2. Config & Data Loader Utilities
# ═══════════════════════════════════════════════════════════════════════

class Config:
    DNABERT_MODEL = "zhihan1996/DNABERT-2-117M"
    EMBEDDING_DIM = 768
    CROSS_ATTN_D_MODEL = 128
    CROSS_ATTN_NHEAD = 4
    CROSS_ATTN_LAYERS = 1
    CROSS_ATTN_DROPOUT = 0.1
    USE_CROSS_ATTN = True
    SHAPE_CHANNELS = 5
    MSCNN_OUT_CHANNELS = 256
    SEQ_MSCNN_KERNELS = [7, 9, 11, 15]
    SHAPE_MSCNN_KERNELS = [4, 8, 12, 16]
    USE_LAYER_ATTN = True
    LAYER_ATTN_N = 6
    HIDDEN_DIM = 256
    FUSION_DROPOUT = 0.7
    NUM_CLASSES = 4
    RANDOM_SEED = 42

def find_file(filename, fallback_dir="data/processed"):
    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if filename in files:
                return os.path.join(root, filename)
    if fallback_dir and os.path.exists(fallback_dir):
        p1 = os.path.join(fallback_dir, filename)
        if os.path.exists(p1): return p1
        for root, _, files in os.walk(fallback_dir):
            if filename in files:
                return os.path.join(root, filename)
    if os.path.exists(filename):
        return filename
    return None

def load_fasta(filepath):
    sequences = []
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

import re as regex

class DualBranchDataset(Dataset):
    def __init__(self, sequences, labels, shape_features, tokenizer, max_length=32):
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
                features.append([0.0, 0.0, 0.0])
                continue
            c_count = seq.upper().count('C')
            g_count = seq.upper().count('G')
            cg_count = seq.upper().count('CG')
            cpg_oe = (cg_count * L) / (c_count * g_count) if (c_count * g_count) > 0 else 0.0
            gc_content = (c_count + g_count) / L
            g4_present = 1.0 if self.g4_pattern.search(seq) else 0.0
            features.append([cpg_oe, gc_content, g4_present])
        return np.array(features, dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.sequences[idx], padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        shape = torch.tensor(self.shape_features[idx], dtype=torch.float32)
        bio = torch.tensor(self.bio_features[idx], dtype=torch.float32)
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "shape_features": shape,
            "bio_features": bio,
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ═══════════════════════════════════════════════════════════════════════
# 3. Main Diagnostics Execution
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run customized SHAP interpretability for Script 21")
    parser.add_argument("--window_size", type=int, default=15, help="Context half-window size around GC-box (e.g. 50 for full sequence)")
    parser.add_argument("--bg_size", type=int, default=50, help="Background sample size for SHAP")
    parser.add_argument("--num_explain", type=int, default=20, help="Number of test samples to explain")
    parser.add_argument("--absolute", action="store_true", help="Calculate absolute position-wise SHAP (no GC-box alignment)")
    parser.add_argument("--target_class", type=int, default=2, choices=[0, 1, 2], help="Target class index to explain: 0 for SP1, 1 for SP2, 2 for SP4")
    parser.add_argument("--model_path", type=str, default="outputs_gcmab_tribranch/models/best_gcmab_tribranch.pt", help="Path to best_gcmab_tribranch.pt")
    parser.add_argument("--output_dir", type=str, default="outputs_gcmab_tribranch/figures", help="Directory to save figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Auto-detect data directories
    fasta_dir = "data/processed"
    shape_dir = "data/processed"
    if os.path.exists("/kaggle/input"):
        # Auto-detect CPg CpG CpG or genomic folders
        fasta_dir = "/kaggle/input"
        shape_dir = "/kaggle/input"

    # Resolve file paths
    sp1_f = find_file("sp1_positive_final.fasta", fasta_dir)
    sp2_f = find_file("sp2_positive_final.fasta", fasta_dir)
    sp4_f = find_file("sp4_positive_final.fasta", fasta_dir)
    
    neg_f = None
    for cand in ["negative_genomic_matched.fasta", "negative_promoter_cpg.fasta", "negative_final.fasta"]:
        neg_f = find_file(cand, fasta_dir)
        if neg_f: break
        
    sp1_s = find_file("dnashape_sp1.npy", shape_dir)
    sp2_s = find_file("dnashape_sp2.npy", shape_dir)
    sp4_s = find_file("dnashape_sp4.npy", shape_dir)
    
    neg_s = None
    for cand in ["dnashape_negative_genomic.npy", "dnashape_negative_cpg.npy", "dnashape_negative.npy"]:
        neg_s = find_file(cand, shape_dir)
        if neg_s: break

    # Load FASTA datasets
    print("Loading FASTA files...")
    all_sequences = []
    all_labels = []
    all_groups = []
    group_id = 0

    classes = ["SP1", "SP2", "SP4", "Negative"]
    files = [sp1_f, sp2_f, sp4_f, neg_f]
    
    for cls_idx, fpath in enumerate(files):
        seqs = load_fasta(fpath)
        all_sequences.extend(seqs)
        all_labels.extend([cls_idx] * len(seqs))
        if cls_idx != 3:
            for _ in range(0, len(seqs), 2):
                all_groups.extend([group_id, group_id])
                group_id += 1
        else:
            for _ in seqs:
                all_groups.append(group_id)
                group_id += 1
                
    all_labels = np.array(all_labels)
    all_groups = np.array(all_groups)

    # Load DNAshape
    print("Loading DNAshape features...")
    shapes = [np.load(sp1_s), np.load(sp2_s), np.load(sp4_s), np.load(neg_s)]
    all_shapes = np.concatenate(shapes, axis=0)
    
    # Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=Config.RANDOM_SEED)
    train_idx, test_idx = next(gss.split(all_sequences, all_labels, all_groups))
    
    seq_train = [all_sequences[i] for i in train_idx]
    seq_test = [all_sequences[i] for i in test_idx]
    y_train = all_labels[train_idx]
    y_test = all_labels[test_idx]
    shape_train = all_shapes[train_idx]
    shape_test = all_shapes[test_idx]

    # Robust scaling
    print("Normalizing DNAshape...")
    n_channels = shape_train.shape[1]
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

    shape_train_norm = np.nan_to_num(shape_train_norm, nan=0.0)
    shape_test_norm = np.nan_to_num(shape_test_norm, nan=0.0)

    # Load tokenizer and backbone model
    print("Initializing DNABERT-2 and model...")
    tokenizer = AutoTokenizer.from_pretrained(Config.DNABERT_MODEL, trust_remote_code=True)
    config = AutoConfig.from_pretrained(Config.DNABERT_MODEL, trust_remote_code=True)
    config.output_hidden_states = True
    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 3

    # Load empty backbone
    _orig = torch.empty
    def _patched(*a, **kw):
        if kw.get("device") == torch.device("meta") or str(kw.get("device", "")) == "meta":
            kw["device"] = "cpu"
        return _orig(*a, **kw)
    torch.empty = _patched
    try:
        backbone = AutoModel.from_pretrained(Config.DNABERT_MODEL, config=config, trust_remote_code=True)
    finally:
         torch.empty = _orig

    model = GCMABmsCNNClassifier(backbone=backbone, cfg=Config())
    print(f"Loading checkpoint weights from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load datasets
    test_dataset = DualBranchDataset(seq_test, y_test, shape_test_norm, tokenizer, max_length=32)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Predictions
    target_class = args.target_class
    target_name = classes[target_class]
    print(f"Running evaluation to identify correctly predicted vs confused {target_name} samples...")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            shape_features = batch["shape_features"].to(device)
            bio_features = batch["bio_features"].to(device)
            logits = model(input_ids, attention_mask, shape_features, bio_features)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch["labels"].numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    subset_A_indices = []
    subset_B_indices = []
    
    # We want subset B to contain samples of target_class that were confused with OTHER TF family classes.
    # TF family classes are [0, 1, 2] (SP1, SP2, SP4). Negative is 3.
    other_tf_classes = [c for c in [0, 1, 2] if c != target_class]
    
    for idx in range(len(all_targets)):
        if all_targets[idx] == target_class:
            pred = all_preds[idx]
            if pred == target_class:
                subset_A_indices.append(idx)
            elif pred in other_tf_classes:
                subset_B_indices.append(idx)
                
    print(f"Found {len(subset_A_indices)} correct {target_name}, {len(subset_B_indices)} confused {target_name} (predicted as {other_tf_classes}).")
    if len(subset_A_indices) == 0 or len(subset_B_indices) == 0:
        print(f"Error: Subset A or B is empty for target class {target_name}. Cannot run comparative SHAP.")
        return

    # Background
    np.random.seed(Config.RANDOM_SEED)
    bg_indices = np.random.choice(len(test_dataset), min(args.bg_size, len(test_dataset)), replace=False)
    bg_input_ids = torch.stack([test_dataset[i]["input_ids"] for i in bg_indices]).to(device)
    bg_attention_mask = torch.stack([test_dataset[i]["attention_mask"] for i in bg_indices]).to(device)
    bg_shapes = torch.stack([test_dataset[i]["shape_features"] for i in bg_indices]).to(device)
    bg_bio = torch.stack([test_dataset[i]["bio_features"] for i in bg_indices]).to(device)
    
    with torch.no_grad():
        bg_embeddings = model._get_bert_features(bg_input_ids, bg_attention_mask)

    # Explaining subsets
    explain_A_indices = subset_A_indices[:args.num_explain]
    explain_B_indices = subset_B_indices[:args.num_explain]

    test_A_input_ids = torch.stack([test_dataset[i]["input_ids"] for i in explain_A_indices]).to(device)
    test_A_attention_mask = torch.stack([test_dataset[i]["attention_mask"] for i in explain_A_indices]).to(device)
    test_A_shapes = torch.stack([test_dataset[i]["shape_features"] for i in explain_A_indices]).to(device)
    test_A_bio = torch.stack([test_dataset[i]["bio_features"] for i in explain_A_indices]).to(device)

    test_B_input_ids = torch.stack([test_dataset[i]["input_ids"] for i in explain_B_indices]).to(device)
    test_B_attention_mask = torch.stack([test_dataset[i]["attention_mask"] for i in explain_B_indices]).to(device)
    test_B_shapes = torch.stack([test_dataset[i]["shape_features"] for i in explain_B_indices]).to(device)
    test_B_bio = torch.stack([test_dataset[i]["bio_features"] for i in explain_B_indices]).to(device)

    with torch.no_grad():
        test_A_embeddings = model._get_bert_features(test_A_input_ids, test_A_attention_mask)
        test_B_embeddings = model._get_bert_features(test_B_input_ids, test_B_attention_mask)

    # Run SHAP GradientExplainer
    print("Running SHAP GradientExplainer...")
    import shap
    wrapper = ModelEmbeddingWrapper(model)
    explainer = shap.GradientExplainer(wrapper, [bg_embeddings, bg_shapes, bg_bio])
    
    shap_values_A = explainer.shap_values([test_A_embeddings, test_A_shapes, test_A_bio])
    shap_values_B = explainer.shap_values([test_B_embeddings, test_B_shapes, test_B_bio])

    # Dynamic extraction of SHAP values
    if isinstance(shap_values_A, list) and len(shap_values_A) == 3:
        seq_shap_A = shap_values_A[0]
        shape_shap_A = shap_values_A[1]
        bio_shap_A = shap_values_A[2]
        if seq_shap_A.ndim > 3: seq_shap_A = seq_shap_A[..., target_class]
        if shape_shap_A.ndim > 3: shape_shap_A = shape_shap_A[..., target_class]
        if bio_shap_A.ndim > 2: bio_shap_A = bio_shap_A[..., target_class]
    else:
        seq_shap_A = shap_values_A[target_class][0]
        shape_shap_A = shap_values_A[target_class][1]
        bio_shap_A = shap_values_A[target_class][2]

    if isinstance(shap_values_B, list) and len(shap_values_B) == 3:
        seq_shap_B = shap_values_B[0]
        shape_shap_B = shap_values_B[1]
        bio_shap_B = shap_values_B[2]
        if seq_shap_B.ndim > 3: seq_shap_B = seq_shap_B[..., target_class]
        if shape_shap_B.ndim > 3: shape_shap_B = shape_shap_B[..., target_class]
        if bio_shap_B.ndim > 2: bio_shap_B = bio_shap_B[..., target_class]
    else:
        seq_shap_B = shap_values_B[target_class][0]
        shape_shap_B = shap_values_B[target_class][1]
        bio_shap_B = shap_values_B[target_class][2]

    # Plot DNAshape Feature Importance
    importance_shape_A = np.abs(shape_shap_A).sum(axis=2).mean(axis=0)
    importance_shape_B = np.abs(shape_shap_B).sum(axis=2).mean(axis=0)
    features = ["MGW", "ProT", "Roll", "HelT", "EP"]
    x = np.arange(len(features))
    width = 0.35
    
    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, importance_shape_A, width, label=f"Subset A (Correct {target_name})", color="#4CAF50")
    plt.bar(x + width/2, importance_shape_B, width, label="Subset B (Confused)", color="#F44336")
    plt.xticks(x, features)
    plt.ylabel("Mean Absolute SHAP Value")
    plt.title(f"DNAshape Feature Importance (Subset A vs Subset B) for {target_name}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    bar_path = os.path.join(args.output_dir, f"gcmab_mscnn_shap_dnashape_bar_w{args.window_size}_{target_name}.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"Saved: {bar_path}")

    # Map sequence tokens back to characters
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
                if not clean_token: continue
                
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

    if args.absolute:
        print("Averaging SHAP values over absolute sequence positions (no GC-box alignment)...")
        avg_aligned_A = np.mean(char_shap_A, axis=0)
        avg_aligned_B = np.mean(char_shap_B, axis=0)
        
        plt.figure(figsize=(12, 3))
        heatmap_data = np.vstack([avg_aligned_A, avg_aligned_B])
        sns.heatmap(heatmap_data, annot=False, cmap="viridis",
                    yticklabels=["Subset A (Correct)", "Subset B (Confused)"],
                    xticklabels=[str(i) for i in range(len(avg_aligned_A))])
        plt.xlabel("Absolute Nucleotide Position (0 to 100)")
        plt.title(f"Sequence Context SHAP Importance (Absolute Positions) for {target_name}")
        plt.tight_layout()
        h_path = os.path.join(args.output_dir, f"gcmab_mscnn_shap_sequence_heatmap_absolute_{target_name}.png")
        plt.savefig(h_path, dpi=150)
        plt.close()
        print(f"Saved: {h_path}")
        
        plt.figure(figsize=(10, 4))
        plt.plot(range(len(avg_aligned_A)), avg_aligned_A, label=f"Subset A (Correct {target_name})", color="#4CAF50", linewidth=2)
        plt.plot(range(len(avg_aligned_B)), avg_aligned_B, label="Subset B (Confused)", color="#F44336", linewidth=2)
        plt.xlabel("Absolute Nucleotide Position (0 to 100)")
        plt.ylabel("Average SHAP Importance (L2 Norm)")
        plt.title(f"Sequence Context SHAP Importance (Absolute Positions) for {target_name}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        l_path = os.path.join(args.output_dir, f"gcmab_mscnn_shap_sequence_line_absolute_{target_name}.png")
        plt.savefig(l_path, dpi=150)
        plt.close()
        print(f"Saved: {l_path}")

    else:
        # Aligned around GC-box center
        def align_and_average_shap(seq_list, char_shaps_list, window_size):
            aligned_list = []
            for seq, char_s in zip(seq_list, char_shaps_list):
                match = re.search(r'GGGCGG|CCGCCC', seq, re.IGNORECASE)
                if not match: continue
                start, end = match.span()
                center = (start + end) // 2
                
                aligned = np.zeros(2 * window_size + 1)
                for i, offset in enumerate(range(-window_size, window_size + 1)):
                    abs_pos = center + offset
                    if 0 <= abs_pos < len(char_s):
                        aligned[i] = char_s[abs_pos]
                aligned_list.append(aligned)
            if len(aligned_list) == 0: return None
            return np.mean(aligned_list, axis=0)

        print(f"Averaging SHAP values aligned around GC-box with half-window size = {args.window_size}...")
        avg_aligned_A = align_and_average_shap(seq_A_list, char_shap_A, args.window_size)
        avg_aligned_B = align_and_average_shap(seq_B_list, char_shap_B, args.window_size)

        if avg_aligned_A is not None and avg_aligned_B is not None:
            heatmap_data = np.vstack([avg_aligned_A, avg_aligned_B])
            
            plt.figure(figsize=(12, 3))
            x_labels = [str(x) for x in range(-args.window_size, args.window_size + 1)]
            sns.heatmap(heatmap_data, annot=False, cmap="viridis",
                        yticklabels=["Subset A (Correct)", "Subset B (Confused)"],
                        xticklabels=x_labels)
            plt.xlabel("Position Relative to GC-box Center (bp)")
            plt.title(f"Sequence Context SHAP Alignment ({target_name} GC-box Flank Analysis, w={args.window_size})")
            plt.tight_layout()
            h_path = os.path.join(args.output_dir, f"gcmab_mscnn_shap_sequence_heatmap_w{args.window_size}_{target_name}.png")
            plt.savefig(h_path, dpi=150)
            plt.close()
            print(f"Saved: {h_path}")
            
            plt.figure(figsize=(10, 4))
            offsets = np.arange(-args.window_size, args.window_size + 1)
            plt.plot(offsets, avg_aligned_A, label=f"Subset A (Correct {target_name})", color="#4CAF50", linewidth=2)
            plt.plot(offsets, avg_aligned_B, label="Subset B (Confused)", color="#F44336", linewidth=2)
            plt.axvline(x=0, color="gray", linestyle="--", alpha=0.7, label="GC-box Center")
            plt.xlabel("Position Relative to GC-box Center (bp)")
            plt.ylabel("Average SHAP Importance (L2 Norm)")
            plt.title(f"Sequence Context SHAP Importance ({target_name} Flank Analysis, w={args.window_size})")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout()
            l_path = os.path.join(args.output_dir, f"gcmab_mscnn_shap_sequence_line_w{args.window_size}_{target_name}.png")
            plt.savefig(l_path, dpi=150)
            plt.close()
            print(f"Saved: {l_path}")
        else:
            print("Warning: Could not align any sequences (no GC-box found).")

    print("\nSHAP custom analysis completed successfully!")

if __name__ == "__main__":
    main()
