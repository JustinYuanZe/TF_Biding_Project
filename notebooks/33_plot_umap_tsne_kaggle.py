"""
Script 33 — t-SNE / UMAP visualisation of the Tri-Branch G-CMAB fusion embeddings.

This is a *faithful* re-implementation: the model classes, the architecture
hyper-parameters, the DNABERT-2 loading path and the DNAshape normalisation are
copied verbatim from the proven trainer (script 28), so the trained checkpoint
loads with NO key/shape mismatch. It then extracts the fused feature vector
(seq-msCNN ++ shape-msCNN ++ bio, dim = (4+4)*256 + 32 = 2080) for a balanced
subset and projects it to 2-D with both UMAP and t-SNE.

Two colourings are produced for each method:
  * 4-class  (SP1 / SP2 / SP4 / Negative)  -> shows the paralog overlap story
  * binary   (Positive vs Negative)        -> shows the separability the model learnt

Dataset: the *pure genomic* FINAL folders only
  FASTA : negative_genomic_matched.fasta + sp{1,2,4}_positive_final.fasta
  SHAPE : dnashape_negative_genomic.npy  + dnashape_sp{1,2,4}.npy
(No dinuc/cpg files are present, so the negative seq<->shape pairing cannot mismatch.)
"""

import os
import sys
import gc
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

# ═══════════════════════════════════════════════════════════════════════
# CONFIG  (architecture values MUST equal script 28 / the trained checkpoint)
# ═══════════════════════════════════════════════════════════════════════


def auto_detect_dir(anchor_file, local_default):
    """Find the directory that contains `anchor_file` (Kaggle input or local)."""
    if os.path.exists(os.path.join(local_default, anchor_file)):
        return local_default
    search_roots = ["/kaggle/input", "/kaggle/working", local_default, "data/processed", "."]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            if anchor_file in files:
                return dirpath
    return local_default


def find_file(name, path):
    if path and os.path.isfile(os.path.join(path, name)):
        return os.path.join(path, name)
    for root in ([path] if path else []) + ["/kaggle/input", "/kaggle/working", "data/processed", "."]:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            if name in files:
                return os.path.join(dirpath, name)
    return None


class CFG:
    FASTA_DIR = auto_detect_dir("sp1_positive_final.fasta", "data/processed/FINAL/datas1")
    SHAPE_DIR = auto_detect_dir("dnashape_sp1.npy", "data/processed/FINAL/datashape")
    OUT_DIR = "/kaggle/working/outputs_figures" if os.path.isdir("/kaggle/working") else "outputs_figures"

    # Checkpoint candidates, best match first (best_tribranch_shap.pt is THIS arch's own output)
    CKPT_CANDIDATES = ["best_tribranch_shap.pt", "best_gcmab_binary.pt", "best_seed42.pt"]

    # ── DNABERT-2 ──
    MODEL_NAME = "zhihan1996/DNABERT-2-117M"
    EMBEDDING_DIM = 768
    AUTO_MAX_LENGTH = True
    MAX_TOKEN_LENGTH = 48
    MAX_LENGTH_CAP = 96
    MAX_LENGTH_FLOOR = 32

    # ── Cross-modal attention ──
    USE_CROSS_ATTN = True
    CROSS_ATTN_D_MODEL = 128
    CROSS_ATTN_NHEAD = 4
    CROSS_ATTN_LAYERS = 1
    CROSS_ATTN_DROPOUT = 0.1

    # ── DNAshape branch ──
    SHAPE_CHANNELS = 5
    SHAPE_SEQ_LEN = 101

    # ── msCNN (CRITICAL: must match checkpoint) ──
    MSCNN_OUT_CHANNELS = 256
    SEQ_MSCNN_KERNELS = [7, 9, 11, 15]
    SHAPE_MSCNN_KERNELS = [4, 8, 12, 16]

    # ── Flags ──
    USE_GROUPNORM = True
    USE_LAYER_ATTN = True
    LAYER_ATTN_N = 6

    # ── Classifier head ──
    HIDDEN_DIM = 256
    FUSION_DROPOUT = 0.7

    # ── Sampling / runtime ──
    MAX_PER_CLASS = 800          # balanced subset per class for the projection
    BATCH_SIZE = 64
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


os.makedirs(CFG.OUT_DIR, exist_ok=True)
torch.manual_seed(CFG.SEED)
np.random.seed(CFG.SEED)

# ═══════════════════════════════════════════════════════════════════════
# DNABERT-2 flash-attention patch (pure-PyTorch; avoids Triton crash on T4)
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
    print(f"  Patched {patched} flash-attention refs -> pure PyTorch." if patched
          else "  No flash-attn refs to patch (will patch after first forward if needed).")


def load_dnabert2(model_name):
    print("\n" + "=" * 60)
    print("LOADING DNABERT-2 BACKBONE")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 3
    if CFG.USE_LAYER_ATTN:
        config.output_hidden_states = True

    model = None
    try:
        model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True, low_cpu_mem_usage=False)
        for name, param in model.named_parameters():
            if param.device == torch.device("meta"):
                raise RuntimeError(f"Parameter {name} on meta device")
        print("  Strategy 1 (direct load) OK")
    except Exception as e:
        print(f"  Strategy 1 failed: {e}")
        model = None
    if model is None:
        _orig = torch.empty

        def _patched(*a, **kw):
            if kw.get("device") == torch.device("meta") or str(kw.get("device", "")) == "meta":
                kw["device"] = "cpu"
            return _orig(*a, **kw)
        torch.empty = _patched
        try:
            model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True, low_cpu_mem_usage=False)
            print("  Strategy 2 (meta->cpu patch) OK")
        finally:
            torch.empty = _orig

    patch_flash_attention()
    for param in model.parameters():
        param.requires_grad = False
    print(f"  Encoder layers: {len(model.encoder.layer)} (all frozen for inference)")
    return model, tokenizer

# ═══════════════════════════════════════════════════════════════════════
# ARCHITECTURE — copied verbatim from script 28 (+ extract_features)
# ═══════════════════════════════════════════════════════════════════════


class LayerAttention(nn.Module):
    """ELMo-style scalar-mix over BERT hidden states (N learnable weights + gamma)."""
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
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
        self.norm = nn.GroupNorm(num_groups=min(16, out_channels), num_channels=out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.pointwise(self.depthwise(x))))


class MSCNNBranchStack(nn.Module):
    def __init__(self, in_channels, out_channels, kernels):
        super().__init__()
        self.branches = nn.ModuleList([
            DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size=k) for k in kernels
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
        self.cross_attn_seq2shape = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_seq = nn.LayerNorm(d_model)
        self.ffn_seq = FeedForward(d_model, expansion=4, dropout=dropout)
        self.cross_attn_shape2seq = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_shape = nn.LayerNorm(d_model)
        self.ffn_shape = FeedForward(d_model, expansion=4, dropout=dropout)

    def forward(self, seq_features, shape_features, seq_key_padding_mask=None):
        attended_seq, _ = self.cross_attn_seq2shape(query=seq_features, key=shape_features, value=shape_features)
        seq_out = self.ffn_seq(self.norm_seq(seq_features + attended_seq))
        attended_shape, _ = self.cross_attn_shape2seq(query=shape_features, key=seq_features, value=seq_features,
                                                      key_padding_mask=seq_key_padding_mask)
        shape_out = self.ffn_shape(self.norm_shape(shape_features + attended_shape))
        return seq_out, shape_out


class TriBranchClassifier(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        d_model = cfg.CROSS_ATTN_D_MODEL
        self.use_cross_attn = getattr(cfg, "USE_CROSS_ATTN", True)

        self.bio_branch = nn.Sequential(
            nn.Linear(3, 16), nn.BatchNorm1d(16), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(16, 32), nn.GELU(),
        )

        self.use_layer_attn = cfg.USE_LAYER_ATTN
        if self.use_layer_attn:
            self.layer_attention = LayerAttention(n_layers=cfg.LAYER_ATTN_N)
            self._layer_attn_fallback = False
        else:
            self.layer_attention = None

        self.seq_projection = nn.Sequential(
            nn.Linear(cfg.EMBEDDING_DIM, d_model), nn.LayerNorm(d_model), nn.GELU(),
        )

        if self.use_cross_attn:
            self.shape_projection = nn.Sequential(
                nn.Conv1d(cfg.SHAPE_CHANNELS, d_model, kernel_size=1),
                nn.GroupNorm(min(16, d_model), d_model), nn.GELU(),
            )
            self.cross_attention_layers = nn.ModuleList([
                CrossModalAttentionLayer(d_model, cfg.CROSS_ATTN_NHEAD, cfg.CROSS_ATTN_DROPOUT)
                for _ in range(cfg.CROSS_ATTN_LAYERS)
            ])
            self.seq_mscnn = MSCNNBranchStack(d_model, cfg.MSCNN_OUT_CHANNELS, cfg.SEQ_MSCNN_KERNELS)
            self.shape_mscnn = MSCNNBranchStack(d_model, cfg.MSCNN_OUT_CHANNELS, cfg.SHAPE_MSCNN_KERNELS)
        else:
            self.seq_mscnn = MSCNNBranchStack(d_model, cfg.MSCNN_OUT_CHANNELS, cfg.SEQ_MSCNN_KERNELS)
            self.shape_mscnn = MSCNNBranchStack(cfg.SHAPE_CHANNELS, cfg.MSCNN_OUT_CHANNELS, cfg.SHAPE_MSCNN_KERNELS)

        in_features = (len(cfg.SEQ_MSCNN_KERNELS) + len(cfg.SHAPE_MSCNN_KERNELS)) * cfg.MSCNN_OUT_CHANNELS + 32
        self.classifier = nn.Sequential(
            nn.Linear(in_features, cfg.HIDDEN_DIM), nn.GELU(),
            nn.Dropout(cfg.FUSION_DROPOUT), nn.Linear(cfg.HIDDEN_DIM, 1),
        )

    def _get_bert_features(self, input_ids, attention_mask):
        if self.use_layer_attn and not getattr(self, '_layer_attn_fallback', False):
            try:
                outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                                        output_hidden_states=True)
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    all_hidden = outputs.hidden_states
                    n = min(self.cfg.LAYER_ATTN_N, len(all_hidden) - 1)
                    return self.layer_attention(list(all_hidden[-n:]))
                elif isinstance(outputs, tuple) and len(outputs) > 2 and isinstance(outputs[2], (tuple, list)):
                    all_hidden = outputs[2]
                    n = min(self.cfg.LAYER_ATTN_N, len(all_hidden) - 1)
                    return self.layer_attention(list(all_hidden[-n:]))
                else:
                    self._layer_attn_fallback = True
            except Exception as e:
                print(f"  [LayerAttn] output_hidden_states failed ({e}); using hook fallback.")
                self._layer_attn_fallback = True

        if self.use_layer_attn and getattr(self, '_layer_attn_fallback', False):
            hidden_states_collected, hooks = [], []
            n = min(self.cfg.LAYER_ATTN_N, len(self.backbone.encoder.layer))
            start_layer = len(self.backbone.encoder.layer) - n

            def make_hook(storage):
                def hook_fn(module, inp, output):
                    storage.append(output[0] if isinstance(output, tuple) else output)
                return hook_fn

            for i in range(start_layer, len(self.backbone.encoder.layer)):
                hooks.append(self.backbone.encoder.layer[i].register_forward_hook(make_hook(hidden_states_collected)))
            _ = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            for h in hooks:
                h.remove()
            mixed = self.layer_attention(hidden_states_collected)
            if mixed.dim() == 2:
                B, T, D = attention_mask.size(0), attention_mask.size(1), mixed.size(-1)
                padded = torch.zeros(B, T, D, dtype=mixed.dtype, device=mixed.device)
                padded[attention_mask.bool()] = mixed
                mixed = padded
            return mixed

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        elif hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs

    def _trunk(self, seq_embeddings, shape_features, bio_features, attention_mask):
        """Shared trunk -> fused vector (the visualised embedding)."""
        seq_features = self.seq_projection(seq_embeddings)
        if self.use_cross_attn:
            shape_proj = self.shape_projection(shape_features)
            shape_feats = shape_proj.transpose(1, 2)
            seq_key_padding_mask = (attention_mask == 0)
            for cross_layer in self.cross_attention_layers:
                seq_features, shape_feats = cross_layer(seq_features, shape_feats,
                                                        seq_key_padding_mask=seq_key_padding_mask)
            seq_in = seq_features.transpose(1, 2)
            shape_in = shape_feats.transpose(1, 2)
        else:
            seq_in = seq_features.transpose(1, 2)
            shape_in = shape_features
        seq_pooled = self.seq_mscnn(seq_in)
        shape_pooled = self.shape_mscnn(shape_in)
        bio_out = self.bio_branch(bio_features)
        fused = torch.cat([seq_pooled, shape_pooled, bio_out], dim=1)
        return fused

    def extract_features(self, input_ids, attention_mask, shape_features, bio_features):
        hidden_states = self._get_bert_features(input_ids, attention_mask)
        return self._trunk(hidden_states, shape_features, bio_features, attention_mask)

    def forward(self, input_ids, attention_mask, shape_features, bio_features):
        fused = self.extract_features(input_ids, attention_mask, shape_features, bio_features)
        return self.classifier(fused).squeeze(-1)

# ═══════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════

import re as regex


class TriBranchDataset(Dataset):
    def __init__(self, sequences, labels, true_class, shape_features, tokenizer, max_length):
        self.sequences = sequences
        self.labels = labels
        self.true_class = true_class
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
                features.append([0.0, 0.0, 0.0]); continue
            c = seq.count('C'); g = seq.count('G'); cg = seq.count('CG')
            cpg_oe = (cg * L) / (c * g) if (c * g) > 0 else 0.0
            gc_content = (c + g) / L
            g4 = 1.0 if self.g4_pattern.search(seq) else 0.0
            features.append([cpg_oe, gc_content, g4])
        return np.array(features, dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.sequences[idx], padding="max_length", truncation=True,
                             max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "shape_features": torch.tensor(self.shape_features[idx], dtype=torch.float32),
            "bio_features": torch.tensor(self.bio_features[idx], dtype=torch.float32),
            "true_class": self.true_class[idx],
        }


def load_fasta(filepath):
    sequences = []
    with open(filepath, "r") as f:
        seq_lines = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_lines:
                    sequences.append("".join(seq_lines).upper()); seq_lines = []
            else:
                seq_lines.append(line)
        if seq_lines:
            sequences.append("".join(seq_lines).upper())
    return sequences


def robust_normalize_shapes(shapes):
    """Robust scaler (median-center, divide by P1-P99 spread) — same FORM as training."""
    n_channels = shapes.shape[1]
    out = np.copy(shapes).astype(np.float32)
    names = ["MGW", "ProT", "Roll", "HelT", "EP"]
    for ch in range(n_channels):
        vals = shapes[:, ch, :].flatten()
        valid = vals[~np.isnan(vals)]
        median_val = np.median(valid)
        p1, p99 = np.percentile(valid, 1), np.percentile(valid, 99)
        scale = max(p99 - p1, 1e-9)
        out[:, ch, :] = (out[:, ch, :] - median_val) / scale
        print(f"  {names[ch]:>5s}: median={median_val:>8.4f} P1={p1:>8.4f} P99={p99:>8.4f}")
    n_nan = int(np.isnan(out).sum())
    out = np.nan_to_num(out, nan=0.0)
    print(f"  NaN filled with 0: {n_nan}")
    return out


def compute_max_token_length(sequences, tok, sample_size=2000, percentile=99, floor=32, cap=96):
    import random
    sample = sequences if len(sequences) <= sample_size else random.sample(sequences, sample_size)
    lengths = [len(tok(s, add_special_tokens=True)["input_ids"]) for s in sample]
    p_val = int(np.percentile(lengths, percentile))
    chosen = int(max(floor, min(cap, p_val + 2)))
    print(f"  AUTO MAX TOKEN LENGTH: p{percentile}={p_val} -> chosen={chosen}")
    return chosen


def load_checkpoint_into(model):
    ckpt_path = None
    for cand in CFG.CKPT_CANDIDATES:
        ckpt_path = find_file(cand, None)
        if ckpt_path:
            break
    if not ckpt_path:
        print(f"  [WARN] No checkpoint found among {CFG.CKPT_CANDIDATES}. "
              f"Embeddings will be from the UNTRAINED head -> plots are meaningless.")
        return False
    print(f"  Loading checkpoint: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu")
    sd = obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj
    sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
    result = model.load_state_dict(sd, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith("backbone.")]
    unexpected = [k for k in result.unexpected_keys if not k.startswith("backbone.")]
    # backbone keys are expected to fully match (same DNABERT-2). Head must match.
    head_missing = [k for k in missing if any(p in k for p in
                    ["seq_mscnn", "shape_mscnn", "classifier", "cross_attention", "layer_attention", "bio_branch"])]
    if head_missing:
        raise RuntimeError(f"Checkpoint arch MISMATCH — missing head keys: {head_missing[:8]} ...")
    print(f"  load_state_dict OK (non-backbone missing={len(missing)}, unexpected={len(unexpected)})")
    return True

# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════


def main():
    print(f"[FASTA_DIR] {CFG.FASTA_DIR}")
    print(f"[SHAPE_DIR] {CFG.SHAPE_DIR}")
    print(f"[DEVICE]    {CFG.DEVICE}")

    backbone, tokenizer = load_dnabert2(CFG.MODEL_NAME)
    model = TriBranchClassifier(backbone, CFG)
    load_checkpoint_into(model)
    model.to(CFG.DEVICE).eval()

    # ---- load pure-genomic data (paired negative shape) ----
    NEG_FASTA = "negative_genomic_matched.fasta"
    NEG_SHAPE = "dnashape_negative_genomic.npy"
    sources = [
        ("SP1", "sp1_positive_final.fasta", "dnashape_sp1.npy"),
        ("SP2", "sp2_positive_final.fasta", "dnashape_sp2.npy"),
        ("SP4", "sp4_positive_final.fasta", "dnashape_sp4.npy"),
        ("Negative", NEG_FASTA, NEG_SHAPE),
    ]

    all_seqs, all_true, shape_chunks = [], [], []
    for cls_name, fa, npy in sources:
        fpath = find_file(fa, CFG.FASTA_DIR)
        spath = find_file(npy, CFG.SHAPE_DIR)
        if not fpath:
            raise FileNotFoundError(f"FASTA not found: {fa}")
        if not spath:
            raise FileNotFoundError(f"DNAshape not found: {npy} (paired with {fa})")
        seqs = load_fasta(fpath)
        shapes = np.load(spath)
        n = min(len(seqs), shapes.shape[0], CFG.MAX_PER_CLASS)
        seqs, shapes = seqs[:n], shapes[:n]
        all_seqs.extend(seqs)
        all_true.extend([cls_name] * n)
        shape_chunks.append(shapes)
        print(f"  {cls_name:9s}: {n} samples  (fasta={os.path.basename(fpath)}, shape={os.path.basename(spath)})")

    all_shapes = np.concatenate(shape_chunks, axis=0)
    print("\nNormalising DNAshape (robust P1-P99):")
    all_shapes = robust_normalize_shapes(all_shapes)

    max_len = (compute_max_token_length(all_seqs, tokenizer,
                                        floor=CFG.MAX_LENGTH_FLOOR, cap=CFG.MAX_LENGTH_CAP)
               if CFG.AUTO_MAX_LENGTH else CFG.MAX_TOKEN_LENGTH)

    dummy_labels = [1 if c != "Negative" else 0 for c in all_true]
    dataset = TriBranchDataset(all_seqs, dummy_labels, all_true, all_shapes, tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=CFG.BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

    print("\nExtracting fused features ...")
    feats, true_classes = [], []
    with torch.no_grad():
        for batch in tqdm(loader):
            fused = model.extract_features(
                batch["input_ids"].to(CFG.DEVICE),
                batch["attention_mask"].to(CFG.DEVICE),
                batch["shape_features"].to(CFG.DEVICE),
                batch["bio_features"].to(CFG.DEVICE),
            )
            feats.append(fused.float().cpu().numpy())
            true_classes.extend(batch["true_class"])
    feats = np.concatenate(feats, axis=0)
    print(f"  Fused feature matrix: {feats.shape}  (expected dim "
          f"{(len(CFG.SEQ_MSCNN_KERNELS)+len(CFG.SHAPE_MSCNN_KERNELS))*CFG.MSCNN_OUT_CHANNELS + 32})")

    true_classes = list(true_classes)
    binary_classes = ["Positive" if c != "Negative" else "Negative" for c in true_classes]

    del model, backbone
    gc.collect()
    if CFG.DEVICE == "cuda":
        torch.cuda.empty_cache()

    _project_and_plot(feats, true_classes, binary_classes)


def _scatter(emb, classes, title, out_path, palette):
    plt.figure(figsize=(9, 7.5))
    if _HAS_SNS:
        sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=classes,
                        palette=palette, s=14, alpha=0.8, edgecolor="none")
    else:
        for cls in sorted(set(classes)):
            m = np.array([c == cls for c in classes])
            plt.scatter(emb[m, 0], emb[m, 1], s=14, alpha=0.8, label=cls)
    plt.legend(title="", markerscale=2, frameon=True)
    plt.title(title)
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def _project_and_plot(feats, true_classes, binary_classes):
    pal4 = {"SP1": "#e41a1c", "SP2": "#377eb8", "SP4": "#4daf4a", "Negative": "#999999"}
    pal2 = {"Positive": "#d7191c", "Negative": "#2c7bb6"}

    # ---- UMAP ----
    try:
        import umap
        print("\nUMAP ...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=CFG.SEED)
        emb = reducer.fit_transform(feats)
        _scatter(emb, true_classes, "UMAP — G-CMAB fusion embedding (4-class)",
                 os.path.join(CFG.OUT_DIR, "umap_4class.png"), pal4)
        _scatter(emb, binary_classes, "UMAP — G-CMAB fusion embedding (Positive vs Negative)",
                 os.path.join(CFG.OUT_DIR, "umap_binary.png"), pal2)
    except Exception as e:
        print(f"  UMAP failed: {e}")

    # ---- t-SNE ----
    try:
        from sklearn.manifold import TSNE
        print("\nt-SNE ...")
        perp = min(30, max(5, (len(feats) - 1) // 3))
        tsne = TSNE(n_components=2, random_state=CFG.SEED, perplexity=perp, init="pca")
        emb = tsne.fit_transform(feats)
        _scatter(emb, true_classes, "t-SNE — G-CMAB fusion embedding (4-class)",
                 os.path.join(CFG.OUT_DIR, "tsne_4class.png"), pal4)
        _scatter(emb, binary_classes, "t-SNE — G-CMAB fusion embedding (Positive vs Negative)",
                 os.path.join(CFG.OUT_DIR, "tsne_binary.png"), pal2)
    except Exception as e:
        print(f"  t-SNE failed: {e}")


if __name__ == "__main__":
    main()
