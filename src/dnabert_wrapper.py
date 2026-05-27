import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig


# ──────────────────────────────────────────────────────────────────────
# Pure-PyTorch replacement for the Triton-based flash attention kernel.
# DNABERT-2 ships its own flash_attn_triton.py which uses a deprecated
# Triton API (`tl.dot(..., trans_b=True)`) that breaks on newer Triton
# versions (≥ 3.x) commonly found on Kaggle/Colab.
# ──────────────────────────────────────────────────────────────────────

def _pytorch_flash_attn_qkvpacked(qkv, bias=None, causal=False, softmax_scale=None):
    """
    Drop-in replacement for FlashAttnQKVPackedFunc.apply().

    Args:
        qkv:  (batch, seqlen, 3, nheads, headdim)
        bias: None or (batch, nheads, seqlen, seqlen)
        causal: bool
        softmax_scale: float or None
    Returns:
        out:  (batch, seqlen, nheads, headdim)
    """
    q, k, v = qkv.unbind(dim=2)            # each: (B, S, H, D)
    q = q.transpose(1, 2)                   # (B, H, S, D)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    d = q.shape[-1]
    scale = softmax_scale if softmax_scale is not None else (d ** -0.5)

    attn = torch.matmul(q, k.transpose(-2, -1)) * scale        # (B, H, S, S)

    if bias is not None:
        attn = attn + bias

    if causal:
        S = q.shape[2]
        mask = torch.triu(torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float("-inf"))

    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(attn, v)             # (B, H, S, D)
    return out.transpose(1, 2).contiguous()  # (B, S, H, D)


def _pytorch_flash_attn_kvpacked(q, kv, bias=None, causal=False, softmax_scale=None):
    """
    Drop-in replacement for FlashAttnKVPackedFunc.apply().

    Args:
        q:   (batch, seqlen_q, nheads, headdim)
        kv:  (batch, seqlen_k, 2, nheads, headdim)
        bias: None or (batch, nheads, seqlen_q, seqlen_k)
    Returns:
        out: (batch, seqlen_q, nheads, headdim)
    """
    k, v = kv.unbind(dim=2)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

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
    """
    Drop-in replacement for FlashAttnFunc.apply().
    """
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

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


def _patch_flash_attention():
    """
    Monkey-patch ALL cached HuggingFace modules that reference DNABERT-2's
    flash_attn_triton functions so they use our pure-PyTorch replacements.
    Must be called AFTER the model class has been imported / instantiated.
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
        # Patch both the triton module itself and any module that imported from it
        if "flash_attn_triton" in mod_name or "bert_layers" in mod_name:
            for attr_name, replacement in targets.items():
                if hasattr(mod, attr_name):
                    setattr(mod, attr_name, replacement)
                    patched += 1

    if patched:
        print(f"  ✅ Patched {patched} flash-attention references → pure PyTorch (Triton-free).")
    else:
        print("  ⚠️  No flash-attention references found to patch.")


# ──────────────────────────────────────────────────────────────────────
# Device safety probe
# ──────────────────────────────────────────────────────────────────────

def _get_safe_device(requested_device=None):
    """
    Determine a safe torch device.
    Probes CUDA with a tiny tensor to catch incompatible GPU architectures
    (e.g., Tesla P100 sm_60 with PyTorch compiled for sm_70+).
    """
    if requested_device is not None:
        dev = torch.device(requested_device)
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dev.type == "cuda" and torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device=dev)
            return dev
        except Exception as e:
            print(f"⚠️  CUDA probe failed ({e}). Falling back to CPU.")
            return torch.device("cpu")
    return dev


# ──────────────────────────────────────────────────────────────────────
# Main wrapper class
# ──────────────────────────────────────────────────────────────────────

class DNABERTWrapper:
    def __init__(self, model_name="zhihan1996/DNABERT-2-117M", trust_remote_code=True, device=None):
        """
        Initialize the DNABERT-2 model and BPE tokenizer.

        Handles two known compatibility issues:
        1. 'meta device' RuntimeError in ALiBi layers  → multi-strategy loading
        2. Triton CompilationError (trans_b removed)    → PyTorch flash-attn patch
        """
        self.device = _get_safe_device(device)

        print(f"Loading DNABERT-2 model and tokenizer on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)

        # Load config and ensure pad_token_id is set
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
            config.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 3

        # ── Robust model loading (3 strategies) ──
        model = None

        # --- Strategy 1: Direct load ---
        try:
            model = AutoModel.from_pretrained(
                model_name, config=config,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=False,
            )
            for name, param in model.named_parameters():
                if param.device == torch.device("meta"):
                    raise RuntimeError(f"Parameter {name} still on meta device")
            for name, buf in model.named_buffers():
                if buf.device == torch.device("meta"):
                    raise RuntimeError(f"Buffer {name} still on meta device")
            print("  Strategy 1 (direct load) succeeded.")
        except (RuntimeError, Exception) as e:
            print(f"  Strategy 1 failed: {e}")
            model = None

        # --- Strategy 2: Empty init + manual state_dict ---
        if model is None:
            try:
                print("  Trying Strategy 2 (empty init + state_dict reload)...")
                from huggingface_hub import hf_hub_download

                with torch.no_grad():
                    model = AutoModel.from_config(config, trust_remote_code=trust_remote_code)

                try:
                    weight_file = hf_hub_download(repo_id=model_name, filename="model.safetensors")
                    from safetensors.torch import load_file
                    state_dict = load_file(weight_file, device="cpu")
                except Exception:
                    weight_file = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
                    state_dict = torch.load(weight_file, map_location="cpu", weights_only=False)

                result = model.load_state_dict(state_dict, strict=False)
                if result.missing_keys:
                    print(f"  Missing keys (OK for ALiBi buffers): {result.missing_keys[:5]}...")
                model = model.to("cpu")
                print("  Strategy 2 succeeded.")
            except Exception as e:
                print(f"  Strategy 2 failed: {e}")
                model = None

        # --- Strategy 3: Monkey-patch torch.empty meta→cpu ---
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
                        trust_remote_code=trust_remote_code,
                        low_cpu_mem_usage=False,
                    )
                finally:
                    torch.empty = _orig_empty
                print("  Strategy 3 succeeded.")
            except Exception as e:
                print(f"  Strategy 3 failed: {e}")
                raise RuntimeError(
                    "All strategies failed to load DNABERT-2. "
                    "Please try: pip install --upgrade transformers torch safetensors"
                ) from e

        # ── Patch flash attention (Triton trans_b fix) ──
        _patch_flash_attention()

        self.model = model.to(self.device)
        self.model.eval()
        print(f"DNABERT-2 loaded successfully on {self.device}.")

    def get_embeddings(self, sequences, batch_size=64, max_length=105):
        """
        Extract token-level embeddings of shape (n_samples, seq_len, 768) from DNA sequences.
        Ensures a fixed sequence length by using padding='max_length'.
        """
        embeddings_list = []

        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting DNABERT-2 Embeddings"):
                batch_seqs = sequences[i:i + batch_size]

                inputs = self.tokenizer(
                    batch_seqs,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length
                ).to(self.device)

                outputs = self.model(**inputs)
                batch_embeddings = outputs[0].cpu().numpy().astype(np.float32)
                embeddings_list.append(batch_embeddings)

        return np.concatenate(embeddings_list, axis=0)

    def get_cls_embeddings(self, sequences, batch_size=64):
        """
        Extract CLS token embeddings of shape (n_samples, 768) from DNA sequences.
        Useful for simple classification or visualization (t-SNE).
        """
        cls_list = []
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting CLS Embeddings"):
                batch_seqs = sequences[i:i + batch_size]
                inputs = self.tokenizer(
                    batch_seqs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                outputs = self.model(**inputs)
                batch_cls = outputs[0][:, 0, :].cpu().numpy().astype(np.float32)
                cls_list.append(batch_cls)
        return np.concatenate(cls_list, axis=0)
