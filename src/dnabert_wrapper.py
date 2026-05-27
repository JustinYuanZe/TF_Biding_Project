import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig

class DNABERTWrapper:
    def __init__(self, model_name="zhihan1996/DNABERT-2-117M", trust_remote_code=True, device=None):
        """
        Initialize the DNABERT-2 model and BPE tokenizer.
        
        Uses a robust loading strategy to avoid the known 'meta device' RuntimeError
        in DNABERT-2's custom ALiBi attention layers (bert_layers.py).
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Loading DNABERT-2 model and tokenizer on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        
        # Load config and ensure pad_token_id is set
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
            config.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 3
        
        # ── Robust model loading ──
        # Strategy 1: Standard loading (works on most environments)
        # Strategy 2: Force all tensors off 'meta' device via state_dict reload
        # Strategy 3: Monkey-patch rebuild_alibi_tensor to fix device mismatch
        model = None
        
        # --- Strategy 1: Direct load with explicit settings ---
        try:
            model = AutoModel.from_pretrained(
                model_name,
                config=config,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=False,
            )
            # Verify no meta tensors remain
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
        
        # --- Strategy 2: Instantiate empty model, then load state dict manually ---
        if model is None:
            try:
                print("  Trying Strategy 2 (empty init + state_dict reload)...")
                from transformers import AutoModel as AM
                from huggingface_hub import hf_hub_download
                import json
                
                # Instantiate model on CPU with random init
                with torch.no_grad():
                    model = AM.from_config(config, trust_remote_code=trust_remote_code)
                
                # Download and load the state dict
                try:
                    weight_file = hf_hub_download(repo_id=model_name, filename="model.safetensors")
                    from safetensors.torch import load_file
                    state_dict = load_file(weight_file, device="cpu")
                except Exception:
                    weight_file = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
                    state_dict = torch.load(weight_file, map_location="cpu", weights_only=False)
                
                # Load state dict (allow missing/unexpected keys for ALiBi buffers)
                result = model.load_state_dict(state_dict, strict=False)
                if result.missing_keys:
                    print(f"  Missing keys (OK for ALiBi buffers): {result.missing_keys[:5]}...")
                
                # Force all buffers to CPU
                model = model.to("cpu")
                print("  Strategy 2 succeeded.")
            except Exception as e:
                print(f"  Strategy 2 failed: {e}")
                model = None
        
        # --- Strategy 3: Monkey-patch + reload ---
        if model is None:
            try:
                print("  Trying Strategy 3 (monkey-patch ALiBi)...")
                import importlib
                import sys
                
                # Load on CPU, patching any meta-device tensor creation
                _orig_empty = torch.empty
                def _patched_empty(*args, **kwargs):
                    if kwargs.get("device") == torch.device("meta") or str(kwargs.get("device", "")) == "meta":
                        kwargs["device"] = "cpu"
                    return _orig_empty(*args, **kwargs)
                
                torch.empty = _patched_empty
                try:
                    model = AutoModel.from_pretrained(
                        model_name,
                        config=config,
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
        
        self.model = model.to(self.device)
        self.model.eval()
        print(f"DNABERT-2 loaded successfully on {self.device}.")

    def get_embeddings(self, sequences, batch_size=64, max_length=105):
        """
        Extract token-level embeddings of shape (n_samples, seq_len, 768) from DNA sequences.
        Ensures a fixed sequence length by using padding='max_length'.
        """
        embeddings_list = []
        
        # Disable gradient computation for feature extraction
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting DNABERT-2 Embeddings"):
                batch_seqs = sequences[i:i + batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_seqs,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length
                ).to(self.device)
                
                # Pass through DNABERT-2
                outputs = self.model(**inputs)
                
                # outputs[0] contains the token-level hidden states (batch_size, seq_len, 768)
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
                # Take the CLS embedding (first token of the sequence)
                batch_cls = outputs[0][:, 0, :].cpu().numpy().astype(np.float32)
                cls_list.append(batch_cls)
        return np.concatenate(cls_list, axis=0)
