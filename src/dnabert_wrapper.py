import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

class DNABERTWrapper:
    def __init__(self, model_name="zhihan1996/DNABERT-2-117M", trust_remote_code=True, device=None):
        """
        Initialize the DNABERT-2 model and BPE tokenizer.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Loading DNABERT-2 model and tokenizer on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code).to(self.device)
        self.model.eval()
        print("DNABERT-2 loaded successfully.")

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
