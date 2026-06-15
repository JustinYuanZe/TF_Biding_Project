"""
Pipeline utilities for end-to-end DNA sequence modeling.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DNAPipelineDataset(Dataset):
    """
    PyTorch Dataset for on-the-fly tokenization of DNA sequences.
    Avoids storing heavy extracted embeddings in RAM.
    """
    def __init__(self, sequences: List[str], labels: List[int], tokenizer: Any, max_length: int = 105) -> None:
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        # Tokenize single sequence on CPU
        inputs = self.tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Squeeze batch dimension (1, seq_len) -> (seq_len)
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        return item, label


class DNABERT_mCNN(nn.Module):
    """
    End-to-end model combining DNABERT-2 and Multi-Scale CNN.
    Can run with frozen or unfrozen DNABERT-2 weights.
    """
    def __init__(self, dnabert_model: nn.Module, mcnn_model: nn.Module, freeze_dnabert: bool = True) -> None:
        super(DNABERT_mCNN, self).__init__()
        self.dnabert = dnabert_model
        self.mcnn = mcnn_model
        self.freeze_dnabert = freeze_dnabert
        
        if freeze_dnabert:
            print("Freezing DNABERT-2 weights (training only mCNN head)...")
            for param in self.dnabert.parameters():
                param.requires_grad = False
        else:
            print("Fine-tuning DNABERT-2 weights along with mCNN...")

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
        # Extract embeddings
        if self.freeze_dnabert:
            with torch.no_grad():
                outputs = self.dnabert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        else:
            outputs = self.dnabert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            
        # outputs[0] has shape: (batch_size, seq_len, 768)
        embeddings = outputs[0]
        
        # Pass embeddings to MultiScaleCNN
        logits = self.mcnn(embeddings)
        return logits

