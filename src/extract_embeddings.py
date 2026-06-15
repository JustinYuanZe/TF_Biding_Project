"""
Extract DNABERT-2 embeddings for the TF binding dataset.
"""

import argparse
import os
import sys
from typing import List

import numpy as np

# Fix import path for direct script execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dnabert_wrapper import DNABERTWrapper

def load_fasta(path: str) -> List[str]:
    """Load sequences from a FASTA file."""
    seqs: List[str] = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('>'):
                seqs.append(line.upper())
    return seqs

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DNABERT-2 embeddings and save to disk as NumPy tensors")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for extraction")
    parser.add_argument("--max_length", type=int, default=105, help="Max sequence length")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory to save npy files")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define dataset paths
    sp1_path = os.path.join("data", "processed", "sp1_positive_final.fasta")
    sp2_path = os.path.join("data", "processed", "sp2_positive_final.fasta")
    sp4_path = os.path.join("data", "processed", "sp4_positive_final.fasta")
    neg_path = os.path.join("data", "processed", "negative_final.fasta")
    
    print("Loading FASTA files...")
    seqs_sp1 = load_fasta(sp1_path)
    seqs_sp2 = load_fasta(sp2_path)
    seqs_sp4 = load_fasta(sp4_path)
    seqs_neg = load_fasta(neg_path)
    
    sequences = seqs_sp1 + seqs_sp2 + seqs_sp4 + seqs_neg
    labels = np.concatenate([
        np.zeros(len(seqs_sp1)),
        np.ones(len(seqs_sp2)),
        np.full(len(seqs_sp4), 2),
        np.full(len(seqs_neg), 3)
    ], axis=0)
    
    print(f"Total sequences: {len(sequences)}")
    print(f"Labels distribution: {np.bincount(labels.astype(int))}")
    
    # Extract embeddings
    wrapper = DNABERTWrapper(device=args.device)
    print("\n--- Starting embedding extraction (float16) ---")
    embeddings = wrapper.get_embeddings(sequences, batch_size=args.batch_size, max_length=args.max_length, dtype=np.float16)
    print(f"Extraction complete. Shape: {embeddings.shape}")
    
    # Save to disk
    emb_path = os.path.join(args.output_dir, "dnabert_embeddings.npy")
    lbl_path = os.path.join(args.output_dir, "dnabert_labels.npy")
    
    print(f"Saving embeddings to {emb_path}...")
    np.save(emb_path, embeddings)
    
    print(f"Saving labels to {lbl_path}...")
    np.save(lbl_path, labels)
    
    print("Tensors saved successfully to disk.")

if __name__ == "__main__":
    main()
