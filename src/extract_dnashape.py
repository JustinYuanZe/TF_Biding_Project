"""
Extract DNAshape features (MGW, ProT, Roll, HelT, EP) from DNA sequences
using a pre-computed pentamer lookup table.
"""

import os
from typing import List, Tuple

import numpy as np

from dnashape_lookup import DNASHAPE_LOOKUP

def revcomp(seq: str) -> str:
    comp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    return "".join(comp.get(base, base) for base in reversed(seq))

def load_fasta(file_path: str) -> Tuple[List[str], List[str]]:
    sequences: List[str] = []
    headers: List[str] = []
    with open(file_path, 'r') as f:
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append("".join(current_seq).upper())
                    current_seq = []
                headers.append(line)
            else:
                current_seq.append(line)
        if current_seq:
            sequences.append("".join(current_seq).upper())
    return headers, sequences

def compute_shape_features(sequence: str) -> np.ndarray:
    N = len(sequence)
    # Output arrays
    mgw_arr = np.full(N, np.nan)
    prot_arr = np.full(N, np.nan)
    ep_arr = np.full(N, np.nan)
    roll_arr = np.full(N, np.nan)
    helt_arr = np.full(N, np.nan)
    
    # 1. Nucleotide parameters: MGW, ProT, EP
    # Sliding window pentamer from index 2 to N-3 (central nucleotide index 2 to N-3)
    for j in range(2, N - 2):
        pentamer = sequence[j-2:j+3]
        if pentamer in DNASHAPE_LOOKUP:
            mgw_arr[j] = DNASHAPE_LOOKUP[pentamer][0]
            prot_arr[j] = DNASHAPE_LOOKUP[pentamer][1]
            ep_arr[j] = DNASHAPE_LOOKUP[pentamer][6]
        else:
            rc_pentamer = revcomp(pentamer)
            if rc_pentamer in DNASHAPE_LOOKUP:
                mgw_arr[j] = DNASHAPE_LOOKUP[rc_pentamer][0]
                prot_arr[j] = DNASHAPE_LOOKUP[rc_pentamer][1]
                ep_arr[j] = DNASHAPE_LOOKUP[rc_pentamer][6]
            else:
                # print(f"Warning: Pentamer {pentamer} not found in lookup")
                pass

    # 2. Base pair-step parameters: Roll, HelT (N-1 steps)
    # Step j is between base j and j+1. Range of valid step is 1 to N-3.
    # index 0 is NA. index N-2 is NA.
    for j in range(1, N - 2):
        # We have 2 overlapping pentamers:
        # Pentamer 1 at j-1 (corresponds to f1_index = j-1 in C++)
        # Pentamer 2 at j-2 (corresponds to f2_index = j-2 in C++)
        p1 = sequence[j-1:j+4]
        p2 = sequence[j-2:j+3]
        
        # Get values for Pentamer 1 (step parameter 1: roll1 or twist1)
        val1_roll, val1_helt = np.nan, np.nan
        if p1 in DNASHAPE_LOOKUP:
            val1_roll = DNASHAPE_LOOKUP[p1][2]   # roll1
            val1_helt = DNASHAPE_LOOKUP[p1][4]   # twist1
        else:
            rc_p1 = revcomp(p1)
            if rc_p1 in DNASHAPE_LOOKUP:
                # Complementary strand -> swap roll1 and roll2, twist1 and twist2
                val1_roll = DNASHAPE_LOOKUP[rc_p1][3]   # roll2
                val1_helt = DNASHAPE_LOOKUP[rc_p1][5]   # twist2
        
        # Get values for Pentamer 2 (step parameter 2: roll2 or twist2)
        val2_roll, val2_helt = np.nan, np.nan
        if p2 in DNASHAPE_LOOKUP:
            val2_roll = DNASHAPE_LOOKUP[p2][3]   # roll2
            val2_helt = DNASHAPE_LOOKUP[p2][5]   # twist2
        else:
            rc_p2 = revcomp(p2)
            if rc_p2 in DNASHAPE_LOOKUP:
                # Complementary strand -> swap
                val2_roll = DNASHAPE_LOOKUP[rc_p2][2]   # roll1
                val2_helt = DNASHAPE_LOOKUP[rc_p2][4]   # twist1

        # Average the values from the two pentamers if both are valid
        # Roll
        if not np.isnan(val1_roll) and not np.isnan(val2_roll):
            roll_arr[j] = (val1_roll + val2_roll) / 2.0
        elif not np.isnan(val1_roll):
            roll_arr[j] = val1_roll
        elif not np.isnan(val2_roll):
            roll_arr[j] = val2_roll
            
        # HelT
        if not np.isnan(val1_helt) and not np.isnan(val2_helt):
            helt_arr[j] = (val1_helt + val2_helt) / 2.0
        elif not np.isnan(val1_helt):
            helt_arr[j] = val1_helt
        elif not np.isnan(val2_helt):
            helt_arr[j] = val2_helt
            
    # Return as a matrix [5, N]
    return np.vstack([mgw_arr, prot_arr, roll_arr, helt_arr, ep_arr])

def extract_features_for_fasta(fasta_path: str, save_npy_path: str) -> None:
    print(f"Extracting features from {fasta_path}...")
    headers, sequences = load_fasta(fasta_path)
    
    features_list = []
    for seq in sequences:
        feat = compute_shape_features(seq)
        features_list.append(feat)
        
    features_arr = np.array(features_list) # Shape: [N, 5, 101]
    np.save(save_npy_path, features_arr)
    print(f"Saved {len(sequences)} sequences to {save_npy_path}. Shape: {features_arr.shape}")
    
    # Print NaN ratio
    nan_ratio = np.isnan(features_arr).mean()
    print(f"NaN ratio: {nan_ratio:.2%}")

if __name__ == "__main__":
    data_dir = r"data\processed"
    os.makedirs(data_dir, exist_ok=True)
    
    # 4 classes
    files = {
        "sp1": os.path.join(data_dir, "sp1_positive_final.fasta"),
        "sp2": os.path.join(data_dir, "sp2_positive_final.fasta"),
        "sp4": os.path.join(data_dir, "sp4_positive_final.fasta"),
        "negative": os.path.join(data_dir, "negative_final.fasta")
    }
    
    for label, fasta_file in files.items():
        npy_file = os.path.join(data_dir, f"dnashape_{label}.npy")
        extract_features_for_fasta(fasta_file, npy_file)
