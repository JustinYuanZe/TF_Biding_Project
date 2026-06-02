#!/usr/bin/env python3
"""
Prepare final balanced datasets for training (1:1:1:1 ratio).
1. Applies reverse complement data augmentation to positive datasets.
   Each positive dataset goes from 1,696 -> 3,392 sequences.
2. Randomly samples negatives from the 3 shuffled negative datasets
   (1,131 from SP1, 1,131 from SP2, 1,130 from SP4) to get 3,392 sequences.
3. Overwrites data/processed/{sp1,sp2,sp4}_positive_final.fasta and
   data/processed/negative_final.fasta.
"""

import os
import sys
import random

# Paths
POS_DIR = "data/processed/positive_datasets_fasta"
NEG_DIR = "data/processed/negative_dinuc_shuffled"
OUT_DIR = "data/processed"

POS_FILES = {
    "SP1": "SP1_downsampled_101bp.fasta",
    "SP2": "SP2_downsampled_101bp.fasta",
    "SP4": "SP4_downsampled_101bp.fasta",
}

NEG_FILES = {
    "SP1": "SP1_dinuc_shuffled.fasta",
    "SP2": "SP2_dinuc_shuffled.fasta",
    "SP4": "SP4_dinuc_shuffled.fasta",
}

COMPLEMENT = {
    'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N',
    'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'n': 'n'
}

def reverse_complement(seq):
    comp = "".join(COMPLEMENT.get(c, c) for c in seq)
    return comp[::-1]

def read_fasta(path):
    records = []
    if not os.path.exists(path):
        print(f"Warning: File {path} not found.")
        return records
    with open(path, "r") as f:
        header = ""
        seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    records.append((header, "".join(seq)))
                header = line
                seq = []
            else:
                seq.append(line)
        if header:
            records.append((header, "".join(seq)))
    return records

def write_fasta(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for h, s in records:
            f.write(f"{h}\n{s}\n")

def main():
    # Set seed for reproducibility
    random.seed(42)
    print("=== FINAL DATASET PREPARATION (1:1:1:1 Balance) ===")

    # 1. Process Positives with Reverse Complement Augmentation
    pos_records = {}
    augmented_counts = {}

    for tf, fname in POS_FILES.items():
        path = os.path.join(POS_DIR, fname)
        records = read_fasta(path)
        if not records:
            print(f"Error: No sequences loaded for positive {tf}.")
            sys.exit(1)
        print(f"Loaded {len(records)} original positive sequences for {tf}.")

        augmented_records = []
        for h, s in records:
            # Original sequence
            augmented_records.append((h, s))
            # Reverse complement sequence
            rc_h = f"{h}_revcomp"
            rc_s = reverse_complement(s)
            augmented_records.append((rc_h, rc_s))

        pos_records[tf] = augmented_records
        augmented_counts[tf] = len(augmented_records)

        # Write to data/processed/spX_positive_final.fasta
        out_path = os.path.join(OUT_DIR, f"{tf.lower()}_positive_final.fasta")
        write_fasta(out_path, augmented_records)
        print(f"  -> Augmented with RevComp to {len(augmented_records)} sequences. Saved to {out_path}")

    # The target size for each of the 4 classes is now 3,392
    target_neg_size = augmented_counts["SP1"]  # 3392
    print(f"\nTarget negative dataset size: {target_neg_size} (to match augmented positive size)")

    # 2. Sample Negatives from the 3 Shuffled Negative Sets
    # We want to sample evenly: 3392 = 1131 + 1131 + 1130
    neg_shares = {
        "SP1": 1131,
        "SP2": 1131,
        "SP4": 1130
    }

    final_neg_records = []
    for tf, share_size in neg_shares.items():
        path = os.path.join(NEG_DIR, NEG_FILES[tf])
        records = read_fasta(path)
        if not records:
            print(f"Error: No sequences loaded for negative shuffled {tf}.")
            sys.exit(1)
        print(f"Loaded {len(records)} shuffled negative sequences for {tf}.")

        # Sample share_size sequences randomly
        sampled = random.sample(records, share_size)
        print(f"  -> Randomly sampled {len(sampled)} sequences.")
        
        # Modify headers to indicate they are shuffled and sourced from this TF
        modified_sampled = []
        for h, s in sampled:
            new_h = h.replace(">", f">neg_{tf.lower()}_")
            modified_sampled.append((new_h, s))
            
        final_neg_records.extend(modified_sampled)

    # Shuffle the combined negative set to mix them up
    random.shuffle(final_neg_records)

    # Write to data/processed/negative_final.fasta
    out_neg_path = os.path.join(OUT_DIR, "negative_final.fasta")
    write_fasta(out_neg_path, final_neg_records)
    print(f"\nTotal negative sequences collected: {len(final_neg_records)}")
    print(f"Saved final negative dataset to {out_neg_path}")

    # 3. Final Verification
    print("\n=== FINAL DATASET SUMMARY (VERIFICATION) ===")
    for tf in ["sp1", "sp2", "sp4"]:
        p = os.path.join(OUT_DIR, f"{tf}_positive_final.fasta")
        print(f"  {tf.upper()} Positive Final: {len(read_fasta(p))} sequences")
    print(f"  Negative Final:         {len(read_fasta(out_neg_path))} sequences")

    # Ratio check
    sizes = [len(pos_records["SP1"]), len(pos_records["SP2"]), len(pos_records["SP4"]), len(final_neg_records)]
    if len(set(sizes)) == 1:
        print("  -> BALANCE STATUS: PERFECTLY BALANCED 1:1:1:1 (3392 sequences per class)!")
    else:
        print("  -> BALANCE STATUS: Warning! Imbalance detected.")

    print("\nCompleted successfully!")

if __name__ == "__main__":
    main()
