#!/usr/bin/env python3
"""
Convert all lowercase bases (a, c, g, t, n) in FASTA files to uppercase (A, C, G, T, N) in-place,
and report the number of converted characters.
"""

import os
import sys

FASTA_DIR = "data/processed/positive_datasets_fasta"
FILES = [
    "SP1_downsampled_101bp.fasta",
    "SP2_downsampled_101bp.fasta",
    "SP4_downsampled_101bp.fasta",
]

def capitalize_fasta_file(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File not found {filepath}")
        return 0, 0

    converted_count = 0
    total_seqs = 0
    updated_lines = []

    with open(filepath, "r") as f:
        for line in f:
            if line.startswith(">"):
                updated_lines.append(line)
                total_seqs += 1
            else:
                # Count lowercase bases in this sequence line
                stripped = line.rstrip("\n\r")
                lowers = sum(1 for c in stripped if c.islower())
                converted_count += lowers
                
                # Convert line to uppercase
                updated_lines.append(line.upper())

    # Write back in-place
    with open(filepath, "w") as f:
        f.writelines(updated_lines)

    return total_seqs, converted_count

def main():
    print("=== FASTA Capitalization & Quality Standardisation ===")
    total_converted = 0
    total_files = 0
    
    for filename in FILES:
        path = os.path.join(FASTA_DIR, filename)
        if os.path.exists(path):
            seqs, count = capitalize_fasta_file(path)
            print(f"File: {filename}")
            print(f"  - Total sequences processed: {seqs}")
            print(f"  - Lowercase characters converted to uppercase: {count:,}")
            total_converted += count
            total_files += 1
        else:
            print(f"File not found: {path}")

    print("=" * 50)
    print(f"Completed! Capitalized {total_converted:,} characters across {total_files} files.")

if __name__ == "__main__":
    main()
