#!/usr/bin/env python3
"""
Extract DNA sequences (FASTA) from an indexed reference genome (hg38.fa + hg38.fa.fai).

Designed for Google Colab with hg38.fa stored on Google Drive.
Uses binary seek on the .fai index — fast, low memory, no external dependencies.
Preserves original casing (upper/lowercase) and N characters as-is.

Usage:
    python extract_fasta.py --fasta /content/drive/MyDrive/ML_Project/data/hg38.fa
    python extract_fasta.py --test   # run built-in unit tests
"""

import os
import sys
import argparse
import time


class FastaIndexedReader:
    """Read sequences from an indexed FASTA file using byte-level seek."""

    def __init__(self, fa_path):
        self.fa_path = fa_path
        self.fai_path = fa_path + ".fai"
        self.index = {}
        self.fa_file = None

        if not os.path.exists(self.fa_path):
            raise FileNotFoundError(f"FASTA file not found: {self.fa_path}")
        if not os.path.exists(self.fai_path):
            raise FileNotFoundError(
                f"Index file not found: {self.fai_path}\n"
                f"Create it first: samtools faidx {os.path.basename(self.fa_path)}"
            )

        self._load_index()
        self.fa_file = open(self.fa_path, 'rb')

    def _load_index(self):
        """Parse the .fai index file."""
        with open(self.fai_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    chrom = parts[0]
                    self.index[chrom] = {
                        'length': int(parts[1]),
                        'offset': int(parts[2]),
                        'line_bases': int(parts[3]),
                        'line_bytes': int(parts[4]),
                    }
        print(f"Loaded index for {len(self.index)} chromosomes.")

    def get_sequence(self, chrom, start, end):
        """Extract sequence from chrom:start-end (0-based, half-open BED coordinates)."""
        if chrom not in self.index:
            alt = chrom[3:] if chrom.startswith('chr') else 'chr' + chrom
            if alt in self.index:
                chrom = alt
            else:
                raise KeyError(f"Chromosome '{chrom}' not found in index.")

        info = self.index[chrom]
        start = max(0, start)
        end = min(end, info['length'])
        if start >= end:
            return ""

        line_bases = info['line_bases']
        line_bytes = info['line_bytes']
        newline_bytes = line_bytes - line_bases

        start_line = start // line_bases
        start_offset = start % line_bases
        start_byte = info['offset'] + start_line * line_bytes + start_offset

        length = end - start
        num_lines = (start_offset + length + line_bases - 1) // line_bases
        bytes_to_read = length + num_lines * newline_bytes + 10

        self.fa_file.seek(start_byte)
        data = self.fa_file.read(bytes_to_read)
        seq = data.decode('ascii', errors='ignore').replace('\n', '').replace('\r', '')
        return seq[:length]

    def close(self):
        if self.fa_file:
            self.fa_file.close()
            self.fa_file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def process_bed_file(bed_path, reader, out_fasta_path):
    """Read a BED file and extract sequences to FASTA."""
    print(f"Processing: {os.path.basename(bed_path)} -> {os.path.basename(out_fasta_path)}")

    count = 0
    t0 = time.time()

    with open(bed_path, 'r') as f_in, open(out_fasta_path, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue

            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            try:
                seq = reader.get_sequence(chrom, start, end)
                f_out.write(f">{chrom}:{start}-{end}\n{seq}\n")
                count += 1
            except Exception as e:
                print(f"Error extracting {chrom}:{start}-{end}: {e}", file=sys.stderr)

    print(f"Extracted {count} sequences in {time.time() - t0:.2f}s.")
    return count


def run_self_test():
    """Run built-in unit tests with a mock genome."""
    print("--- RUNNING SELF-TEST ---")
    mock_fa = "mock_genome.fa"
    mock_fai = mock_fa + ".fai"

    chr1_seq = "ACGT" * 25        # 100bp
    chr2_seq = "NacgT" * 24       # 120bp, mixed case + N

    with open(mock_fa, 'w', newline='\n') as f:
        f.write(">chr1\n")
        for i in range(0, 100, 50):
            f.write(chr1_seq[i:i+50] + "\n")
        f.write(">chr2\n")
        for i in range(0, 120, 40):
            f.write(chr2_seq[i:i+40] + "\n")

    with open(mock_fa, 'rb') as f:
        content = f.read()

    offset_chr1 = content.find(b"ACGT")
    offset_chr2 = content.find(b"NacgT")

    with open(mock_fai, 'w', newline='\n') as f:
        f.write(f"chr1\t100\t{offset_chr1}\t50\t51\n")
        f.write(f"chr2\t120\t{offset_chr2}\t40\t41\n")

    print("Created mock genome and index.")

    try:
        with FastaIndexedReader(mock_fa) as reader:
            # Test 1: cross-line read
            seq1 = reader.get_sequence("chr1", 45, 55)
            assert len(seq1) == 10, f"Wrong length: {len(seq1)}"
            assert seq1 == "CGTACGTACG", f"Wrong content: {seq1}"
            print("Test 1 (cross-line read chr1) PASSED")

            # Test 2: preserve N and lowercase
            seq2 = reader.get_sequence("chr2", 0, 15)
            assert seq2 == "NacgTNacgTNacgT", f"Wrong content: {seq2}"
            print("Test 2 (preserve N + lowercase chr2) PASSED")

            # Test 3: boundary clipping
            seq3 = reader.get_sequence("chr1", 95, 105)
            assert len(seq3) == 5, f"Wrong length: {len(seq3)}"
            print("Test 3 (boundary clipping) PASSED")

        print("ALL TESTS PASSED")
    finally:
        for p in [mock_fa, mock_fai]:
            if os.path.exists(p):
                os.remove(p)


def main():
    parser = argparse.ArgumentParser(
        description="Extract FASTA sequences from an indexed genome using .fai index."
    )
    parser.add_argument("--fasta", type=str, help="Path to hg38.fa")
    parser.add_argument("--bed_dir", type=str, help="Directory containing downsampled BED files")
    parser.add_argument("--out_dir", type=str, help="Output directory for FASTA files")
    parser.add_argument("--test", action="store_true", help="Run built-in unit tests")

    args = parser.parse_args()

    if args.test:
        run_self_test()
        sys.exit(0)

    bed_dir = args.bed_dir or "data/processed/downsampled_101bp"
    out_dir = args.out_dir or "data/processed/downsampled_101bp"

    if not args.fasta:
        print("NOTE: --fasta not specified. Searching default paths...")
        possible_paths = [
            "/content/drive/MyDrive/ML_Project/data/hg38.fa",
            "/content/drive/MyDrive/SP1_TF_Binding_Project/data/hg38.fa",
            "data/raw/hg38.fa",
            "data/hg38.fa",
            "./hg38.fa",
        ]
        fasta_path = next((p for p in possible_paths if os.path.exists(p)), None)
        if not fasta_path:
            print("\nCould not find hg38.fa at any default location.")
            print("Usage: python extract_fasta.py --fasta /path/to/hg38.fa")
            sys.exit(1)
    else:
        fasta_path = args.fasta

    print(f"Reference genome: {fasta_path}")
    print(f"BED directory:    {bed_dir}")
    print(f"Output directory: {out_dir}")

    tfs = ["SP1", "SP2", "SP4"]
    bed_files = {}
    for tf in tfs:
        path = os.path.join(bed_dir, f"{tf}_downsampled_101bp.bed")
        if os.path.exists(path):
            bed_files[tf] = path
        else:
            print(f"Warning: BED file not found for {tf}: {path}")

    if not bed_files:
        print("Error: No BED files found.")
        sys.exit(1)

    try:
        with FastaIndexedReader(fasta_path) as reader:
            os.makedirs(out_dir, exist_ok=True)
            for tf, bed_path in bed_files.items():
                out_path = os.path.join(out_dir, f"{tf}_downsampled_101bp.fasta")
                process_bed_file(bed_path, reader, out_path)
        print("\nDone.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
