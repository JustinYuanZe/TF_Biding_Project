#!/usr/bin/env python3
"""
Generate negative datasets using two methods:
1. Dinucleotide shuffling (Altschul-Erickson algorithm, 1985)
   - Preserves exact mono- and di-nucleotide frequencies
   - Uses Eulerian path on the dinucleotide graph
2. Reverse complement
   - Complement each base (A<->T, C<->G) then reverse the strand

Input:  data/processed/positive_datasets_fasta/SP{1,2,4}_downsampled_101bp.fasta
Output: data/processed/negative_dinuc_shuffled/SP{1,2,4}_dinuc_shuffled.fasta
        data/processed/negative_revcomp/SP{1,2,4}_revcomp.fasta
"""

import os
import sys
import random
from collections import defaultdict, Counter

# ============================================================
# 1. Altschul-Erickson Dinucleotide Shuffle
# ============================================================

def count_dinucleotides(seq):
    """Count all dinucleotide frequencies in a sequence."""
    counts = Counter()
    for i in range(len(seq) - 1):
        counts[seq[i:i+2]] += 1
    return counts


def dinucleotide_shuffle(seq, rng=None):
    """
    Shuffle a DNA sequence preserving exact dinucleotide frequencies.
    
    Algorithm (Altschul & Erickson, 1985):
    1. Build directed multigraph: nodes = nucleotides, edges = dinucleotides
    2. For each node, randomly permute outgoing edges EXCEPT the last edge
       (last edge in original traversal order is kept last to guarantee
       the Eulerian path exists and doesn't get stuck)
    3. Greedily traverse the Eulerian path from the first character
    
    Returns: shuffled sequence with identical dinucleotide composition
    """
    if rng is None:
        rng = random.Random()

    n = len(seq)
    if n <= 2:
        return seq

    # Build outgoing edge lists for each node (in original order)
    edges = defaultdict(list)
    for i in range(n - 1):
        edges[seq[i]].append(seq[i + 1])

    # For each node: shuffle all outgoing edges EXCEPT the last one
    # Keeping the last edge last ensures the path won't dead-end prematurely
    for node in edges:
        edge_list = edges[node]
        if len(edge_list) > 1:
            last = edge_list[-1]
            rest = edge_list[:-1]
            rng.shuffle(rest)
            edges[node] = rest + [last]

    # Traverse the Eulerian path greedily
    ptr = {node: 0 for node in edges}
    result = [seq[0]]
    current = seq[0]

    for _ in range(n - 1):
        nxt = edges[current][ptr[current]]
        ptr[current] += 1
        result.append(nxt)
        current = nxt

    return ''.join(result)


# ============================================================
# 2. Reverse Complement
# ============================================================

COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}


def reverse_complement(seq):
    """
    Generate the reverse complement of a DNA sequence.
    Step 1: Complement each base (A<->T, C<->G)
    Step 2: Reverse the strand
    """
    comp = ''.join(COMPLEMENT.get(c, c) for c in seq)
    return comp[::-1]


# ============================================================
# I/O Helpers
# ============================================================

def read_fasta(filepath):
    """Parse a FASTA file into list of (header, sequence) tuples."""
    records = []
    with open(filepath, 'r') as f:
        header = ""
        seq_parts = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header:
                    records.append((header, ''.join(seq_parts)))
                header = line
                seq_parts = []
            else:
                seq_parts.append(line)
        if header:
            records.append((header, ''.join(seq_parts)))
    return records


def write_fasta(filepath, records):
    """Write list of (header, sequence) tuples to a FASTA file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for header, seq in records:
            f.write(f"{header}\n{seq}\n")


def gc_content(seq):
    """Calculate GC content as a percentage."""
    gc = seq.count('G') + seq.count('C')
    return (gc / len(seq) * 100) if len(seq) > 0 else 0.0


def mono_freqs(seq):
    """Calculate mononucleotide frequencies."""
    c = Counter(seq)
    total = len(seq)
    return {base: c[base] / total * 100 for base in 'ACGT'}


# ============================================================
# Analysis & Reporting
# ============================================================

def analyze_dinuc_preservation(original_seqs, shuffled_seqs, label, rng):
    """
    Verify that dinucleotide frequencies are preserved after shuffling.
    Also report GC content, mono-nucleotide frequencies, and random samples.
    """
    print(f"\n{'='*70}")
    print(f"  DINUCLEOTIDE SHUFFLE REPORT: {label}")
    print(f"{'='*70}")

    # Aggregate dinucleotide counts across ALL sequences
    orig_dinuc_total = Counter()
    shuf_dinuc_total = Counter()
    orig_mono_total = Counter()
    shuf_mono_total = Counter()
    orig_gc_values = []
    shuf_gc_values = []

    mismatches = 0

    for (_, orig_seq), (_, shuf_seq) in zip(original_seqs, shuffled_seqs):
        # Per-sequence dinucleotide check
        orig_dinuc = count_dinucleotides(orig_seq)
        shuf_dinuc = count_dinucleotides(shuf_seq)
        if orig_dinuc != shuf_dinuc:
            mismatches += 1

        orig_dinuc_total += orig_dinuc
        shuf_dinuc_total += shuf_dinuc
        orig_mono_total += Counter(orig_seq)
        shuf_mono_total += Counter(shuf_seq)
        orig_gc_values.append(gc_content(orig_seq))
        shuf_gc_values.append(gc_content(shuf_seq))

    n = len(original_seqs)
    print(f"\n  Total sequences: {n}")
    print(f"  Dinucleotide preservation check: "
          f"{'ALL PASSED' if mismatches == 0 else f'{mismatches} MISMATCHES!'}")

    # Aggregated GC content
    avg_orig_gc = sum(orig_gc_values) / n
    avg_shuf_gc = sum(shuf_gc_values) / n
    print(f"\n  Average GC content:")
    print(f"    Original:  {avg_orig_gc:.2f}%")
    print(f"    Shuffled:  {avg_shuf_gc:.2f}%")
    print(f"    Delta:     {abs(avg_orig_gc - avg_shuf_gc):.4f}%")

    # Mononucleotide comparison
    total_orig = sum(orig_mono_total.values())
    total_shuf = sum(shuf_mono_total.values())
    print(f"\n  Mononucleotide frequencies (aggregated):")
    print(f"    {'Base':<6} {'Original':>10} {'Shuffled':>10} {'Match':>8}")
    for base in 'ACGT':
        o_pct = orig_mono_total[base] / total_orig * 100
        s_pct = shuf_mono_total[base] / total_shuf * 100
        match = "YES" if abs(o_pct - s_pct) < 0.01 else "NO"
        print(f"    {base:<6} {o_pct:>9.2f}% {s_pct:>9.2f}% {match:>8}")

    # Dinucleotide comparison (top 16)
    all_dinucs = sorted(set(list(orig_dinuc_total.keys()) + list(shuf_dinuc_total.keys())))
    total_orig_di = sum(orig_dinuc_total.values())
    total_shuf_di = sum(shuf_dinuc_total.values())
    print(f"\n  Dinucleotide frequencies (aggregated):")
    print(f"    {'Dinuc':<6} {'Original':>10} {'Shuffled':>10} {'Match':>8}")
    for di in all_dinucs:
        o_pct = orig_dinuc_total[di] / total_orig_di * 100
        s_pct = shuf_dinuc_total[di] / total_shuf_di * 100
        match = "YES" if abs(o_pct - s_pct) < 0.01 else "NO"
        print(f"    {di:<6} {o_pct:>9.2f}% {s_pct:>9.2f}% {match:>8}")

    # Random samples
    print(f"\n  --- Random Samples (3 sequences) ---")
    sample_indices = rng.sample(range(n), min(3, n))
    for idx in sample_indices:
        orig_h, orig_s = original_seqs[idx]
        _, shuf_s = shuffled_seqs[idx]
        orig_di = count_dinucleotides(orig_s)
        shuf_di = count_dinucleotides(shuf_s)
        preserved = "PRESERVED" if orig_di == shuf_di else "BROKEN!"
        print(f"\n  Sample (index {idx}):")
        print(f"    Header:   {orig_h}")
        print(f"    Original: {orig_s}")
        print(f"    Shuffled: {shuf_s}")
        print(f"    Length:   {len(orig_s)} -> {len(shuf_s)}")
        print(f"    GC:       {gc_content(orig_s):.1f}% -> {gc_content(shuf_s):.1f}%")
        print(f"    Dinuc:    {preserved}")
        # Show a few dinucleotide counts as proof
        top_di = orig_di.most_common(4)
        di_str = ", ".join(f"{d}={c}" for d, c in top_di)
        print(f"    Top dinucs (original): {di_str}")


def analyze_revcomp(original_seqs, revcomp_seqs, label, rng):
    """Report on reverse complement generation."""
    print(f"\n{'='*70}")
    print(f"  REVERSE COMPLEMENT REPORT: {label}")
    print(f"{'='*70}")

    n = len(original_seqs)
    length_ok = all(len(o[1]) == len(r[1]) for o, r in zip(original_seqs, revcomp_seqs))
    
    # Verify complement + reverse correctness
    correct = 0
    for (_, orig_s), (_, rc_s) in zip(original_seqs, revcomp_seqs):
        expected = reverse_complement(orig_s)
        if rc_s == expected:
            correct += 1

    orig_gc_values = [gc_content(s) for _, s in original_seqs]
    rc_gc_values = [gc_content(s) for _, s in revcomp_seqs]
    avg_orig_gc = sum(orig_gc_values) / n
    avg_rc_gc = sum(rc_gc_values) / n

    print(f"\n  Total sequences: {n}")
    print(f"  Length preserved: {'YES' if length_ok else 'NO'}")
    print(f"  Correctness check: {correct}/{n} "
          f"({'ALL CORRECT' if correct == n else 'ERRORS FOUND!'})")
    print(f"\n  Average GC content:")
    print(f"    Original:         {avg_orig_gc:.2f}%")
    print(f"    Reverse Compl.:   {avg_rc_gc:.2f}%")
    print(f"    Delta:            {abs(avg_orig_gc - avg_rc_gc):.4f}%")
    print(f"    (GC content is mathematically invariant under reverse complement)")

    # Show A<->T and C<->G swap in aggregate
    orig_mono = Counter()
    rc_mono = Counter()
    for _, s in original_seqs:
        orig_mono += Counter(s)
    for _, s in revcomp_seqs:
        rc_mono += Counter(s)

    total = sum(orig_mono.values())
    print(f"\n  Mononucleotide swap verification (A<->T, C<->G):")
    print(f"    {'Base':<6} {'Original':>10} {'RevComp':>10} {'Expected swap':>15}")
    for base, comp_base in [('A', 'T'), ('T', 'A'), ('C', 'G'), ('G', 'C')]:
        o_pct = orig_mono[base] / total * 100
        r_pct = rc_mono[base] / total * 100
        exp_pct = orig_mono[comp_base] / total * 100
        match = "OK" if abs(r_pct - exp_pct) < 0.01 else "MISMATCH"
        print(f"    {base:<6} {o_pct:>9.2f}% {r_pct:>9.2f}% {exp_pct:>9.2f}% ({match})")

    # Random samples
    print(f"\n  --- Random Samples (3 sequences) ---")
    sample_indices = rng.sample(range(n), min(3, n))
    for idx in sample_indices:
        orig_h, orig_s = original_seqs[idx]
        _, rc_s = revcomp_seqs[idx]
        # Show step-by-step: complement then reverse
        comp_s = ''.join(COMPLEMENT.get(c, c) for c in orig_s)
        print(f"\n  Sample (index {idx}):")
        print(f"    Header:      {orig_h}")
        print(f"    Original:    {orig_s}")
        print(f"    Complement:  {comp_s}")
        print(f"    Rev.Compl.:  {rc_s}")
        print(f"    Length:      {len(orig_s)} -> {len(rc_s)}")
        print(f"    GC:          {gc_content(orig_s):.1f}% -> {gc_content(rc_s):.1f}%")
        # Quick sanity: first base of original = complement of last base of revcomp
        print(f"    Sanity:      orig[0]={orig_s[0]}, compl(rc[-1])="
              f"{COMPLEMENT.get(rc_s[-1],'?')} -> "
              f"{'MATCH' if orig_s[0] == COMPLEMENT.get(rc_s[-1]) else 'FAIL'}")


# ============================================================
# Main
# ============================================================

def main():
    random.seed(42)
    rng = random.Random(42)

    input_dir = "data/processed/positive_datasets_fasta"
    out_dinuc = "data/processed/negative_dinuc_shuffled"
    out_revcomp = "data/processed/negative_revcomp"

    tfs = ["SP1", "SP2", "SP4"]

    print("=" * 70)
    print("  NEGATIVE DATASET GENERATION")
    print("  Method 1: Dinucleotide Shuffle (Altschul-Erickson, 1985)")
    print("  Method 2: Reverse Complement")
    print("=" * 70)

    for tf in tfs:
        input_path = os.path.join(input_dir, f"{tf}_downsampled_101bp.fasta")
        if not os.path.exists(input_path):
            print(f"WARNING: {input_path} not found, skipping {tf}")
            continue

        print(f"\n>>> Processing {tf}...")
        original = read_fasta(input_path)
        print(f"    Loaded {len(original)} sequences from {os.path.basename(input_path)}")

        # --- Method 1: Dinucleotide shuffle ---
        shuffled = []
        for header, seq in original:
            shuf_seq = dinucleotide_shuffle(seq, rng)
            new_header = header.replace(">", ">dinuc_shuffle|")
            shuffled.append((new_header, shuf_seq))

        out_path_dinuc = os.path.join(out_dinuc, f"{tf}_dinuc_shuffled.fasta")
        write_fasta(out_path_dinuc, shuffled)
        print(f"    Saved dinucleotide-shuffled: {out_path_dinuc}")

        analyze_dinuc_preservation(original, shuffled, tf, rng)

        # --- Method 2: Reverse complement ---
        revcomps = []
        for header, seq in original:
            rc_seq = reverse_complement(seq)
            new_header = header.replace(">", ">revcomp|")
            revcomps.append((new_header, rc_seq))

        out_path_rc = os.path.join(out_revcomp, f"{tf}_revcomp.fasta")
        write_fasta(out_path_rc, revcomps)
        print(f"    Saved reverse-complement: {out_path_rc}")

        analyze_revcomp(original, revcomps, tf, rng)

    # Final summary
    print(f"\n{'='*70}")
    print("  FINAL SUMMARY")
    print(f"{'='*70}")
    for tf in tfs:
        dinuc_path = os.path.join(out_dinuc, f"{tf}_dinuc_shuffled.fasta")
        rc_path = os.path.join(out_revcomp, f"{tf}_revcomp.fasta")
        dinuc_size = os.path.getsize(dinuc_path) if os.path.exists(dinuc_path) else 0
        rc_size = os.path.getsize(rc_path) if os.path.exists(rc_path) else 0
        print(f"  {tf}:")
        print(f"    Dinuc shuffled:  {dinuc_path} ({dinuc_size:,} bytes)")
        print(f"    Reverse compl.:  {rc_path} ({rc_size:,} bytes)")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
