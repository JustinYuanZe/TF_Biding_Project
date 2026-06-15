#!/usr/bin/env python3
"""
Deep diagnostic analysis of the training pipeline.
Investigates why binary classification (Positive vs Negative) only reaches 77%
when the majority-class baseline is 75%.
"""

import os
import random
from collections import Counter
from typing import Dict, List, Set, Tuple

import numpy as np

random.seed(42)

DATA_DIR = "data/processed"

def read_fasta(path: str) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    with open(path, 'r') as f:
        header: str = ''
        seq: List[str] = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header:
                    records.append((header, ''.join(seq)))
                header = line
                seq = []
            else:
                seq.append(line)
        if header:
            records.append((header, ''.join(seq)))
    return records

def gc_content(seq: str) -> float:
    gc: int = seq.count('G') + seq.count('C')
    return gc / len(seq) * 100

def get_kmers(seq: str, k: int = 6) -> Set[str]:
    return set(seq[i:i+k] for i in range(len(seq)-k+1))

def get_kmer_freq(seq: str, k: int = 6) -> Counter[str]:
    kmers: Counter[str] = Counter()
    for i in range(len(seq)-k+1):
        kmers[seq[i:i+k]] += 1
    return kmers


# =====================================================
# 1. BASIC DATASET STATS
# =====================================================
print("=" * 70)
print("1. BASIC DATASET STATISTICS")
print("=" * 70)

pos_all = {}
for tf in ['sp1', 'sp2', 'sp4']:
    recs = read_fasta(os.path.join(DATA_DIR, f"{tf}_positive_final.fasta"))
    pos_all[tf] = recs
    gc_vals = [gc_content(s) for h, s in recs]
    lengths = [len(s) for h, s in recs]
    print(f"{tf.upper()}: {len(recs)} seqs, len={min(lengths)}-{max(lengths)}, "
          f"GC={np.mean(gc_vals):.2f}% +/- {np.std(gc_vals):.2f}%")

neg_recs = read_fasta(os.path.join(DATA_DIR, "negative_final.fasta"))
neg_gc = [gc_content(s) for h, s in neg_recs]
neg_lengths = [len(s) for h, s in neg_recs]
print(f"NEG:  {len(neg_recs)} seqs, len={min(neg_lengths)}-{max(neg_lengths)}, "
      f"GC={np.mean(neg_gc):.2f}% +/- {np.std(neg_gc):.2f}%")

# =====================================================
# 2. NEGATIVE SOURCE DISTRIBUTION
# =====================================================
print("\n" + "=" * 70)
print("2. NEGATIVE SOURCE DISTRIBUTION")
print("=" * 70)

sources = Counter()
for h, s in neg_recs:
    if 'neg_sp1' in h:
        sources['from_SP1_shuffle'] += 1
    elif 'neg_sp2' in h:
        sources['from_SP2_shuffle'] += 1
    elif 'neg_sp4' in h:
        sources['from_SP4_shuffle'] += 1
    else:
        sources['unknown'] += 1
for k, v in sources.items():
    print(f"  {k}: {v}")

# =====================================================
# 3. SEQUENCE IDENTITY CHECK
# =====================================================
print("\n" + "=" * 70)
print("3. EXACT SEQUENCE IDENTITY CHECK")
print("=" * 70)

pos_all_seqs = set()
for tf in ['sp1', 'sp2', 'sp4']:
    for h, s in pos_all[tf]:
        pos_all_seqs.add(s)

neg_seqs_set = set(s for h, s in neg_recs)
neg_in_pos = sum(1 for h, s in neg_recs if s in pos_all_seqs)
print(f"  Unique positive sequences: {len(pos_all_seqs)}")
print(f"  Unique negative sequences: {len(neg_seqs_set)}")
print(f"  Negative sequences identical to a positive: {neg_in_pos}")

# =====================================================
# 4. K-MER OVERLAP ANALYSIS
# =====================================================
print("\n" + "=" * 70)
print("4. K-MER OVERLAP ANALYSIS")
print("=" * 70)

for k in [4, 5, 6, 8]:
    pos_kmers = set()
    for tf in ['sp1', 'sp2', 'sp4']:
        for h, s in random.sample(pos_all[tf], min(200, len(pos_all[tf]))):
            pos_kmers.update(get_kmers(s, k))
    
    neg_kmers = set()
    for h, s in random.sample(neg_recs, min(200, len(neg_recs))):
        neg_kmers.update(get_kmers(s, k))
    
    overlap = pos_kmers & neg_kmers
    print(f"  k={k}: pos_unique={len(pos_kmers)}, neg_unique={len(neg_kmers)}, "
          f"overlap={len(overlap)} ({100*len(overlap)/max(len(pos_kmers),1):.1f}% of pos)")

# =====================================================
# 5. DINUCLEOTIDE FREQUENCY COMPARISON
# =====================================================
print("\n" + "=" * 70)
print("5. DINUCLEOTIDE FREQUENCY COMPARISON")
print("=" * 70)

def dinuc_freq(sequences: List[str]) -> Dict[str, float]:
    counts: Counter[str] = Counter()
    total: int = 0
    for s in sequences:
        for i in range(len(s) - 1):
            counts[s[i:i+2]] += 1
            total += 1
    return {k: v / total for k, v in counts.items()}

pos_seqs = [s for tf in ['sp1','sp2','sp4'] for h, s in pos_all[tf]]
neg_seqs = [s for h, s in neg_recs]

pos_dinuc = dinuc_freq(pos_seqs)
neg_dinuc = dinuc_freq(neg_seqs)

dinucs = sorted(set(list(pos_dinuc.keys()) + list(neg_dinuc.keys())))
max_diff = 0
print(f"  {'Dinuc':<8} {'Positive':>10} {'Negative':>10} {'Diff':>10}")
for di in dinucs:
    p = pos_dinuc.get(di, 0)
    n = neg_dinuc.get(di, 0)
    diff = abs(p - n)
    max_diff = max(max_diff, diff)
    print(f"  {di:<8} {p:>9.4f}  {n:>9.4f}  {diff:>9.6f}")
print(f"\n  Max dinucleotide frequency difference: {max_diff:.6f}")
print(f"  >>> Dinucleotide shuffle PRESERVES these frequencies by design!")
print(f"  >>> This is why the model struggles - the negative is too similar!")

# =====================================================
# 6. CRITICAL: RevComp in Positive vs Negative
# =====================================================
print("\n" + "=" * 70)
print("6. CRITICAL: RevComp AUGMENTATION ANALYSIS")
print("=" * 70)
print("  Positive datasets: original + revcomp (paired every 2 lines)")
print("  Negative dataset:  dinuc-shuffled ONLY (no revcomp)")
print()

# Check if positive sequences really have revcomp pairs
for tf in ['sp1', 'sp2', 'sp4']:
    recs = pos_all[tf]
    pairs_ok = 0
    pairs_fail = 0
    for i in range(0, len(recs)-1, 2):
        h1, s1 = recs[i]
        h2, s2 = recs[i+1]
        if '_revcomp' in h2:
            comp = {'A':'T','T':'A','C':'G','G':'C','N':'N'}
            expected_rc = ''.join(comp.get(c,c) for c in reversed(s1))
            if s2 == expected_rc:
                pairs_ok += 1
            else:
                pairs_fail += 1
        else:
            pairs_fail += 1
    print(f"  {tf.upper()}: {pairs_ok} correct revcomp pairs, {pairs_fail} failures")

# =====================================================
# 7. DNAshape FEATURE ANALYSIS
# =====================================================
print("\n" + "=" * 70)
print("7. DNAshape FEATURE ANALYSIS")
print("=" * 70)

channel_names = ["MGW", "ProT", "Roll", "HelT", "EP"]
for label in ['sp1', 'sp2', 'sp4', 'negative']:
    fpath = os.path.join(DATA_DIR, f"dnashape_{label}.npy")
    shape = np.load(fpath)
    nan_ratio = np.isnan(shape).mean()
    print(f"\n  {label.upper()} shape: {shape.shape}, NaN ratio: {nan_ratio:.4f}")
    for ch in range(5):
        vals = shape[:, ch, :].flatten()
        valid = vals[~np.isnan(vals)]
        print(f"    {channel_names[ch]}: mean={np.mean(valid):.4f}, std={np.std(valid):.4f}, "
              f"min={np.min(valid):.4f}, max={np.max(valid):.4f}")

# =====================================================
# 8. SHAPE DISCRIMINABILITY TEST
# =====================================================
print("\n" + "=" * 70)
print("8. SHAPE DISCRIMINABILITY (Positive vs Negative)")
print("=" * 70)

# Load all shapes
sp1_shape = np.load(os.path.join(DATA_DIR, "dnashape_sp1.npy"))
sp2_shape = np.load(os.path.join(DATA_DIR, "dnashape_sp2.npy"))
sp4_shape = np.load(os.path.join(DATA_DIR, "dnashape_sp4.npy"))
neg_shape = np.load(os.path.join(DATA_DIR, "dnashape_negative.npy"))

pos_shape = np.concatenate([sp1_shape, sp2_shape, sp4_shape], axis=0)

for ch in range(5):
    pos_mean = np.nanmean(pos_shape[:, ch, :])
    neg_mean = np.nanmean(neg_shape[:, ch, :])
    pos_std = np.nanstd(pos_shape[:, ch, :])
    neg_std = np.nanstd(neg_shape[:, ch, :])
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((pos_std**2 + neg_std**2) / 2)
    if pooled_std > 0:
        cohens_d = abs(pos_mean - neg_mean) / pooled_std
    else:
        cohens_d = 0
    
    print(f"  {channel_names[ch]}: pos_mean={pos_mean:.4f}, neg_mean={neg_mean:.4f}, "
          f"diff={abs(pos_mean-neg_mean):.4f}, Cohen's d={cohens_d:.4f}")

print()
print("  Interpretation:")
print("  Cohen's d < 0.2 = negligible difference")
print("  Cohen's d 0.2-0.5 = small difference")
print("  Cohen's d 0.5-0.8 = medium difference")
print("  Cohen's d > 0.8 = large difference")

# =====================================================
# 9. CRITICAL: POSITION-WISE SHAPE COMPARISON
# =====================================================
print("\n" + "=" * 70)
print("9. POSITION-WISE SHAPE (Center 20 positions, MGW channel)")
print("=" * 70)

center = 50  # center of 101bp
for ch_idx, ch_name in [(0, "MGW"), (4, "EP")]:
    print(f"\n  {ch_name} channel, positions {center-10} to {center+10}:")
    print(f"  {'Pos':<6} {'Positive':>10} {'Negative':>10} {'Diff':>10}")
    for pos in range(center-10, center+11):
        p = np.nanmean(pos_shape[:, ch_idx, pos])
        n = np.nanmean(neg_shape[:, ch_idx, pos])
        diff = p - n
        marker = " ***" if abs(diff) > 0.1 else ""
        print(f"  {pos:<6} {p:>10.4f} {n:>10.4f} {diff:>10.4f}{marker}")

# =====================================================
# 10. FINAL DIAGNOSIS
# =====================================================
print("\n" + "=" * 70)
print("10. FINAL DIAGNOSIS")
print("=" * 70)
print("""
  The dinucleotide shuffle method creates negatives that are EXTREMELY
  similar to positives because:
  
  1. SAME GC content (by construction)
  2. SAME dinucleotide frequencies (by construction)
  3. Very high k-mer overlap (especially for shorter k-mers)
  4. Similar DNAshape profiles (because shape depends on local sequence)
  
  The only difference is the ARRANGEMENT of dinucleotides - the specific
  motif pattern (e.g., GGGCGG for SP1) is disrupted. But with 101bp,
  the TF binding motif is only ~6-10bp in the CENTER. The rest of the
  sequence is flanking genomic context that is ALSO GC-rich.
  
  For a 101bp window with a ~8bp core motif, the motif represents only
  ~8% of the sequence. The model must detect this tiny signal in a sea
  of very similar sequence context.
""")
