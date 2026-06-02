#!/usr/bin/env python3
"""
generate_negatives_v2.py — Biologically Sound Negative Dataset Generation

Implements two strategies:
  Plan B: GC-matched random genomic regions (PRIMARY, gold-standard)
           Following DNABERT (Ji et al., 2021) and DNABERT-2 (Zhou et al., 2023)
  Plan C: CpG island regions not bound by SP1/SP2/SP4 (SUPPLEMENTARY)
           Hard negatives from GC-rich functional regions for ablation study

References:
  - DNABERT (Ji et al., 2021): GC-matched genomic negatives for 690 ENCODE benchmark
  - DNABERT-2 / GUE benchmark (Zhou et al., 2023): Same methodology
  - Tourné et al. (2026): Systematic benchmark confirming genomic > shuffle
  - ENCODE Blacklist V2 (ENCFF356LFX): Artifact region exclusion

Designed for Google Colab with hg38.fa on Google Drive.

Usage:
  python generate_negatives_v2.py \\
    --fasta /content/drive/MyDrive/ML_Project/data/hg38.fa \\
    --bed_dir data/processed/downsampled_101bp \\
    --pos_dir data/processed/positive_datasets_fasta \\
    --out_dir data/processed \\
    --plan both
"""

import os
import sys
import random
import argparse
import time
from collections import Counter, defaultdict
from bisect import bisect_left, bisect_right

# Optional imports for downloads
try:
    import urllib.request
    import gzip
    HAS_DOWNLOAD = True
except ImportError:
    HAS_DOWNLOAD = False

# ============================================================
# Configuration
# ============================================================

VALID_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
SEQ_LEN = 101
PEAK_PADDING = 500        # Exclude peak regions ± 500bp
TARGET_NEG_COUNT = 3392   # Match positive class size
GC_BIN_SIZE = 0.02        # 2% GC bins for matching
MAX_N_FRACTION = 0.05     # Allow up to 5% N characters
N_CANDIDATES_DEFAULT = 500000  # Initial random candidate pool size

BLACKLIST_URL = ("https://www.encodeproject.org/files/ENCFF356LFX/"
                 "@@download/ENCFF356LFX.bed.gz")
CPG_ISLAND_URL = ("https://hgdownload.cse.ucsc.edu/goldenpath/hg38/"
                  "database/cpgIslandExt.txt.gz")


# ============================================================
# FastaIndexedReader (adapted from extract_fasta.py)
# ============================================================

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
                f"Create it: samtools faidx {os.path.basename(self.fa_path)}")

        self._load_index()
        self.fa_file = open(self.fa_path, 'rb')

    def _load_index(self):
        with open(self.fai_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    self.index[parts[0]] = {
                        'length': int(parts[1]),
                        'offset': int(parts[2]),
                        'line_bases': int(parts[3]),
                        'line_bytes': int(parts[4]),
                    }

    def get_sequence(self, chrom, start, end):
        """Extract sequence from chrom:start-end (0-based, half-open)."""
        if chrom not in self.index:
            return None
        info = self.index[chrom]
        start = max(0, start)
        end = min(end, info['length'])
        if start >= end:
            return ""

        line_bases = info['line_bases']
        line_bytes = info['line_bytes']
        start_line = start // line_bases
        start_offset = start % line_bases
        start_byte = info['offset'] + start_line * line_bytes + start_offset

        length = end - start
        num_lines = (start_offset + length + line_bases - 1) // line_bases
        bytes_to_read = length + num_lines * (line_bytes - line_bases) + 10

        self.fa_file.seek(start_byte)
        data = self.fa_file.read(bytes_to_read)
        seq = data.decode('ascii', errors='ignore'
                          ).replace('\n', '').replace('\r', '')
        return seq[:length]

    def get_chrom_sizes(self):
        return {c: self.index[c]['length']
                for c in self.index if c in VALID_CHROMS}

    def close(self):
        if self.fa_file:
            self.fa_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ============================================================
# ExclusionZones — Merged interval tree with binary search
# ============================================================

class ExclusionZones:
    """Manages genomic intervals to exclude (peaks + blacklist)."""

    def __init__(self):
        self._raw = defaultdict(list)
        self._merged = {}
        self._starts = {}

    def add(self, chrom, start, end):
        self._raw[chrom].append((start, end))

    def finalize(self):
        """Sort and merge overlapping intervals per chromosome."""
        total = 0
        total_bp = 0
        for chrom in self._raw:
            sorted_ivs = sorted(self._raw[chrom])
            merged = []
            for s, e in sorted_ivs:
                if merged and s <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))
            self._merged[chrom] = merged
            self._starts[chrom] = [iv[0] for iv in merged]
            total += len(merged)
            total_bp += sum(e - s for s, e in merged)
        print(f"  Exclusion zones: {total:,} merged intervals, "
              f"{total_bp / 1e6:.1f} Mbp across {len(self._merged)} chroms")

    def overlaps(self, chrom, start, end):
        """Check if region [start, end) overlaps any exclusion zone."""
        if chrom not in self._merged:
            return False
        intervals = self._merged[chrom]
        starts = self._starts[chrom]
        # Find last interval starting before 'end'
        idx = bisect_left(starts, end) - 1
        if idx >= 0 and intervals[idx][1] > start:
            return True
        if idx + 1 < len(intervals) and intervals[idx + 1][0] < end:
            return True
        return False


# ============================================================
# Utility Functions
# ============================================================

def gc_content(seq):
    """GC content as fraction (0.0–1.0)."""
    s = seq.upper()
    gc = s.count('G') + s.count('C')
    return gc / len(s) if len(s) > 0 else 0.0


def n_fraction(seq):
    """Fraction of N characters."""
    return seq.upper().count('N') / len(seq) if seq else 1.0


def stdev(values):
    """Standard deviation (sample)."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return (sum((x - mean) ** 2 for x in values) / (n - 1)) ** 0.5


def read_fasta(filepath):
    """Parse FASTA file → list of (header, sequence)."""
    records = []
    with open(filepath, 'r') as f:
        header, seq_parts = "", []
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
    """Write list of (header, sequence) to FASTA."""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w', newline='\n') as f:
        for header, seq in records:
            f.write(f"{header}\n{seq.upper()}\n")


def read_bed(filepath):
    """Read BED file → list of (chrom, start, end)."""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                try:
                    records.append((parts[0], int(parts[1]), int(parts[2])))
                except ValueError:
                    continue
    return records


# ============================================================
# Download Helpers
# ============================================================

def download_bed_gz(url, label, cache_dir):
    """Download a gzipped BED file, cache locally, return intervals."""
    if not HAS_DOWNLOAD:
        print(f"  ⚠ urllib not available, skipping {label}")
        return []

    os.makedirs(cache_dir, exist_ok=True)
    basename = url.split('/')[-1]
    cache_path = os.path.join(cache_dir, basename)

    if not os.path.exists(cache_path):
        print(f"  Downloading {label}...")
        try:
            urllib.request.urlretrieve(url, cache_path)
        except Exception as e:
            print(f"  ⚠ Download failed: {e}")
            return []
    else:
        print(f"  Using cached {label}")

    intervals = []
    with gzip.open(cache_path, 'rt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                try:
                    intervals.append((parts[0], int(parts[1]), int(parts[2])))
                except ValueError:
                    continue
    print(f"  Loaded {len(intervals):,} intervals from {label}")
    return intervals


def download_cpg_islands(cache_dir):
    """Download CpG island annotations from UCSC."""
    if not HAS_DOWNLOAD:
        print(f"  ⚠ urllib not available, skipping CpG islands")
        return []

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "cpgIslandExt.txt.gz")

    if not os.path.exists(cache_path):
        print(f"  Downloading CpG island annotations from UCSC...")
        try:
            urllib.request.urlretrieve(CPG_ISLAND_URL, cache_path)
        except Exception as e:
            print(f"  ⚠ Download failed: {e}")
            return []
    else:
        print(f"  Using cached CpG islands")

    islands = []
    with gzip.open(cache_path, 'rt') as f:
        for line in f:
            parts = line.strip().split('\t')
            # Format: bin, chrom, chromStart, chromEnd, name, ...
            if len(parts) >= 4:
                try:
                    islands.append((parts[1], int(parts[2]), int(parts[3])))
                except (ValueError, IndexError):
                    continue
    print(f"  Loaded {len(islands):,} CpG islands")
    return islands


# ============================================================
# Plan B: GC-matched Random Genomic Regions
# ============================================================

def generate_random_candidates(reader, chrom_sizes, exclusion,
                               n_candidates, rng):
    """Generate random 101bp candidate regions from the genome."""
    print(f"\n  Generating {n_candidates:,} random candidate regions...")

    # Build chromosome-proportional sampling
    chroms = sorted(chrom_sizes.keys())
    sizes = [chrom_sizes[c] for c in chroms]
    total_size = sum(sizes)
    cum_sizes = []
    cum = 0
    for s in sizes:
        cum += s
        cum_sizes.append(cum)

    # Generate random positions, grouped by chromosome
    candidates_by_chrom = defaultdict(list)
    for _ in range(n_candidates):
        r = rng.randint(0, total_size - 1)
        idx = bisect_right(cum_sizes, r)
        chrom = chroms[idx]
        offset = r - (cum_sizes[idx - 1] if idx > 0 else 0)
        start = offset
        if start + SEQ_LEN > chrom_sizes[chrom]:
            start = chrom_sizes[chrom] - SEQ_LEN
        candidates_by_chrom[chrom].append(start)

    # Sort within each chromosome for sequential disk reads
    for chrom in candidates_by_chrom:
        candidates_by_chrom[chrom].sort()

    # Extract sequences and filter
    valid = []
    n_excluded = 0
    n_has_n = 0
    n_bad_len = 0
    processed = 0

    t0 = time.time()
    for chrom in sorted(candidates_by_chrom.keys()):
        positions = candidates_by_chrom[chrom]
        for start in positions:
            processed += 1
            if processed % 100000 == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"    Progress: {processed:,}/{n_candidates:,} "
                      f"({processed/n_candidates*100:.0f}%) "
                      f"[{rate:.0f} seq/s, valid={len(valid):,}]")

            end = start + SEQ_LEN

            # Check exclusion zones
            if exclusion.overlaps(chrom, start, end):
                n_excluded += 1
                continue

            # Extract sequence
            seq = reader.get_sequence(chrom, start, end)
            if seq is None or len(seq) != SEQ_LEN:
                n_bad_len += 1
                continue

            seq = seq.upper()

            # Check N content
            if n_fraction(seq) > MAX_N_FRACTION:
                n_has_n += 1
                continue

            gc = gc_content(seq)
            valid.append((chrom, start, end, seq, gc))

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({processed/elapsed:.0f} seq/s)")
    print(f"  Valid candidates: {len(valid):,} / {n_candidates:,}")
    print(f"  Filtered out: {n_excluded:,} peak/blacklist, "
          f"{n_has_n:,} N-content, {n_bad_len:,} bad length")

    return valid


def gc_match_sample(candidates, positive_gc_values, target_n,
                    bin_size, rng):
    """Sample candidates to match the GC distribution of positives."""
    print(f"\n  GC-matching {len(candidates):,} candidates → "
          f"{target_n:,} targets...")

    # Build positive GC histogram
    pos_hist = defaultdict(int)
    for gc in positive_gc_values:
        bin_idx = int(gc / bin_size)
        pos_hist[bin_idx] += 1

    # Target counts per bin (proportional to positive distribution)
    total_pos = sum(pos_hist.values())
    target_per_bin = {}
    for bin_idx, count in pos_hist.items():
        target_per_bin[bin_idx] = max(1, round(count / total_pos * target_n))

    # Adjust rounding error to hit exact target_n
    current_total = sum(target_per_bin.values())
    if current_total != target_n:
        largest_bin = max(target_per_bin, key=target_per_bin.get)
        target_per_bin[largest_bin] += (target_n - current_total)

    # Group candidates by GC bin
    cand_bins = defaultdict(list)
    for cand in candidates:
        bin_idx = int(cand[4] / bin_size)
        cand_bins[bin_idx].append(cand)

    # Sample from each bin
    selected = []
    shortfall_total = 0
    for bin_idx in sorted(target_per_bin.keys()):
        target_count = target_per_bin[bin_idx]
        available = cand_bins.get(bin_idx, [])
        gc_lo = bin_idx * bin_size * 100
        gc_hi = (bin_idx + 1) * bin_size * 100

        if len(available) >= target_count:
            selected.extend(rng.sample(available, target_count))
        else:
            selected.extend(available)
            deficit = target_count - len(available)
            shortfall_total += deficit
            if deficit > 0:
                print(f"    ⚠ Bin [{gc_lo:.0f}%-{gc_hi:.0f}%]: "
                      f"need {target_count}, have {len(available)}, "
                      f"deficit {deficit}")

    # Fill shortfall from nearest available bins
    if shortfall_total > 0:
        remaining = target_n - len(selected)
        print(f"  ⚠ Total shortfall: {shortfall_total}. "
              f"Filling {remaining} from nearest bins...")

        selected_ids = set(id(c) for c in selected)
        unused = [c for c in candidates if id(c) not in selected_ids]

        # Sort unused by distance to mean positive GC
        mean_gc = sum(positive_gc_values) / len(positive_gc_values)
        unused.sort(key=lambda c: abs(c[4] - mean_gc))
        fill = unused[:remaining]
        selected.extend(fill)
        print(f"  Filled {len(fill)} from neighboring bins")

    rng.shuffle(selected)
    result = selected[:target_n]

    # Report
    if result:
        result_gc = [c[4] for c in result]
        print(f"\n  ✓ Selected {len(result):,} GC-matched negatives")
        print(f"    GC: mean={sum(result_gc)/len(result_gc)*100:.1f}%, "
              f"std={stdev(result_gc)*100:.1f}%")

    return result


# ============================================================
# Plan C: CpG Island Negatives
# ============================================================

def generate_cpg_negatives(reader, cpg_islands, exclusion,
                           target_n, rng):
    """Generate negatives from CpG island regions not overlapping peaks."""
    print(f"\n  Extracting candidates from {len(cpg_islands):,} CpG islands...")

    # Filter to valid chromosomes
    valid = [(c, s, e) for c, s, e in cpg_islands if c in VALID_CHROMS]
    rng.shuffle(valid)
    print(f"  Valid CpG islands (autosomes + chrX): {len(valid):,}")

    candidates = []
    n_excluded = 0
    n_has_n = 0
    n_short = 0

    t0 = time.time()
    for chrom, island_start, island_end in valid:
        island_len = island_end - island_start

        # Generate 101bp windows from the island
        if island_len < SEQ_LEN:
            # Center a single window
            center = (island_start + island_end) // 2
            ws = center - SEQ_LEN // 2
            windows = [(ws, ws + SEQ_LEN)]
        else:
            # Tile non-overlapping windows across the island
            windows = []
            for pos in range(island_start, island_end - SEQ_LEN + 1, SEQ_LEN):
                windows.append((pos, pos + SEQ_LEN))
            # Also add one centered window for variety
            center = (island_start + island_end) // 2
            ws = center - SEQ_LEN // 2
            windows.append((ws, ws + SEQ_LEN))

        for ws, we in windows:
            if exclusion.overlaps(chrom, ws, we):
                n_excluded += 1
                continue

            seq = reader.get_sequence(chrom, ws, we)
            if seq is None or len(seq) != SEQ_LEN:
                n_short += 1
                continue

            seq = seq.upper()
            if n_fraction(seq) > MAX_N_FRACTION:
                n_has_n += 1
                continue

            gc = gc_content(seq)
            candidates.append((chrom, ws, we, seq, gc))

        # Early stop if we have enough candidates (3x target for diversity)
        if len(candidates) >= target_n * 3:
            break

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Valid CpG candidates: {len(candidates):,}")
    print(f"  Filtered: {n_excluded:,} peak overlap, "
          f"{n_has_n:,} N-content, {n_short:,} too short")

    # Sample target_n
    if len(candidates) >= target_n:
        selected = rng.sample(candidates, target_n)
    else:
        selected = candidates
        print(f"  ⚠ Only {len(selected):,} available (target: {target_n:,})")

    if selected:
        gc_vals = [c[4] for c in selected]
        print(f"  ✓ Selected {len(selected):,} CpG negatives")
        print(f"    GC: mean={sum(gc_vals)/len(gc_vals)*100:.1f}%, "
              f"std={stdev(gc_vals)*100:.1f}%")

    return selected


# ============================================================
# QC Report
# ============================================================

def print_qc_report(pos_gc, neg_b_gc, neg_c_gc=None):
    """Print QC comparison of GC distributions."""
    print(f"\n{'='*70}")
    print(f"  QC REPORT — GC-content Distribution Comparison")
    print(f"{'='*70}")

    def show_stats(values, label):
        n = len(values)
        mean = sum(values) / n * 100
        sd = stdev(values) * 100
        mn = min(values) * 100
        mx = max(values) * 100
        print(f"  {label:<35s} n={n:>5,}  "
              f"GC={mean:>5.1f}% ± {sd:>4.1f}%  "
              f"[{mn:.0f}%-{mx:.0f}%]")

    show_stats(pos_gc, "Positive (SP1+SP2+SP4 original)")
    show_stats(neg_b_gc, "Plan B (GC-matched genomic)")
    if neg_c_gc:
        show_stats(neg_c_gc, "Plan C (CpG island)")

    # GC histogram
    print(f"\n  GC% Histogram (5% bins):")
    header = f"  {'GC Range':<12s} {'Positive':>10s} {'Plan B':>10s}"
    if neg_c_gc:
        header += f" {'Plan C':>10s}"
    print(header)
    print(f"  {'-'*12} {'-'*10} {'-'*10}", end="")
    if neg_c_gc:
        print(f" {'-'*10}", end="")
    print()

    for lo in range(25, 95, 5):
        hi = lo + 5
        p_n = sum(1 for g in pos_gc if lo/100 <= g < hi/100)
        b_n = sum(1 for g in neg_b_gc if lo/100 <= g < hi/100)
        line = f"  {lo:>3d}-{hi:<3d}%    {p_n:>10,} {b_n:>10,}"
        if neg_c_gc:
            c_n = sum(1 for g in neg_c_gc if lo/100 <= g < hi/100)
            line += f" {c_n:>10,}"
        print(line)

    # Cohen's d between positive and Plan B (should be small)
    mean_p = sum(pos_gc) / len(pos_gc)
    mean_b = sum(neg_b_gc) / len(neg_b_gc)
    sd_p = stdev(pos_gc)
    sd_b = stdev(neg_b_gc)
    pooled_sd = ((sd_p**2 + sd_b**2) / 2) ** 0.5
    cohens_d = abs(mean_p - mean_b) / pooled_sd if pooled_sd > 0 else 0

    print(f"\n  GC Cohen's d (Positive vs Plan B): {cohens_d:.4f}")
    if cohens_d < 0.2:
        print(f"  → Negligible difference ✓ (GC-matching successful)")
    elif cohens_d < 0.5:
        print(f"  → Small difference (acceptable)")
    else:
        print(f"  → ⚠ Moderate/large difference (check GC bins)")


# ============================================================
# Overlap Verification
# ============================================================

def verify_no_overlap(neg_records, peak_beds, label):
    """Verify that no negative overlaps with any positive peak."""
    print(f"\n  Verifying {label}: no overlap with positive peaks...")

    # Build peak lookup
    peaks_by_chrom = defaultdict(list)
    for bed_path in peak_beds:
        if not os.path.exists(bed_path):
            continue
        for chrom, start, end in read_bed(bed_path):
            peaks_by_chrom[chrom].append((start, end))
    for chrom in peaks_by_chrom:
        peaks_by_chrom[chrom].sort()

    overlaps = 0
    for header, seq in neg_records:
        # Parse header: >chr:start-end
        coord = header.lstrip('>')
        parts = coord.replace(':', '-').split('-')
        if len(parts) >= 3:
            chrom = parts[0]
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                continue

            for ps, pe in peaks_by_chrom.get(chrom, []):
                if ps < end and start < pe:
                    overlaps += 1
                    break

    if overlaps == 0:
        print(f"  ✓ PASSED: 0 overlaps with {sum(len(v) for v in peaks_by_chrom.values()):,} peaks")
    else:
        print(f"  ✗ FAILED: {overlaps} overlaps detected!")

    return overlaps


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate biologically sound negative datasets for TFBS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--fasta", type=str, required=True,
                        help="Path to hg38.fa (with .fai index)")
    parser.add_argument("--bed_dir", type=str,
                        default="data/processed/downsampled_101bp",
                        help="Directory with SP1/SP2/SP4 BED files")
    parser.add_argument("--pos_dir", type=str,
                        default="data/processed/positive_datasets_fasta",
                        help="Directory with positive FASTA files")
    parser.add_argument("--out_dir", type=str,
                        default="data/processed",
                        help="Output directory")
    parser.add_argument("--plan", type=str, default="both",
                        choices=["B", "C", "both"],
                        help="Which plan to execute (default: both)")
    parser.add_argument("--n_candidates", type=int,
                        default=N_CANDIDATES_DEFAULT,
                        help="Random candidates to generate for Plan B")
    parser.add_argument("--target_n", type=int, default=TARGET_NEG_COUNT,
                        help="Target number of negatives per plan")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()
    rng = random.Random(args.seed)
    random.seed(args.seed)

    print("=" * 70)
    print("  NEGATIVE DATASET GENERATION v2")
    print("  Plan B: GC-matched Random Genomic Regions (PRIMARY)")
    print("  Plan C: CpG Island Regions (SUPPLEMENTARY)")
    print("=" * 70)
    print(f"  Reference genome: {args.fasta}")
    print(f"  Target negatives: {args.target_n:,}")
    print(f"  Random seed: {args.seed}")
    print(f"  Plan: {args.plan}")

    # ---- Step 1: Load genome ----
    print(f"\n[1/7] Loading genome index")
    t_start = time.time()
    reader = FastaIndexedReader(args.fasta)
    chrom_sizes = reader.get_chrom_sizes()
    total_bp = sum(chrom_sizes.values())
    print(f"  {len(chrom_sizes)} chromosomes, "
          f"{total_bp/1e9:.2f} Gbp total")

    # ---- Step 2: Build exclusion zones from peaks ----
    print(f"\n[2/7] Building exclusion zones from positive peaks")
    exclusion = ExclusionZones()
    total_peaks = 0
    peak_bed_paths = []

    for tf in ["SP1", "SP2", "SP4"]:
        bed_path = os.path.join(args.bed_dir, f"{tf}_downsampled_101bp.bed")
        if not os.path.exists(bed_path):
            print(f"  ⚠ Not found: {bed_path}")
            continue
        peak_bed_paths.append(bed_path)
        peaks = read_bed(bed_path)
        for chrom, start, end in peaks:
            exclusion.add(chrom,
                          max(0, start - PEAK_PADDING),
                          end + PEAK_PADDING)
        total_peaks += len(peaks)
        print(f"  {tf}: {len(peaks):,} peaks (± {PEAK_PADDING}bp padding)")
    print(f"  Total: {total_peaks:,} peaks")

    # ---- Step 3: Add ENCODE blacklist ----
    print(f"\n[3/7] Loading ENCODE Blacklist V2 (ENCFF356LFX)")
    cache_dir = os.path.join(args.out_dir, ".cache")

    blacklist = download_bed_gz(BLACKLIST_URL, "ENCODE Blacklist V2",
                                cache_dir)
    for chrom, start, end in blacklist:
        exclusion.add(chrom, start, end)

    exclusion.finalize()

    # ---- Step 4: Compute positive GC distribution ----
    print(f"\n[4/7] Computing positive GC distribution")
    positive_gc = []

    for tf in ["SP1", "SP2", "SP4"]:
        fasta_path = os.path.join(args.pos_dir,
                                  f"{tf}_downsampled_101bp.fasta")
        if not os.path.exists(fasta_path):
            print(f"  ⚠ Not found: {fasta_path}")
            continue
        records = read_fasta(fasta_path)
        for _, seq in records:
            positive_gc.append(gc_content(seq))
        print(f"  {tf}: {len(records):,} sequences loaded")

    if not positive_gc:
        print("  ERROR: No positive sequences found!")
        reader.close()
        sys.exit(1)

    mean_gc = sum(positive_gc) / len(positive_gc)
    print(f"  Positive GC: mean={mean_gc*100:.1f}% "
          f"± {stdev(positive_gc)*100:.1f}%  "
          f"(n={len(positive_gc):,})")

    # ---- Step 5: Plan B ----
    neg_b_records = []
    neg_b_gc = []

    if args.plan in ["B", "both"]:
        print(f"\n{'='*70}")
        print(f"  [5/7] PLAN B: GC-matched Random Genomic Regions")
        print(f"{'='*70}")

        candidates = generate_random_candidates(
            reader, chrom_sizes, exclusion, args.n_candidates, rng)

        # Check if enough high-GC candidates; if not, generate more
        high_gc_count = sum(1 for c in candidates if c[4] > 0.55)
        needed_high_gc = sum(1 for g in positive_gc if g > 0.55)
        needed_high_gc = int(needed_high_gc / len(positive_gc)
                             * args.target_n)

        if high_gc_count < needed_high_gc * 1.5:
            print(f"\n  ⚠ Need more high-GC candidates "
                  f"(have {high_gc_count:,}, need ~{needed_high_gc:,})")
            print(f"  Generating additional candidates...")
            extra = generate_random_candidates(
                reader, chrom_sizes, exclusion,
                args.n_candidates, rng)
            candidates.extend(extra)
            print(f"  Total candidates: {len(candidates):,}")

        selected_b = gc_match_sample(
            candidates, positive_gc, args.target_n, GC_BIN_SIZE, rng)

        neg_b_records = [
            (f">{c}:{s}-{e}", seq)
            for c, s, e, seq, gc in selected_b
        ]
        neg_b_gc = [c[4] for c in selected_b]

        out_b = os.path.join(args.out_dir, "negative_genomic_matched.fasta")
        write_fasta(out_b, neg_b_records)
        print(f"\n  ✅ Plan B saved: {out_b}")
        print(f"     {len(neg_b_records):,} sequences")

        # Verify no overlap
        verify_no_overlap(neg_b_records, peak_bed_paths, "Plan B")
    else:
        print(f"\n  [5/7] Plan B: SKIPPED")

    # ---- Step 6: Plan C ----
    neg_c_records = []
    neg_c_gc = []

    if args.plan in ["C", "both"]:
        print(f"\n{'='*70}")
        print(f"  [6/7] PLAN C: CpG Island Regions")
        print(f"{'='*70}")

        cpg_islands = download_cpg_islands(cache_dir)
        if cpg_islands:
            selected_c = generate_cpg_negatives(
                reader, cpg_islands, exclusion, args.target_n, rng)

            neg_c_records = [
                (f">{c}:{s}-{e}", seq)
                for c, s, e, seq, gc in selected_c
            ]
            neg_c_gc = [c[4] for c in selected_c]

            out_c = os.path.join(args.out_dir,
                                 "negative_promoter_cpg.fasta")
            write_fasta(out_c, neg_c_records)
            print(f"\n  ✅ Plan C saved: {out_c}")
            print(f"     {len(neg_c_records):,} sequences")

            # Verify no overlap
            verify_no_overlap(neg_c_records, peak_bed_paths, "Plan C")
        else:
            print(f"  ⚠ No CpG islands loaded, skipping Plan C")
    else:
        print(f"\n  [6/7] Plan C: SKIPPED")

    # ---- Step 7: QC Report ----
    print(f"\n{'='*70}")
    print(f"  [7/7] QC REPORT")
    print(f"{'='*70}")

    if neg_b_gc:
        print_qc_report(positive_gc, neg_b_gc,
                         neg_c_gc if neg_c_gc else None)

    # ---- Final Summary ----
    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Positive reference: {len(positive_gc):,} sequences "
          f"(GC={mean_gc*100:.1f}%)")

    if neg_b_records:
        mb = sum(neg_b_gc) / len(neg_b_gc) if neg_b_gc else 0
        print(f"  Plan B output: {len(neg_b_records):,} sequences "
              f"(GC={mb*100:.1f}%)")
        print(f"    → {os.path.join(args.out_dir, 'negative_genomic_matched.fasta')}")

    if neg_c_records:
        mc = sum(neg_c_gc) / len(neg_c_gc) if neg_c_gc else 0
        print(f"  Plan C output: {len(neg_c_records):,} sequences "
              f"(GC={mc*100:.1f}%)")
        print(f"    → {os.path.join(args.out_dir, 'negative_promoter_cpg.fasta')}")

    print(f"\n  NEXT STEPS:")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  1. Extract DNAshape for new negative files:")
    print(f"     python src/extract_dnashape.py "
          f"--input negative_genomic_matched.fasta "
          f"--output dnashape_negative_genomic.npy")
    if neg_c_records:
        print(f"     python src/extract_dnashape.py "
              f"--input negative_promoter_cpg.fasta "
              f"--output dnashape_negative_cpg.npy")
    print(f"  2. Upload FASTA + .npy files to Kaggle dataset")
    print(f"  3. Update training script to load new negative files")
    print(f"  4. Run binary sanity check:")
    print(f"     Plan B expected accuracy: ~85-92%")
    if neg_c_records:
        print(f"     Plan C expected accuracy: ~80-88% (harder negatives)")
    print(f"{'='*70}")

    reader.close()
    print(f"\nDone.")


if __name__ == "__main__":
    main()
