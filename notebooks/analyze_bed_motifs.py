#!/usr/bin/env python3
"""
analyze_bed_motifs.py -- Downstream motif / composition analysis of model
predictions (CPU-only, no GPU / DNABERT-2 needed).

Given the per-category BED files exported by the binary tri-branch trainer
(notebooks/28: True_Positive.bed / False_Negative.bed / False_Positive.bed),
or the older 4-class style outputs (True_SP1.bed / Confused_*.bed, etc.), this
script recovers the underlying genomic sequences from the project FASTA files
and computes, per prediction category:

  (a) GC-box consensus motif frequency
        - SP1/Sp-family GC-box:        GGGCGG | CCGCCC   (either strand)
        - canonical SP1 9-mer GC-box:  GGGGCGGGG
  (b) mean GC content
  (c) mean CpG observed/expected (O/E) ratio
        O/E = (N_CG * L) / (N_C * N_G)         [matches notebooks/28 bio-branch]
  (d) G-quadruplex (G4) motif frequency
        regex: (G{3,}[ACGTN]{1,7}){3,}G{3,}    [matches notebooks/28 bio-branch]

These quantify the biological claim that correctly-detected SP-binding sites are
GC-box / CpG-island / G4 enriched relative to missed positives and false
positives.

Outputs (written to --out_dir, default = --pred_dir):
  * motif_analysis.csv   -- one row per category, all metrics
  * motif_analysis.png   -- grouped bar chart of the four frequency/ratio metrics
  * a summary table printed to stdout

Sequence recovery
-----------------
Each BED line is "chrom<TAB>start<TAB>end". We build a lookup from the FASTA
headers:
  positives: ">chr8:142777203-142777304" (and "..._revcomp")  -> key chr8:start-end
  negatives: ">neg_sp2_dinuc_shuffle|chr9:87793641-87793742"  -> key chr9:start-end
A BED line is matched by its (chrom, start, end) key against this lookup. Lines
with no matching FASTA record are counted and reported (the prediction BED may
have been produced against an older/different negative set), but never crash the
run.

Robustness: missing or empty BED files are skipped with a warning; categories
with zero recovered sequences are skipped (not written to the CSV/plot).

Usage
-----
  python notebooks/analyze_bed_motifs.py
  python notebooks/analyze_bed_motifs.py --pred_dir figures/outputs_tribranch_shap
  python notebooks/analyze_bed_motifs.py --pred_dir <dir> --fasta_dir data/processed \
                                         --out_dir <dir>
"""

import argparse
import csv
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# `regex` is available locally and is what notebooks/28 uses; fall back to `re`
# (the patterns here are also valid stdlib-`re` patterns).
try:
    import regex as _re
except ImportError:  # pragma: no cover - regex is installed per project facts
    import re as _re


# ═══════════════════════════════════════════════════════════════════════
# Motif patterns
# ═══════════════════════════════════════════════════════════════════════

# Sp-family GC-box core hexamers (sense GGGCGG; CCGCCC is its rev-comp, i.e. the
# motif on the opposite strand). Matching both makes the count strand-agnostic.
GC_BOX_PATTERN = _re.compile(r"GGGCGG|CCGCCC", _re.IGNORECASE)
# Canonical SP1 GC-box 9-mer consensus.
SP1_CANONICAL_PATTERN = _re.compile(r"GGGGCGGGG", _re.IGNORECASE)
# G-quadruplex putative-forming sequence (identical to notebooks/28 bio-branch).
G4_PATTERN = _re.compile(r"(G{3,}[ACGTN]{1,7}){3,}G{3,}", _re.IGNORECASE)


# BED file name groups we know how to interpret. Anything else found in the
# directory that ends in .bed is also analysed under its bare stem name.
KNOWN_BINARY = ["True_Positive", "False_Negative", "False_Positive"]
# Friendly descriptions for the binary categories (used only for the printout).
CATEGORY_DESC = {
    "True_Positive": "Correctly-detected positives (TP)",
    "False_Negative": "Missed positives (FN)",
    "False_Positive": "False positives (FP)",
}


# ═══════════════════════════════════════════════════════════════════════
# File location helpers (mirrors notebooks/28 auto-detection conventions)
# ═══════════════════════════════════════════════════════════════════════

def find_file(filename, fallback_dir="data/processed"):
    """Search for filename in absolute paths, Kaggle input, fallback dirs, CWD."""
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if filename in files:
                return os.path.join(root, filename)
    if fallback_dir and os.path.exists(fallback_dir):
        p1 = os.path.join(fallback_dir, filename)
        if os.path.exists(p1):
            return p1
        for root, _, files in os.walk(fallback_dir):
            if filename in files:
                return os.path.join(root, filename)
    if os.path.exists(filename):
        return filename
    return None


def auto_detect_pred_dir():
    """Find a directory that contains at least one recognised prediction BED."""
    candidates = [
        "figures/outputs_tribranch_shap",
        ".",
        "figures",
        "outputs_tribranch_shap",
    ]
    targets = [n + ".bed" for n in KNOWN_BINARY] + ["True_SP1.bed"]
    # First, direct candidates.
    for cand in candidates:
        if os.path.isdir(cand) and _dir_has_bed(cand):
            return cand
    # Then, walk figures/ looking for any directory with a known BED.
    search_roots = [r for r in ["figures", "."] if os.path.isdir(r)]
    for root_dir in search_roots:
        for root, _, files in os.walk(root_dir):
            if any(t in files for t in targets):
                return root
    return "."


def _dir_has_bed(d):
    try:
        return any(f.lower().endswith(".bed") for f in os.listdir(d))
    except OSError:
        return False


# ═══════════════════════════════════════════════════════════════════════
# FASTA loading + coordinate lookup
# ═══════════════════════════════════════════════════════════════════════

def load_fasta(filepath):
    """Return (sequences, headers); sequences upper-cased. Empty if missing."""
    sequences, headers = [], []
    if not filepath or not os.path.exists(filepath):
        return sequences, headers
    with open(filepath, "r") as f:
        seq_lines, current_header = [], None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    sequences.append("".join(seq_lines).upper())
                    headers.append(current_header)
                    seq_lines = []
                current_header = line[1:]
            else:
                seq_lines.append(line)
        if current_header is not None:
            sequences.append("".join(seq_lines).upper())
            headers.append(current_header)
    return sequences, headers


def header_to_key(header):
    """Map a FASTA header to a (chrom, start, end) coordinate key, or None.

    positives:  'chr8:142777203-142777304'   and  '..._revcomp'
                -> chrom is the text before ':'
    negatives:  'neg_sp2_dinuc_shuffle|chr9:87793641-87793742'
                -> take the part after '|', then chrom before ':'
    """
    try:
        clean = header.split("_revcomp")[0]
        if "|" in clean:
            clean = clean.split("|", 1)[1]
        chrom, coords = clean.split(":")
        start, end = coords.split("-")
        int(start)
        int(end)
        return (chrom, start, end)
    except (ValueError, IndexError):
        return None


def build_coord_lookup(fasta_dir):
    """Build SEPARATE positive- and negative-set coordinate lookups.

    IMPORTANT: the dinucleotide-shuffled negatives reuse the genomic coordinates
    of the positive regions (header '...|chrX:start-end' is the *source* locus),
    so positive and negative sequences collide on the same (chrom,start,end) key.
    We therefore keep two dicts. BED categories are matched against the
    appropriate pool (TP/FN -> positives, FP -> negatives), with a fallback to
    the other pool only when a key is absent from the preferred one.

    Returns (pos_lookup, neg_lookup, info_lines).
    """
    pos_files = {
        "sp1": "sp1_positive_final.fasta",
        "sp2": "sp2_positive_final.fasta",
        "sp4": "sp4_positive_final.fasta",
    }
    neg_files = {"negative": "negative_final.fasta"}

    def _load_into(files, info):
        lookup = {}
        for tag, fname in files.items():
            path = find_file(fname, fasta_dir)
            if not path:
                info.append("  [warn] FASTA not found: %s (skipped)" % fname)
                continue
            seqs, hdrs = load_fasta(path)
            added = 0
            for seq, hdr in zip(seqs, hdrs):
                key = header_to_key(hdr)
                if key is None:
                    continue
                # Keep first occurrence; a revcomp record shares its forward
                # mate's coordinate key (same genomic interval).
                if key not in lookup:
                    lookup[key] = seq
                    added += 1
            info.append("  [ok]   %-9s %4d records, %4d coords -> %s"
                        % (tag, len(seqs), added, os.path.basename(path)))
        return lookup

    info = []
    pos_lookup = _load_into(pos_files, info)
    neg_lookup = _load_into(neg_files, info)
    return pos_lookup, neg_lookup, info


# ═══════════════════════════════════════════════════════════════════════
# BED parsing + sequence recovery
# ═══════════════════════════════════════════════════════════════════════

def parse_bed_line(line):
    """Return (chrom, start, end) from a BED line, or None if not parseable."""
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 3:
        parts = line.split()
    if len(parts) < 3:
        return None
    chrom, start, end = parts[0], parts[1], parts[2]
    if not chrom:
        return None
    try:
        int(start)
        int(end)
    except ValueError:
        return None
    return (chrom, start, end)


def recover_sequences(bed_path, primary, fallback):
    """Recover sequences for one BED file.

    `primary` / `fallback` are coordinate->sequence dicts. A BED interval is
    looked up in `primary` first, then `fallback` (used when a category's source
    pool is ambiguous). Returns (sequences, n_lines, n_unmatched). Exact-duplicate
    BED lines (forward/revcomp pairs share an interval) contribute once.
    """
    seqs = []
    seen = set()
    n_lines = 0
    n_unmatched = 0
    with open(bed_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            key = parse_bed_line(line)
            if key is None:
                continue
            n_lines += 1
            if key in seen:
                continue
            seen.add(key)
            seq = primary.get(key)
            if seq is None and fallback is not None:
                seq = fallback.get(key)
            if seq is None:
                n_unmatched += 1
                continue
            seqs.append(seq)
    return seqs, n_lines, n_unmatched


# ═══════════════════════════════════════════════════════════════════════
# Per-sequence metrics
# ═══════════════════════════════════════════════════════════════════════

def gc_content(seq):
    L = len(seq)
    if L == 0:
        return 0.0
    return (seq.count("C") + seq.count("G")) / L


def cpg_oe(seq):
    """CpG observed/expected ratio, matching notebooks/28: (N_CG * L)/(N_C * N_G)."""
    L = len(seq)
    if L == 0:
        return 0.0
    c = seq.count("C")
    g = seq.count("G")
    cg = seq.count("CG")
    if c * g == 0:
        return 0.0
    return (cg * L) / (c * g)


def has_gc_box(seq):
    return GC_BOX_PATTERN.search(seq) is not None


def has_sp1_canonical(seq):
    return SP1_CANONICAL_PATTERN.search(seq) is not None


def has_g4(seq):
    return G4_PATTERN.search(seq) is not None


def analyze_sequences(seqs):
    """Aggregate metrics for one category. Returns a dict of summary stats."""
    n = len(seqs)
    if n == 0:
        return None
    gc_box = sum(has_gc_box(s) for s in seqs) / n
    sp1_canon = sum(has_sp1_canonical(s) for s in seqs) / n
    g4 = sum(has_g4(s) for s in seqs) / n
    gc = float(np.mean([gc_content(s) for s in seqs]))
    cpg = float(np.mean([cpg_oe(s) for s in seqs]))
    return {
        "n_sequences": n,
        "gc_box_freq": gc_box,            # GGGCGG | CCGCCC
        "sp1_canonical_freq": sp1_canon,  # GGGGCGGGG
        "mean_gc_content": gc,
        "mean_cpg_oe": cpg,
        "g4_freq": g4,
    }


# ═══════════════════════════════════════════════════════════════════════
# Output: CSV, plot, stdout table
# ═══════════════════════════════════════════════════════════════════════

CSV_FIELDS = [
    "category",
    "description",
    "n_sequences",
    "gc_box_freq",
    "sp1_canonical_freq",
    "mean_gc_content",
    "mean_cpg_oe",
    "g4_freq",
]


def write_csv(rows, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def plot_grouped_bar(rows, out_path):
    """Grouped bar chart of the four key metrics across categories."""
    metrics = [
        ("gc_box_freq", "GC-box freq\n(GGGCGG|CCGCCC)"),
        ("sp1_canonical_freq", "SP1 9-mer freq\n(GGGGCGGGG)"),
        ("mean_gc_content", "Mean GC content"),
        ("g4_freq", "G4 motif freq"),
    ]
    categories = [r["category"] for r in rows]
    n_cat = len(categories)
    n_metric = len(metrics)
    x = np.arange(n_metric)
    width = 0.8 / max(n_cat, 1)
    colors = ["#2E7D32", "#F9A825", "#C62828", "#1565C0", "#6A1B9A", "#00838F"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, r in enumerate(rows):
        vals = [r[m[0]] for m in metrics]
        offset = (i - (n_cat - 1) / 2.0) * width
        bars = ax.bar(x + offset, vals, width,
                      label="%s (n=%d)" % (r["category"], r["n_sequences"]),
                      color=colors[i % len(colors)], alpha=0.9)
        for bar in bars:
            ax.annotate("%.2f" % bar.get_height(),
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylabel("Frequency / fraction")
    ax.set_title("Motif & composition analysis of prediction categories",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.set_ylim(0, max(1.05, ax.get_ylim()[1]))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def print_table(rows):
    cols = [
        ("category", "Category", 18, "s"),
        ("n_sequences", "N", 6, "d"),
        ("gc_box_freq", "GC-box", 8, ".3f"),
        ("sp1_canonical_freq", "SP1-9mer", 9, ".3f"),
        ("mean_gc_content", "GC%", 7, ".3f"),
        ("mean_cpg_oe", "CpG O/E", 8, ".3f"),
        ("g4_freq", "G4", 7, ".3f"),
    ]
    header = "  ".join(("{:>%d}" % w).format(title) for _, title, w, _ in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = []
        for key, _, w, fmt in cols:
            val = r[key]
            if fmt == "s":
                cells.append(("{:>%d}" % w).format(str(val)[:w]))
            elif fmt == "d":
                cells.append(("{:>%dd}" % w).format(int(val)))
            else:
                cells.append(("{:>%d%s}" % (w, fmt)).format(val))
        print("  ".join(cells))


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def discover_bed_files(pred_dir):
    """Return ordered list of (category_name, path) for .bed files in pred_dir.

    Known binary categories first (in canonical order), then any other .bed
    files found in the directory (e.g. older True_SP1.bed / Confused_*.bed).
    """
    found = {}
    try:
        entries = sorted(os.listdir(pred_dir))
    except OSError:
        entries = []
    for fname in entries:
        if not fname.lower().endswith(".bed"):
            continue
        stem = fname[:-4]
        found[stem] = os.path.join(pred_dir, fname)

    ordered = []
    for name in KNOWN_BINARY:
        if name in found:
            ordered.append((name, found.pop(name)))
    for name in sorted(found):
        ordered.append((name, found[name]))
    return ordered


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Downstream motif/composition analysis of model prediction "
                    "BED files (CPU-only).")
    parser.add_argument("--pred_dir", default=None,
                        help="Directory containing the prediction BED files "
                             "(True_Positive.bed etc.). Auto-detected if omitted.")
    parser.add_argument("--fasta_dir", default="data/processed",
                        help="Directory containing project FASTA files "
                             "(default: data/processed).")
    parser.add_argument("--out_dir", default=None,
                        help="Where to write motif_analysis.csv / .png "
                             "(default: --pred_dir).")
    args = parser.parse_args(argv)

    pred_dir = args.pred_dir or auto_detect_pred_dir()
    if not os.path.isdir(pred_dir):
        print("[error] --pred_dir not found: %s" % pred_dir)
        return 1
    out_dir = args.out_dir or pred_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 64)
    print("MOTIF / COMPOSITION ANALYSIS OF PREDICTION CATEGORIES")
    print("=" * 64)
    print("  pred_dir : %s" % os.path.abspath(pred_dir))
    print("  fasta_dir: %s" % os.path.abspath(args.fasta_dir))
    print("  out_dir  : %s" % os.path.abspath(out_dir))

    print("\nBuilding coordinate -> sequence lookup from FASTA files:")
    pos_lookup, neg_lookup, info = build_coord_lookup(args.fasta_dir)
    for line in info:
        print(line)
    print("  -> %d positive coords, %d negative coords indexed"
          % (len(pos_lookup), len(neg_lookup)))
    if not pos_lookup and not neg_lookup:
        print("[error] No FASTA records loaded; cannot recover sequences. "
              "Check --fasta_dir.")
        return 1

    bed_files = discover_bed_files(pred_dir)
    if not bed_files:
        print("\n[error] No .bed files found in %s" % pred_dir)
        return 1

    print("\nProcessing BED files:")
    rows = []
    for category, path in bed_files:
        if not os.path.exists(path):
            print("  [warn] %-16s missing -> skipped" % category)
            continue
        if os.path.getsize(path) == 0:
            print("  [warn] %-16s empty -> skipped" % category)
            continue
        # False positives are negatives the model called positive -> their
        # sequences live in the negative pool. All other categories (TP, FN,
        # True_SP*, Confused_*) are positive-derived. Fallback to the other pool
        # handles odd/legacy exports.
        if category.lower().startswith("false_positive"):
            primary, fallback = neg_lookup, pos_lookup
        else:
            primary, fallback = pos_lookup, neg_lookup
        seqs, n_lines, n_unmatched = recover_sequences(path, primary, fallback)
        if not seqs:
            print("  [warn] %-16s %d lines, 0 sequences recovered "
                  "(%d unmatched coords) -> skipped"
                  % (category, n_lines, n_unmatched))
            continue
        stats = analyze_sequences(seqs)
        stats["category"] = category
        stats["description"] = CATEGORY_DESC.get(category, "")
        rows.append(stats)
        note = ""
        if n_unmatched:
            note = "  [%d/%d unique coords unmatched -> not in FASTA set]" % (
                n_unmatched, n_unmatched + len(seqs))
        print("  [ok]   %-16s %d lines -> %d sequences%s"
              % (category, n_lines, len(seqs), note))

    if not rows:
        print("\n[error] No category yielded recoverable sequences; "
              "nothing to summarise.")
        return 1

    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print_table(rows)

    csv_path = os.path.join(out_dir, "motif_analysis.csv")
    png_path = os.path.join(out_dir, "motif_analysis.png")
    write_csv(rows, csv_path)
    plot_grouped_bar(rows, png_path)

    print("\nSaved:")
    print("  %s" % os.path.abspath(csv_path))
    print("  %s" % os.path.abspath(png_path))
    print("\nNote: 'GC-box', 'SP1-9mer' and 'G4' columns are the FRACTION of "
          "sequences\n      containing the motif; 'CpG O/E' and 'GC%' are means.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
