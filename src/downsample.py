"""
STEP 4: Downsampling — Multi-class Balancing

Inputs:
  - data/processed/filtered_exclusive_101bp/{TF}_exclusive_101bp.bed  (3-col: chrom, start, end)
  - data/processed/filtered_qval2/{TF}_filtered.bed                   (10-col narrowPeak)

Workflow:
  1. Load exclusive BED files (post overlap-removal, 101bp windows).
  2. Load corresponding filtered narrowPeak files for metadata (qval, signal).
  3. Compute summit in both coordinate systems to join them:
       - Exclusive: summit = start + 50
       - Filtered:  summit = original_start + peak (col 10)
  4. Merge to attach qval/signal to each exclusive peak.
  5. Sort descending by qval (highest quality first), signal as tiebreaker.
  6. Take top N = len(SP2) peaks for SP1 and SP4.
  7. Save results.
"""

import os
import numpy as np
import pandas as pd

# Paths
EXCLUSIVE_DIR = os.path.join('data', 'processed', 'filtered_exclusive_101bp')
FILTERED_DIR  = os.path.join('data', 'processed', 'filtered_qval2')
OUTPUT_DIR    = os.path.join('data', 'processed', 'downsampled_101bp')

TFS = ['SP1', 'SP2', 'SP4']
COLS_NARROW = ['chrom', 'start', 'end', 'name', 'score', 'strand',
               'signal', 'pval', 'qval', 'peak']
WINDOW_HALF = 50


def load_exclusive(tf):
    """Load 3-column exclusive BED (output of overlap removal)."""
    path = os.path.join(EXCLUSIVE_DIR, f"{tf}_exclusive_101bp.bed")
    df = pd.read_csv(path, sep='\t', header=None, names=['chrom', 'start', 'end'])
    print(f"  Loaded exclusive {tf}: {len(df):,} peaks from {path}")
    return df


def load_filtered(tf):
    """Load 10-column filtered narrowPeak BED."""
    path = os.path.join(FILTERED_DIR, f"{tf}_filtered.bed")
    df = pd.read_csv(path, sep='\t', header=None, names=COLS_NARROW)
    print(f"  Loaded filtered  {tf}: {len(df):,} peaks from {path}")
    return df


def join_qval(exclusive_df, filtered_df):
    """
    Match exclusive peaks back to their filtered source to retrieve qval.
    
    Key insight:
      - Exclusive peak: start = summit - 50, end = summit + 51
        => summit = start + 50
      - Filtered peak:  summit = original_start + peak (col 10)
        => summit = start + peak
    
    We join on (chrom, summit) to link them.
    """
    excl = exclusive_df.copy()
    filt = filtered_df.copy()

    # Compute summit for both
    excl['summit'] = excl['start'] + WINDOW_HALF
    filt['summit'] = filt['start'].astype(int) + filt['peak'].astype(int)

    # Merge on chrom + summit
    merged = excl.merge(
        filt[['chrom', 'summit', 'qval', 'signal']],
        on=['chrom', 'summit'],
        how='left'
    )

    # Sanity check: every exclusive peak should find a match
    n_missing = merged['qval'].isna().sum()
    if n_missing > 0:
        print(f"    WARNING: {n_missing} exclusive peaks could not be matched!")
    else:
        print(f"    All {len(merged):,} exclusive peaks matched successfully.")

    return merged


def main():
    print("=" * 70)
    print("STEP 4: DOWNSAMPLING (SP2 baseline, highest qval first)")
    print("=" * 70)
    print("Source: data/processed/filtered_exclusive_101bp/ (already completed)")
    print("Metadata from: data/processed/filtered_qval2/")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- 1. Load all exclusive + filtered datasets ----
    exclusive_dfs = {}
    filtered_dfs = {}
    for tf in TFS:
        exclusive_dfs[tf] = load_exclusive(tf)
        filtered_dfs[tf]  = load_filtered(tf)

    # ---- 2. Join qval back to exclusive peaks ----
    print("\nJoining qval metadata to exclusive peaks...")
    merged_dfs = {}
    for tf in TFS:
        print(f"  {tf}:")
        merged_dfs[tf] = join_qval(exclusive_dfs[tf], filtered_dfs[tf])

    # ---- 3. Determine baseline size (SP2) ----
    sp2_size = len(merged_dfs['SP2'])
    print(f"\nBaseline: SP2 = {sp2_size:,} peaks (kept entirely)")

    # ---- 4. Downsample SP1 and SP4 ----
    downsampled_dfs = {}
    downsampled_dfs['SP2'] = merged_dfs['SP2'].copy()

    for tf in ['SP1', 'SP4']:
        df = merged_dfs[tf].copy()
        before = len(df)

        # Sort by qval descending (highest quality first),
        # then by signal descending as tiebreaker
        df_sorted = df.sort_values(by=['qval', 'signal'], ascending=[False, False])

        # Take top sp2_size peaks
        df_down = df_sorted.head(sp2_size).copy()
        downsampled_dfs[tf] = df_down

        print(f"  {tf}: {before:,} -> {len(df_down):,} "
              f"(removed {before - len(df_down):,} lowest-quality peaks)")

    # ---- 5. Report qval distribution of kept peaks ----
    print("\n" + "-" * 70)
    print("Quality statistics of downsampled datasets:")
    print("-" * 70)
    for tf in TFS:
        df = downsampled_dfs[tf]
        q = df['qval']
        s = df['signal']
        print(f"  {tf} ({len(df):,} peaks):")
        print(f"    qval  : min={q.min():.5f}  mean={q.mean():.5f}  max={q.max():.5f}")
        print(f"    signal: min={s.min():.5f}  mean={s.mean():.5f}  max={s.max():.5f}")

    # ---- 6. Save output ----
    print("\nSaving downsampled datasets...")
    for tf in TFS:
        df = downsampled_dfs[tf]

        # 3-column BED for downstream (getfasta etc.)
        bed3_path = os.path.join(OUTPUT_DIR, f"{tf}_downsampled_101bp.bed")
        df[['chrom', 'start', 'end']].to_csv(
            bed3_path, sep='\t', header=False, index=False)

        # Full metadata file (chrom, start, end, summit, qval, signal)
        meta_path = os.path.join(OUTPUT_DIR, f"{tf}_downsampled_meta.tsv")
        df[['chrom', 'start', 'end', 'summit', 'qval', 'signal']].to_csv(
            meta_path, sep='\t', header=True, index=False)

        print(f"  {tf}: {bed3_path}  ({len(df):,} rows)")
        print(f"        {meta_path}")

    # ---- 7. Final summary ----
    print("\n" + "=" * 70)
    print("DOWNSAMPLING SUMMARY")
    print("=" * 70)
    print(f"  {'TF':<6} {'Exclusive':<12} {'Downsampled':<14} {'Removed':<10} {'Method'}")
    print(f"  {'-'*60}")
    for tf in TFS:
        excl = len(exclusive_dfs[tf])
        down = len(downsampled_dfs[tf])
        removed = excl - down
        method = "baseline (kept all)" if tf == 'SP2' else "top by qval desc"
        print(f"  {tf:<6} {excl:<12,} {down:<14,} {removed:<10,} {method}")
    print(f"  {'-'*60}")
    total = sum(len(downsampled_dfs[tf]) for tf in TFS)
    print(f"  Total balanced dataset: {total:,} peaks ({total // 3:,} per class)")
    print("=" * 70)


if __name__ == '__main__':
    main()
