"""
Filtering and Exploratory Data Analysis of TF peak data.
Applies quality filters, generates KDE plots, centers windows around
peak summits, and enforces mutual exclusivity among different TFs.
"""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# Configuration
RAW_PATHS = {
    'SP1': os.path.join('data', 'raw', 'SP1.bed', 'ENCFF333SWC.bed'),
    'SP2': os.path.join('data', 'raw', 'SP2.bed', 'ENCFF480YAW.bed'),
    'SP4': os.path.join('data', 'raw', 'SP4.bed', 'ENCFF938KVY.bed')
}

OUTPUT_DIR = os.path.join('data', 'processed', 'filtered_qval2')
OUTPUT_CENTERED_DIR = os.path.join('data', 'processed', 'filtered_centered_101bp')
OUTPUT_EXCLUSIVE_DIR = os.path.join('data', 'processed', 'filtered_exclusive_101bp')
FIGURES_DIR = 'figures'
WINDOW_HALF = 50  # 50bp left + summit + 50bp right = 101bp

# Column names for narrowPeak/ENCODE bed format
COLS = ['chrom', 'start', 'end', 'name', 'score', 'strand', 'signal', 'pval', 'qval', 'peak']

def load_data(name: str, path: str) -> pd.DataFrame:
    print(f"Loading {name} raw data from {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw dataset file not found: {path}")
    df = pd.read_csv(path, sep='\t', header=None, names=COLS)
    return df

def apply_filter(df: pd.DataFrame) -> pd.DataFrame:
    # Column 9 is 'qval' which is -log10(q-value).
    # q-value <= 0.01 corresponds to -log10(q-value) >= 2.0
    return df[df['qval'] >= 2.0]

def plot_kde(ax: plt.Axes, data_before: pd.DataFrame, data_after: pd.DataFrame, column_name: str, title: str, xlabel: str, log_scale: bool = False) -> None:
    # Drop NaNs or infinite values if any
    db = data_before[column_name].dropna()
    da = data_after[column_name].dropna()
    
    if log_scale:
        db = np.log10(db + 1e-5)
        da = np.log10(da + 1e-5)
        xlabel = f"log10({xlabel})"

    # Histograms
    ax.hist(db, bins=40, density=True, alpha=0.3, color='#1f77b4', label='Before Filter')
    ax.hist(da, bins=40, density=True, alpha=0.4, color='#ff7f0e', label='After Filter (qval >= 2)')

    # Smooth KDE Curves
    try:
        if len(db) > 1:
            kde_b = gaussian_kde(db)
            x_b = np.linspace(db.min(), db.max(), 200)
            ax.plot(x_b, kde_b(x_b), color='#1f77b4', lw=2)
        if len(da) > 1:
            kde_a = gaussian_kde(da)
            x_a = np.linspace(da.min(), da.max(), 200)
            ax.plot(x_a, kde_a(x_a), color='#ff7f0e', lw=2)
    except Exception as e:
        print(f"KDE fitting failed for {title}: {e}")

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    stats = []

    # Store loaded dataframes for plotting
    datasets_raw = {}
    datasets_filtered = {}

    for name, path in RAW_PATHS.items():
        df_raw = load_data(name, path)
        df_filtered = apply_filter(df_raw)
        
        # Save filtered dataset (only original 10 columns, no extra derived columns)
        output_path = os.path.join(OUTPUT_DIR, f"{name}_filtered.bed")
        df_filtered[COLS].to_csv(output_path, sep='\t', header=False, index=False)
        print(f"Saved filtered data to {output_path} ({len(df_filtered)} / {len(df_raw)} peaks kept)")

        datasets_raw[name] = df_raw
        datasets_filtered[name] = df_filtered

        # Calculate statistics (compute length on-the-fly for EDA only, not saved to file)
        raw_length = df_raw['end'] - df_raw['start']
        filt_length = df_filtered['end'] - df_filtered['start']
        pct_kept = (len(df_filtered) / len(df_raw)) * 100
        stats.append({
            'Dataset': name,
            'Raw Count': len(df_raw),
            'Filtered Count': len(df_filtered),
            'Keep %': f"{pct_kept:.2f}%",
            'Mean Length (Raw)': f"{raw_length.mean():.1f} bp",
            'Mean Length (Filtered)': f"{filt_length.mean():.1f} bp",
            'Mean Signal (Raw)': f"{df_raw['signal'].mean():.2f}",
            'Mean Signal (Filtered)': f"{df_filtered['signal'].mean():.2f}",
            'Mean -log10(qval) (Raw)': f"{df_raw['qval'].mean():.3f}",
            'Mean -log10(qval) (Filtered)': f"{df_filtered['qval'].mean():.3f}"
        })

    # Convert stats to DataFrame and print as markdown table
    df_stats = pd.DataFrame(stats)
    
    # Custom markdown formatter to avoid tabulate dependency
    def df_to_md(df: pd.DataFrame) -> str:
        headers = list(df.columns)
        md = "| " + " | ".join(headers) + " |\n"
        md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for _, row in df.iterrows():
            md += "| " + " | ".join(str(val) for val in row) + " |\n"
        return md

    md_table = df_to_md(df_stats)
    print("\n=== DATASET PRE- AND POST-FILTERING STATISTICS ===")
    print(md_table)

    # Save stats to markdown file for user documentation
    with open(os.path.join(FIGURES_DIR, 'filtering_stats.md'), 'w', encoding='utf-8') as f:
        f.write(md_table)

    # Styling settings for nice plots
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['axes.linewidth'] = 0.8

    # PLOT 1: Peak Counts Before and After Filtering
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    names = list(RAW_PATHS.keys())
    raw_counts = [len(datasets_raw[n]) for n in names]
    filt_counts = [len(datasets_filtered[n]) for n in names]
    
    x = np.arange(len(names))
    width = 0.35

    rects1 = ax1.bar(x - width/2, raw_counts, width, label='Raw Peaks', color='#4A90E2', edgecolor='#2c5f9e', alpha=0.8)
    rects2 = ax1.bar(x + width/2, filt_counts, width, label='Filtered Peaks (qval >= 2)', color='#F5A623', edgecolor='#b9770e', alpha=0.8)

    ax1.set_ylabel('Number of Peaks', fontsize=12, fontweight='bold')
    ax1.set_title('Transcription Factor Peak Counts Before & After Filtering (qvalue <= 0.01)', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=11, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)

    # Attach a text label above each bar in rects1 and rects2, displaying its height.
    def autolabel(rects: Any) -> None:
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'{height:,}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='semibold')

    autolabel(rects1)
    autolabel(rects2)
    fig1.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'peak_counts_comparison.png'), dpi=150)
    plt.close()

    # PLOT 2: -log10(q-value) Distribution Comparison
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    for i, name in enumerate(names):
        plot_kde(axes2[i], datasets_raw[name], datasets_filtered[name], 'qval', 
                 title=f"{name}: -log10(qvalue) Distribution", xlabel="-log10(q-value)")
    fig2.suptitle('Distribution of -log10(q-value) (Before vs After Filter)', fontsize=15, fontweight='bold', y=0.98)
    fig2.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'qvalue_distribution_comparison.png'), dpi=150)
    plt.close()

    # PLOT 3: Signal Value Distribution Comparison (log-scaled representation because signal values span multiple orders)
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    for i, name in enumerate(names):
        plot_kde(axes3[i], datasets_raw[name], datasets_filtered[name], 'signal', 
                 title=f"{name}: signalValue Distribution", xlabel="signalValue", log_scale=True)
    fig3.suptitle('Distribution of signalValue (Log10 scale, Before vs After Filter)', fontsize=15, fontweight='bold', y=0.98)
    fig3.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'signal_distribution_comparison.png'), dpi=150)
    plt.close()

    # PLOT 4: Peak Length Distribution Comparison
    # Add temporary 'length' column for plotting only
    for n in names:
        datasets_raw[n]['length'] = datasets_raw[n]['end'] - datasets_raw[n]['start']
        datasets_filtered[n]['length'] = datasets_filtered[n]['end'] - datasets_filtered[n]['start']

    fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5))
    for i, name in enumerate(names):
        plot_kde(axes4[i], datasets_raw[name], datasets_filtered[name], 'length', 
                 title=f"{name}: Peak Length Distribution", xlabel="Peak Length (bp)")
    fig4.suptitle('Distribution of Peak Lengths in bp (Before vs After Filter)', fontsize=15, fontweight='bold', y=0.98)
    fig4.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'peak_length_distribution_comparison.png'), dpi=150)
    plt.close()

    # Drop temporary column
    for n in names:
        datasets_raw[n].drop(columns=['length'], inplace=True)
        datasets_filtered[n].drop(columns=['length'], inplace=True)

    print("\nEDA figures generated and saved successfully to figures/ directory!")

    # =========================================================================
    # STEP 2: SPATIAL NORMALIZATION — Center 101bp window on peak summit
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Spatial Normalization (101bp centered on peak summit)")
    print("=" * 60)
    print(f"  Window: summit - {WINDOW_HALF} bp  to  summit + {WINDOW_HALF} bp  =  {2*WINDOW_HALF+1} bp")
    print(f"  Formula: summit = start + peak (column 10)")

    os.makedirs(OUTPUT_CENTERED_DIR, exist_ok=True)
    centered_dfs = {}  # Store centered DataFrames for overlap removal in STEP 3

    for name in names:
        df = datasets_filtered[name].copy()
        initial_count = len(df)

        # Ensure integer types for coordinate arithmetic
        df[['start', 'end', 'peak']] = df[['start', 'end', 'peak']].astype(int)

        # Calculate absolute summit position: start + peak_offset
        df['summit'] = df['start'] + df['peak']

        # Create centered 101bp window around summit
        df['new_start'] = df['summit'] - WINDOW_HALF       # summit - 50
        df['new_end']   = df['summit'] + WINDOW_HALF + 1   # summit + 51  (BED is 0-based, half-open)

        # Filter out peaks where the window exceeds chromosome boundaries (new_start < 0)
        df = df[df['new_start'] >= 0]
        final_count = len(df)
        removed = initial_count - final_count

        # Keep only the 3 BED columns for downstream use
        centered = df[['chrom', 'new_start', 'new_end']].copy()
        centered.columns = ['chrom', 'start', 'end']
        centered_dfs[name] = centered

        # Save centered 101bp BED (3 columns: chrom, start, end)
        output_path = os.path.join(OUTPUT_CENTERED_DIR, f"{name}_filtered_centered_101bp.bed")
        centered.to_csv(output_path, sep='\t', header=False, index=False)

        print(f"\n  {name}:")
        print(f"    Filtered peaks:         {initial_count}")
        print(f"    Boundary violations:    {removed} removed (new_start < 0)")
        print(f"    Final centered peaks:   {final_count}")
        print(f"    Saved to: {output_path}")

    print("\n" + "=" * 60)
    print("DONE - All filtered datasets have been spatially normalized to 101bp.")
    print(f"Output directory: {OUTPUT_CENTERED_DIR}")
    print("=" * 60)

    # =========================================================================
    # STEP 3: CROSS-CLASS OVERLAP REMOVAL (Baseline: >= 1bp overlap -> remove)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Cross-Class Overlap Removal (Baseline >= 1bp)")
    print("=" * 60)
    print("  Rule: If a 101bp region of TF_X overlaps >= 1bp with ANY")
    print("         101bp region of TF_Y or TF_Z -> REMOVE from TF_X.")
    print("  Goal: Ensure each TF's binding sites are 100% exclusive.")

    os.makedirs(OUTPUT_EXCLUSIVE_DIR, exist_ok=True)
    exclusive_dfs = {}

    for name in names:
        query = centered_dfs[name]
        # Build exclusion set: union of ALL other TFs
        other_names = [n for n in names if n != name]
        exclusion = pd.concat([centered_dfs[n] for n in other_names], ignore_index=True)

        print(f"\n  {name} vs {other_names}:")
        print(f"    Query regions:     {len(query):,}")
        print(f"    Exclusion regions: {len(exclusion):,} ({' + '.join(f'{n}={len(centered_dfs[n]):,}' for n in other_names)})")

        # Per-chromosome overlap detection using NumPy broadcasting
        # Overlap condition: query_start < excl_end AND excl_start < query_end
        keep_mask = np.ones(len(query), dtype=bool)
        overlap_by_chrom = {}

        for chrom in query['chrom'].unique():
            q_mask = (query['chrom'] == chrom).values
            s_mask = (exclusion['chrom'] == chrom).values

            if not s_mask.any():
                continue

            q_starts = query.loc[q_mask, 'start'].values
            q_ends   = query.loc[q_mask, 'end'].values
            s_starts = exclusion.loc[s_mask, 'start'].values
            s_ends   = exclusion.loc[s_mask, 'end'].values

            # Broadcasting: (n_query, 1) vs (1, n_subject) -> (n_query, n_subject)
            has_overlap = np.any(
                (q_starts[:, None] < s_ends[None, :]) &
                (s_starts[None, :] < q_ends[:, None]),
                axis=1
            )

            n_overlaps = has_overlap.sum()
            if n_overlaps > 0:
                overlap_by_chrom[chrom] = n_overlaps

            # Map back to original indices
            q_indices = np.where(q_mask)[0]
            keep_mask[q_indices[has_overlap]] = False

        removed = (~keep_mask).sum()
        kept = keep_mask.sum()
        pct_kept = (kept / len(query)) * 100

        exclusive_dfs[name] = query[keep_mask].copy()

        # Save exclusive BED
        output_path = os.path.join(OUTPUT_EXCLUSIVE_DIR, f"{name}_exclusive_101bp.bed")
        exclusive_dfs[name].to_csv(output_path, sep='\t', header=False, index=False)

        print(f"    Overlapping peaks:  {removed:,} removed")
        print(f"    Exclusive peaks:    {kept:,} kept ({pct_kept:.2f}%)")
        print(f"    Saved to: {output_path}")

        # Show top chromosomes with most overlaps
        if overlap_by_chrom:
            sorted_chroms = sorted(overlap_by_chrom.items(), key=lambda x: x[1], reverse=True)[:5]
            top_str = ', '.join(f"{c}={v}" for c, v in sorted_chroms)
            print(f"    Top overlap chroms: {top_str}")

    # Final summary table
    print("\n" + "=" * 60)
    print("OVERLAP REMOVAL SUMMARY")
    print("=" * 60)
    print(f"  {'Dataset':<10} {'Before':>10} {'After':>10} {'Removed':>10} {'Keep %':>10}")
    print(f"  {'-'*50}")
    for name in names:
        before = len(centered_dfs[name])
        after  = len(exclusive_dfs[name])
        removed = before - after
        pct = (after / before) * 100
        print(f"  {name:<10} {before:>10,} {after:>10,} {removed:>10,} {pct:>9.2f}%")
    print(f"  {'-'*50}")
    total_before = sum(len(centered_dfs[n]) for n in names)
    total_after  = sum(len(exclusive_dfs[n]) for n in names)
    total_removed = total_before - total_after
    total_pct = (total_after / total_before) * 100
    print(f"  {'TOTAL':<10} {total_before:>10,} {total_after:>10,} {total_removed:>10,} {total_pct:>9.2f}%")
    print("\n" + "=" * 60)
    print("DONE - All datasets are now cross-class exclusive (pure).")
    print(f"Output directory: {OUTPUT_EXCLUSIVE_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    main()
