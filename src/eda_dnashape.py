import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_skew(x):
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0
    return np.mean((x - mean) ** 3) / (std ** 3)

def calculate_kurtosis(x):
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0
    return np.mean((x - mean) ** 4) / (std ** 4) - 3

def to_markdown_manual(df):
    cols = df.columns
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for idx, r in df.iterrows():
        row_vals = []
        for c in cols:
            val = r[c]
            if isinstance(val, float):
                row_vals.append(f"{val:.6f}")
            else:
                row_vals.append(str(val))
        row = "| " + " | ".join(row_vals) + " |"
        rows.append(row)
    return "\n".join([header, separator] + rows)

def perform_eda():
    data_dir = r"data\processed"
    fig_dir = r"figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    classes = ["sp1", "sp2", "sp4", "negative"]
    feature_names = ["MGW", "ProT", "Roll", "HelT", "EP"]
    
    # Load data
    data = {}
    for c in classes:
        npy_path = os.path.join(data_dir, f"dnashape_{c}.npy")
        if not os.path.exists(npy_path):
            print(f"File {npy_path} not found. Please extract features first.")
            return
        data[c] = np.load(npy_path)
        print(f"Loaded {c} shape dataset: {data[c].shape}")
        
    # Aggregate values per feature across all sequences to compute overall statistics
    all_flat_features = {feat: [] for feat in feature_names}
    class_flat_features = {c: {feat: [] for feat in feature_names} for c in classes}
    
    for c in classes:
        arr = data[c] # Shape: [N, 5, 101]
        for f_idx, feat in enumerate(feature_names):
            vals = arr[:, f_idx, :].flatten()
            valid_vals = vals[~np.isnan(vals)]
            class_flat_features[c][feat] = valid_vals
            all_flat_features[feat].extend(valid_vals)
            
    # Calculate descriptive stats
    stats_rows = []
    for feat in feature_names:
        vals = np.array(all_flat_features[feat])
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        median_val = np.median(vals)
        min_val = np.min(vals)
        max_val = np.max(vals)
        p1 = np.percentile(vals, 1)
        p5 = np.percentile(vals, 5)
        p95 = np.percentile(vals, 95)
        p99 = np.percentile(vals, 99)
        skew_val = calculate_skew(vals)
        kurt_val = calculate_kurtosis(vals)
        
        stats_rows.append({
            "Feature": feat,
            "Mean": mean_val,
            "SD": std_val,
            "Median": median_val,
            "Min": min_val,
            "Max": max_val,
            "P1": p1,
            "P5": p5,
            "P95": p95,
            "P99": p99,
            "Skewness": skew_val,
            "Kurtosis": kurt_val
        })
        
    df_stats = pd.DataFrame(stats_rows)
    print("\n--- GLOBAL DESCRIPTIVE STATISTICS ---")
    print(df_stats.to_string(index=False))
    
    # Save statistics report
    report_path = r"scratch\dnashape_eda_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# DNAshape Features EDA Report\n\n")
        f.write("## 1. Global Descriptive Statistics\n\n")
        f.write("Below is the descriptive statistics table computed on the entire dataset (including both positive and negative) after removing boundary `NaN` values:\n\n")
        f.write(to_markdown_manual(df_stats) + "\n\n")
        f.write("## 2. Distribution Comments for Each Feature:\n\n")
        for feat in feature_names:
            row = df_stats[df_stats["Feature"] == feat].iloc[0]
            f.write(f"### {feat}\n")
            f.write(f"- **Value Range (Min - Max):** {row['Min']:.2f} to {row['Max']:.2f}\n")
            f.write(f"- **Median (Mean):** {row['Median']:.2f} ({row['Mean']:.2f})\n")
            f.write(f"- **P1 - P99 (Interval containing 98% of data):** {row['P1']:.2f} to {row['P99']:.2f}\n")
            f.write(f"- **Skewness:** {row['Skewness']:.3f}\n")
            f.write(f"- **Kurtosis:** {row['Kurtosis']:.3f}\n")
            
            # Provide initial assessment
            if abs(row['Skewness']) > 1.0:
                f.write(f"- *Comment:* Feature is highly skewed. ")
            elif abs(row['Skewness']) > 0.5:
                f.write(f"- *Comment:* Feature is moderately skewed. ")
            else:
                f.write(f"- *Comment:* Feature is relatively symmetric. ")
                
            if row['Kurtosis'] > 1.0:
                f.write(f"Distribution has heavy tails with potential outliers.\n\n")
            else:
                f.write(f"Distribution is relatively normal or thin-tailed.\n\n")
                
    # Plot distributions (using histograms as density plot alternatives)
    plt.figure(figsize=(18, 12))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for idx, feat in enumerate(feature_names):
        plt.subplot(2, 3, idx+1)
        for c_idx, c in enumerate(classes):
            vals = class_flat_features[c][feat]
            plt.hist(vals, bins=50, density=True, alpha=0.2, color=colors[c_idx])
            counts, bin_edges = np.histogram(vals, bins=50, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            plt.plot(bin_centers, counts, label=c, color=colors[c_idx], linewidth=1.5)
            
        plt.title(f"Distribution of {feat}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        
    plt.tight_layout()
    dist_fig_path = os.path.join(fig_dir, "dnashape_distributions.png")
    plt.savefig(dist_fig_path, dpi=150)
    plt.close()
    print(f"Saved distribution plot to {dist_fig_path}")
    
    # Plot boxplots to see outliers
    plt.figure(figsize=(18, 10))
    for idx, feat in enumerate(feature_names):
        plt.subplot(2, 3, idx+1)
        boxplot_data = [class_flat_features[c][feat] for c in classes]
        plt.boxplot(boxplot_data, labels=classes, patch_artist=True,
                    boxprops=dict(facecolor='#dbeef6', color='#1f77b4'),
                    medianprops=dict(color='#d62728', linewidth=1.5))
        plt.title(f"Boxplot of {feat}")
        plt.ylabel("Value")
        plt.grid(True, linestyle="--", alpha=0.5)
        
    plt.tight_layout()
    box_fig_path = os.path.join(fig_dir, "dnashape_boxplots.png")
    plt.savefig(box_fig_path, dpi=150)
    plt.close()
    print(f"Saved boxplot visualization to {box_fig_path}")

    # Plot correlation between features (using manual imshow)
    aligned_data = []
    for c in classes:
        arr = data[c]
        mid_vals = arr[:, :, 50] # Shape: [N, 5]
        aligned_data.append(mid_vals)
    aligned_data = np.vstack(aligned_data) # Shape: [Total_N, 5]
    
    df_corr = pd.DataFrame(aligned_data, columns=feature_names)
    corr_matrix = df_corr.corr(method='pearson')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
    
    fig.colorbar(im, ax=ax)
    
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names)
    ax.set_yticklabels(feature_names)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            val = corr_matrix.iloc[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", 
                    color="white" if abs(val) > 0.5 else "black")
            
    ax.set_title("Correlation Matrix of DNAshape Features (at middle position 50)")
    plt.tight_layout()
    corr_fig_path = os.path.join(fig_dir, "dnashape_correlation.png")
    plt.savefig(corr_fig_path, dpi=150)
    plt.close()
    print(f"Saved correlation plot to {corr_fig_path}")
    
    # Append correlation info to report
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("## 3. Pearson Correlation Coefficients (calculated at center index 50)\n\n")
        f.write(to_markdown_manual(corr_matrix) + "\n\n")
        f.write("## 4. Proposed Normalization Method (Feature Scaling):\n\n")
        f.write("Based on the statistics and distribution plots above, we will analyze and select the most appropriate normalization method for the model.\n")
        
    print(f"EDA report successfully saved to {report_path}")

if __name__ == "__main__":
    perform_eda()
