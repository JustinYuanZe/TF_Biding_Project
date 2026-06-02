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
        f.write("# Báo cáo Khảo sát EDA Đặc trưng DNAshape\n\n")
        f.write("## 1. Thống kê mô tả toàn cục\n\n")
        f.write("Dưới đây là bảng thống kê mô tả được tính toán trên toàn bộ tập dữ liệu (bao gồm cả positive và negative) sau khi loại bỏ các giá trị biên `NaN`:\n\n")
        f.write(to_markdown_manual(df_stats) + "\n\n")
        f.write("## 2. Nhận xét về phân phối của từng đặc trưng:\n\n")
        for feat in feature_names:
            row = df_stats[df_stats["Feature"] == feat].iloc[0]
            f.write(f"### {feat}\n")
            f.write(f"- **Phạm vi giá trị (Min - Max):** {row['Min']:.2f} đến {row['Max']:.2f}\n")
            f.write(f"- **Median (Mean):** {row['Median']:.2f} ({row['Mean']:.2f})\n")
            f.write(f"- **P1 - P99 (Khoảng chứa 98% dữ liệu):** {row['P1']:.2f} đến {row['P99']:.2f}\n")
            f.write(f"- **Hệ số bất đối xứng (Skewness):** {row['Skewness']:.3f}\n")
            f.write(f"- **Hệ số nhọn (Kurtosis):** {row['Kurtosis']:.3f}\n")
            
            # Đưa ra nhận định ban đầu
            if abs(row['Skewness']) > 1.0:
                f.write(f"- *Nhận xét:* Đặc trưng bị lệch rất mạnh (skewed). ")
            elif abs(row['Skewness']) > 0.5:
                f.write(f"- *Nhận xét:* Đặc trưng bị lệch vừa phải. ")
            else:
                f.write(f"- *Nhận xét:* Đặc trưng khá đối xứng. ")
                
            if row['Kurtosis'] > 1.0:
                f.write(f"Phân phối có đuôi rất dày (heavy tails), có nhiều outliers tiềm năng.\n\n")
            else:
                f.write(f"Phân phối tương đối bình thường hoặc đuôi mỏng.\n\n")
                
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
        f.write("## 3. Hệ số tương quan Pearson (tính tại vị trí trung tâm index 50)\n\n")
        f.write(to_markdown_manual(corr_matrix) + "\n\n")
        f.write("## 4. Đề xuất phương án chuẩn hóa (Feature Scaling):\n\n")
        f.write("Dựa trên kết quả thống kê và đồ thị phân phối ở trên, chúng ta sẽ phân tích và chọn phương pháp chuẩn hóa phù hợp nhất cho mô hình.\n")
        
    print(f"EDA report successfully saved to {report_path}")

if __name__ == "__main__":
    perform_eda()
