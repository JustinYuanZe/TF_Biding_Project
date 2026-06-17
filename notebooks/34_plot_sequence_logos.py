import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import logomaker
except ImportError:
    print("logomaker is not installed. Please install it using: pip install logomaker")
    sys.exit(1)

def load_fasta(filepath):
    sequences = []
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return sequences
    with open(filepath, "r") as f:
        seq_lines = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_lines:
                    sequences.append("".join(seq_lines).upper())
                    seq_lines = []
            else:
                seq_lines.append(line)
        if seq_lines:
            sequences.append("".join(seq_lines).upper())
    return sequences

def get_center_window(sequences, window_size=20):
    """Lấy cửa sổ trung tâm của mỗi sequence để vẽ Logo (giả định motif nằm ở giữa)."""
    center_seqs = []
    for seq in sequences:
        L = len(seq)
        if L < window_size:
            continue
        start = (L - window_size) // 2
        center_seqs.append(seq[start:start+window_size])
    return center_seqs

def plot_logo(sequences, title, out_path):
    if not sequences:
        print(f"No sequences for {title}")
        return
        
    # Tạo ma trận count (số lần xuất hiện của A, C, G, T tại mỗi vị trí)
    L = len(sequences[0])
    counts = { 'A': np.zeros(L), 'C': np.zeros(L), 'G': np.zeros(L), 'T': np.zeros(L) }
    
    for seq in sequences:
        for i, nt in enumerate(seq):
            if nt in counts:
                counts[nt][i] += 1
                
    counts_df = pd.DataFrame(counts)
    
    # Chuyển đổi thành Information Content matrix
    # logomaker can do this automatically:
    info_mat = logomaker.alignment_to_matrix(sequences, to_type='information')
    
    fig, ax = plt.subplots(figsize=(10, 3))
    logo = logomaker.Logo(info_mat, ax=ax, color_scheme='classic', vpad=.1, width=.8)
    
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    ax.set_ylabel("Information (bits)", labelpad=-1)
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

def main():
    # Sử dụng đường dẫn có thể thay đổi tùy thuộc vào môi trường (Local / Kaggle)
    # Ở đây default local path, user có thể sửa
    data_dir = r"c:\Users\MSI\Desktop\Temporary\SP1_TF_Binding_Project\data\processed"
    if not os.path.exists(os.path.join(data_dir, "sp1_positive_final.fasta")):
        # Fallback: search common locations (local data/processed, Kaggle input)
        for cand in [r"c:\Users\MSI\Desktop\Temporary\SP1_TF_Binding_Project\data\processed",
                     "data/processed", "/kaggle/input/datasets/lehotrongtin/datas1", "/kaggle/input"]:
            if os.path.isdir(cand):
                hit = None
                for root, _, files in os.walk(cand):
                    if "sp1_positive_final.fasta" in files:
                        hit = root; break
                if hit:
                    data_dir = hit; break
    print(f"[data_dir] {data_dir}")
        
    out_dir = r"c:\Users\MSI\Desktop\Temporary\SP1_TF_Binding_Project\outputs_figures"
    os.makedirs(out_dir, exist_ok=True)
    
    tfs = ["SP1", "SP2", "SP4"]
    for tf in tfs:
        fasta_file = os.path.join(data_dir, f"{tf.lower()}_positive_final.fasta")
        seqs = load_fasta(fasta_file)
        if seqs:
            center_seqs = get_center_window(seqs, window_size=15)
            out_file = os.path.join(out_dir, f"{tf}_sequence_logo.png")
            plot_logo(center_seqs, f"{tf} Central Motif Alignment", out_file)
            
if __name__ == "__main__":
    main()
