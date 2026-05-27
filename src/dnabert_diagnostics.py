import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from src.dnabert_wrapper import DNABERTWrapper

def run_dnabert_diagnostics(sequences_dict, device='cuda', save_dir='figures'):
    """
    Run diagnostic checks on DNABERT-2 embeddings:
    1. Extract CLS embeddings for a subset of each class.
    2. Compute PCA and t-SNE dimensional reductions.
    3. Generate diagnostic plots showing the distribution of representations.
    4. Compute inter-class cosine similarity matrix.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n=== Initializing DNABERT-2 Diagnostics ===")
    wrapper = DNABERTWrapper(device=device)
    
    # 1. Prepare subset for visualization (e.g., 500 samples per class to keep t-SNE fast)
    max_samples_per_class = 500
    diagnostic_seqs = []
    diagnostic_labels = []
    class_names = list(sequences_dict.keys())
    
    for label_idx, (class_name, seqs) in enumerate(sequences_dict.items()):
        samples = seqs[:max_samples_per_class]
        diagnostic_seqs.extend(samples)
        diagnostic_labels.extend([label_idx] * len(samples))
        print(f"  Loaded {len(samples)} samples from class '{class_name}'")
        
    diagnostic_labels = np.array(diagnostic_labels)
    
    # 2. Extract CLS embeddings (shape: n_samples, 768)
    print("\nExtracting CLS embeddings for diagnostics...")
    cls_embeddings = wrapper.get_cls_embeddings(diagnostic_seqs, batch_size=64)
    print(f"Extracted CLS embeddings shape: {cls_embeddings.shape}")
    
    # 3. Compute PCA and t-SNE
    print("Computing PCA projection...")
    pca = PCA(n_components=2, random_state=42)
    pca_results = pca.fit_transform(cls_embeddings)
    
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(cls_embeddings)
    
    # 4. Plot PCA and t-SNE side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # PCA Plot
    for i, class_name in enumerate(class_names):
        indices = np.where(diagnostic_labels == i)
        ax1.scatter(
            pca_results[indices, 0], 
            pca_results[indices, 1], 
            label=class_name, 
            color=colors[i], 
            alpha=0.6, 
            edgecolors='none'
        )
    ax1.set_title('PCA of DNABERT-2 CLS Embeddings', fontsize=14, pad=10)
    ax1.set_xlabel('Principal Component 1', fontsize=12)
    ax1.set_ylabel('Principal Component 2', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=11)
    
    # t-SNE Plot
    for i, class_name in enumerate(class_names):
        indices = np.where(diagnostic_labels == i)
        ax2.scatter(
            tsne_results[indices, 0], 
            tsne_results[indices, 1], 
            label=class_name, 
            color=colors[i], 
            alpha=0.6, 
            edgecolors='none'
        )
    ax2.set_title('t-SNE of DNABERT-2 CLS Embeddings', fontsize=14, pad=10)
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=11)
    
    plt.suptitle('DNABERT-2 Representation Space Diagnostic Analysis', fontsize=16, weight='bold', y=0.98)
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, "dnabert_embedding_diagnostics.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Diagnostic visualization plots saved to: {plot_path}")
    
    # 5. Compute Cosine Similarities between classes
    print("\n--- Cosine Similarity Diagnostics ---")
    mean_embeddings = []
    for i, class_name in enumerate(class_names):
        indices = np.where(diagnostic_labels == i)[0]
        class_mean = cls_embeddings[indices].mean(axis=0)
        mean_embeddings.append(class_mean)
        
    mean_embeddings = np.array(mean_embeddings)
    sim_matrix = cosine_similarity(mean_embeddings)
    
    print("\nMean Embedding Cosine Similarity Matrix:")
    header = "          " + "".join([f"{name:>12}" for name in class_names])
    print(header)
    for i, name in enumerate(class_names):
        row_str = f"{name:<10}" + "".join([f"{sim_matrix[i, j]:12.4f}" for j in range(len(class_names))])
        print(row_str)
        
    # Check if the embeddings are collapsed (i.e. all extremely similar)
    all_sims = cosine_similarity(cls_embeddings)
    # Exclude identity diagonals
    np.fill_diagonal(all_sims, np.nan)
    avg_pairwise_sim = np.nanmean(all_sims)
    std_pairwise_sim = np.nanstd(all_sims)
    print(f"\nAverage overall pairwise cosine similarity: {avg_pairwise_sim:.4f} (std: {std_pairwise_sim:.4f})")
    if avg_pairwise_sim > 0.98:
        print("⚠️  Warning: Extremely high similarity (> 0.98) indicates representation collapse.")
    else:
        print("✅ Representation diversity check passed.")
        
    return sim_matrix
