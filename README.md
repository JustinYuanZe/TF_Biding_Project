# SP Family (SP1/SP2/SP4) Transcription Factor Binding Site Prediction

This repository contains a state-of-the-art computational biology pipeline to predict genomic binding sites for the **SP transcription factor family (SP1, SP2, and SP4)** using multimodal deep learning. 

Our proposed **Tri-Branch Architecture** fuses genomic sequence embeddings (from the pre-trained transformer **DNABERT-2**) with physical 3D structural parameters (**DNAshape**) via bidirectional **Cross-Modal Attention**, augmented by local epigenetic/biochemical features (**Bio-features**: CpG islands, GC content, and G-quadruplex motifs).

---

## 🌟 Key Features
*   **Multimodal Deep Fusion**: Combines sequence semantics, spatial structural shapes, and biological markers.
*   **Bidirectional Cross-Modal Attention**: Employs Query-Key-Value interactions to align 1D sequence representations directly with 3D DNA physical geometry.
*   **Ablation Benchmark Hierarchy**: Tracks performance gains from Classical ML baselines $\rightarrow$ Sequence-only Deep Learning $\rightarrow$ Cross-Modal Multimodal Fusion $\rightarrow$ Tri-Branch Proposed Model.
*   **Explainable AI (XAI)**: Demystifies the model's predictions using **SHAP (SHapley Additive exPlanations)**, revealing exactly how sequence motifs, DNA physical shapes, and CpG density contribute to binding decisions.
*   **DDP-Optimized Distributed Training**: Built using **PyTorch Accelerate** for multi-GPU Distributed Data Parallel (DDP) execution (tested on 2x Tesla T4 GPUs).

---

## 📊 Benchmarking Results

Our experiments demonstrate a massive performance leap when combining DNA sequence with physical structure and biochemical features:

| Model | Accuracy | F1-Score | Precision | Recall (Positive) | Recall (Negative) | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical ML (SVM on k-mer TF-IDF)** | 78.50% | 85.30% | 81.20% | 89.80% | 44.50% | 0.8120 | 0.8950 |
| **Sequence-Only (Fine-Tuned DNABERT-2)** | 89.05% | 92.80% | 91.50% | 94.10% | 73.90% | 0.9412 | 0.9780 |
| **Cross-Modal Model (Seq + DNAshape)** | **96.01%** | **97.35%** | **96.90%** | **97.80%** | **90.65%** | **0.9892** | **0.9961** |
| **Proposed Tri-Branch Model (Seq + Shape + Bio)**| **96.42%** | **97.60%** | **97.10%** | **98.10%** | **91.40%** | **0.9904** | **0.9968** |

### Key Takeaways:
1.  **The Shape Boost**: Fusing physical DNA shape parameters via cross-modal attention triggers a **~7% jump in accuracy** and increases the model's ability to filter out false positives (**Negative Recall increases from 73.90% to 90.65%**).
2.  **Biological Saturation Limit**: The proposed Tri-Branch model approaches the physical information limit of the 101 bp sequence window (~96.42%), with CpG islands providing biological promoter markers to further stabilize prediction and reduce false positives.

---

## 🛠️ Data Preprocessing & Feature Engineering

The pipeline processes raw peak files from the ENCODE project (HepG2 cell line, ChIP-seq, hg38 reference genome) through a rigorous preparation workflow:

```
[Raw ENCODE Peaks] 
       │ (Filter q-value ≤ 0.01, center 101bp on summit, remove overlaps)
       ▼
[Augmentation] ──► Positive: Reverse Complement (RC) doubling (1,696 -> 3,392 per class)
       │
       ▼
[Negative Gen] ──► Shuffled Negatives: Dinucleotide Shuffling (Preserves CG/GC density)
       │
       ▼
[Final Dataset] ──► Balanced 1:1:1:1 split (SP1/2/4 Positives: 10,176 | Negatives: 3,392)
```

### Multimodal Feature Streams:
1.  **DNA Sequence**: Tokenized using BPE (Byte Pair Encoding) vocabulary of **DNABERT-2** (zhihan1996/DNABERT-2-117M).
2.  **DNAshape (5 structural features)**: Computes 3D structural parameters for each 101 bp sequence using a sliding pentamer (5-mer) lookup:
    *   **Minor Groove Width (MGW)**
    *   **Propeller Twist (ProT)**
    *   **Roll**
    *   **Helix Twist (HelT)**
    *   **Electrostatic Potential (EP)**
3.  **Bio-features**: Dynamically extracts biochemical signatures:
    *   **CpG Observed/Expected (O/E) ratio**: Promoters/binding marker.
    *   **GC Content**: Stability marker.
    *   **G-quadruplex (G4) Motifs**: Non-B DNA secondary structures found in promoter regions (detected via regex).

---

## 🏗️ Architecture Design

```
                     ┌──────────────────┐
                     │ 101 bp DNA Input │
                     └────────┬─────────┘
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ DNA Sequence  │     │   DNA Shape   │     │ Bio-Features  │
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                     │
        ▼ (BERT-2)            ▼ (CNN 1D)            ▼ (MLP)
┌───────────────┐     ┌───────────────┐             │
│  768-dim Embs │     │  128-dim Feats│             │
└───────┬───────┘     └───────┬───────┘             │
        │                     │                     │
        └─────────┬───────────┘                     │
                  ▼                                 │
     ┌────────────────────────┐                     │
     │ Bidirectional Attention│                     │
     └────────────┬───────────┘                     │
                  │                                 │
                  ▼ (Pooling)                       │
           ┌──────────────┐                         │
           │ Concatenation│◄────────────────────────┘
           └──────┬───────┘
                  ▼
          ┌───────────────┐
          │  Sigmoid Head │ ──► Binding Probability (0 to 1)
          └───────────────┘
```

---

## 📂 Project Structure

```
├── data/
│   └── processed/                    # Final balanced datasets (gitignored)
├── figures/
│   ├── classic/                      # Classical ML metrics, ROC/PR curves
│   ├── BERT_only/                    # Sequence-only (DNABERT-2) curves
│   ├── cross_modal/                  # Cross-Modal Dual-Branch curves
│   ├── Proposed Model/               # Proposed Tri-Branch model curves
│   └── SHAP/                         # Interpretability summary & force plots
├── notebooks/                        # Kaggle-executable wrapper notebooks
│   ├── 23_binary_tribranch_kaggle.py # Proposed Tri-Branch Training script
│   ├── 24_binary_dnabert2_finetune_kaggle.py  # Sequence-only BERT Training script
│   ├── 25_binary_crossmodal_dual_kaggle.py    # Cross-Modal Model Training script
│   └── 27_binary_classical_ml_baselines.py    # Classical ML Benchmarking script
├── src/                              # Core feature extraction & preprocessing
│   ├── filter_and_eda.py             # Peak filtering & QC
│   ├── extract_fasta.py              # Sequence extraction from hg38.fa
│   ├── prepare_final_dataset.py      # Augmentation & Shuffling pipeline
│   └── extract_dnashape.py           # DNAshape pentamer extraction
└── presentation_structure.md         # 17-slide Thesis Defense Outline
```

---

## 🚀 How to Run

### 1. Installation & Environment
Ensure PyTorch, Transformers, and Accelerate are installed. Install dependencies:
```bash
pip install transformers accelerate safetensors scikit-learn matplotlib numpy regex einops shap
```

### 2. Preprocessing & Data Generation
Extract FASTA sequences from the reference genome and run augmentation/shuffling:
```bash
# 1. Filter peaks and extract sequences
python src/filter_and_eda.py
python src/extract_fasta.py --fasta /path/to/hg38.fa

# 2. Build balanced dataset (SP positives + dinuc shuffled negatives)
python src/prepare_final_dataset.py

# 3. Extract DNAshape structural features (numpy arrays)
python src/extract_dnashape.py
```

### 3. Run Baselines
```bash
# Classical ML Baselines (SVM, RF, LogReg, MLP, etc.)
python notebooks/27_binary_classical_ml_baselines.py

# Fine-tune Sequence-Only DNABERT-2
accelerate launch notebooks/24_binary_dnabert2_finetune_kaggle.py
```

### 4. Run Cross-Modal & Proposed Models
To train the cross-modal attention and Tri-Branch models in a distributed multi-GPU setup:
```bash
# Cross-Modal model
accelerate launch notebooks/25_binary_crossmodal_dual_kaggle.py

# Proposed Tri-Branch model
accelerate launch notebooks/23_binary_tribranch_kaggle.py
```

### 5. Model Interpretability (SHAP Analysis)
To generate SHAP plots and analyze model feature importance:
```bash
python notebooks/explain_shap_tribranch.py --model_path outputs_binary_tribranch/models/best_model.pt
```

---

## 🎓 References & Acknowledgements
*   **DNABERT-2**: Zhou, J., et al. "DNABERT-2: Efficient and Effective Foundation Model for DNA Language."
*   **DNAshapeR**: Chiu, T. P., et al. "DNAshapeR: an R/Bioconductor package for genome-wide DNA shape prediction."
*   **ENCODE Project Consortium**: HepG2 ChIP-seq dataset sources (ENCFF333SWC, ENCFF480YAW, ENCFF938KVY).
