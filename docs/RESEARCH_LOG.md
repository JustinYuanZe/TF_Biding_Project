\# 🧬 RESEARCH LOG: Multi-modal SP-Family Binding Prediction

---

## 🗓️ 2026-05-08 | Data Exploration

> [!IMPORTANT]
> **Objective:** Establish a clean 3-class dataset (SP1/SP2/SP4) and standardize parameters.

### ✅ Done
- [x] Dived into ENCODE database to find good candidates.
- [x] Set **SP4** (`ENCSR642PQK`) as anchor (CRISPR, HepG2, NovaSeq 6000) as it is the only high-quality dataset available.
- [x] Mapped **SP1/SP2** to matching technical profiles.
- [x] Excluded old datasets to prevent **Cell Line Bias** (standardized on HepG2) and maintain consistent **Genetic Background** (CRISPR).
- [x] Reconstructed project structure.

### 📊 Dataset Summary
| Protein | ENCODE ID | Cell Line | Platform |
| :--- | :--- | :--- | :--- |
| **SP1** | `ENCSR460YAM` | HepG2 | NextSeq 500 |
| **SP2** | `ENCSR946RZN` | HepG2 | NovaSeq 6000 |
| **SP4** | `ENCSR642PQK` | HepG2 | NovaSeq 6000 |

### 🚀 Next Steps
- [ ] Sketch data preparation pipeline.
- [ ] Design multi-modal mCNN modeling architecture.
- [ ] Further search for related works.

---

## 📚 2026-05-11 | Literature Review & Methodology

### 🔍 Key Findings
> [!NOTE]
> Found 2026 article: *"A DNABERT based deep learning framework for predicting transcription factor binding sites"*

### 🛠️ Proposed Pipeline
**Quality Data** ➡️ **Motif Analysis** ➡️ **3D-Structure Data** ➡️ **Outlier Analysis** (Focus Area)

#### Core Tools & Techniques:
- **DNABERT-2**: For advanced motif understanding.
- **3D DNA Structure**: Leveraging structural information from DNABERT-2.
- **Quantum Mapping**: For outlier explanation and feature enhancement.

---

## 📖 Detailed References

### 🔬 Quantum Parameters
> **"Quantum electronic and geometric parameters for DNA k-mers as features for machine learning"**
> *By: Kairi Masuda, Adib A. Abdullah, Patrick Pflughaupt & Aleksandr B. Sahakya*
> - Provides a comprehensive database of quantum electrical and geometric parameters for all 7-mer permutations.

### 🤖 BERT-Based Architectures
> **"BERT-TFBS: a novel BERT-based model for predicting transcription factor binding sites by transfer learning"**
> *By: Kai Wang, Xuan Zeng, Jingwen Zhou, Fei Liu, Xiaoli Luan, Xinglong Wang*
> - Pioneer paper for combining BERT and mCNN.

> **"A DNABERT based deep learning framework for predicting transcription factor binding sites"**
> *By: Pratik Dutta, Nimisha Ghosh & Daniele Santoni*
> - **Insight:** DNABERT is superior to traditional CNN/RNN for capturing long-range dependencies.
> - **Architecture:** Uses Modified Convolutional Block Attention Module (MCBAM) and Multi-Scale Convolutions with Attention (MSCA).

> **"A transformer architecture based on BERT and 2D convolutional neural network to identify DNA enhancers from sequence information"**
> *By: Nguyen Quoc Khanh Le, Quang-Thai Ho, Trinh-Trung-Duong Nguyen and Yu-Yen Ou*
> - **Insight:** BERT + 2D CNN combination outperforms traditional models.

---

### 🚀 Next Steps
- [ ] Sketch data preparation with DNA-Shape and DNA BERT2.
- [ ] Try the outliers' model, try to analyze and explain them(by modern visualization methods) through quantum parameters.
- [ ] Sketch for multi-modal mCNN modelling architecture.

