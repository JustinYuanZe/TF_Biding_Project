# Slide-by-Slide Presentation Outline: SP TF Binding Prediction

This document provides a professional, academic, and scientific structure for your presentation slides in English. It details a 17-slide flow showing the progression from simple baselines to your proposed Tri-Branch model, incorporating experimental results, file paths for figures, and model interpretability.

---

## Slide 1: Title & Overview
*   **Proposed Title**: 
    *   *Predicting SP Family (SP1, SP2, SP4) Transcription Factor Binding Sites Using a Multimodal Tri-Branch Deep Learning Model with Cross-Modal Attention*
*   **Information**: Presenter name, Advisor name, Date of defense.
*   **Design Concept**: Modern layout, slide corners decorated with a DNA double-helix combined with neural network nodes (Bio-AI concept).

---

## Slide 2: Introduction & Biological Background
*   **Content**:
    *   **SP Family Transcription Factors (SP1, SP2, SP4)**: Crucial DNA-binding proteins that regulate essential genes involved in cell cycle, metabolism, and apoptosis. Dysregulation of these TFs is heavily linked to cancer and cardiovascular diseases.
    *   **Importance of TFBS Prediction**: Accurately mapping SP family binding sites (TFBS) across the genome is key to understanding gene regulation mechanisms and developing targeted gene therapies.
*   **Visual**: Illustration of SP family proteins binding to a DNA strand or a basic gene transcription diagram.

---

## Slide 3: Motivation & The Core Challenge
*   **Content**:
    *   *Limitations of Prior Work*: 
        *   Traditional machine learning and sequence-only deep learning models (such as pure fine-tuned DNABERT-2) only read flat 1D sequence text (A, T, G, C), ignoring structural and biochemical context.
    *   *Biological Reality*: In vivo TF-DNA binding is heavily dictated by **3D DNA physical shape (DNAshape)** and surrounding **biochemical/epigenetic features (Bio-features)**.
    *   *Proposed Solution*: Establish a Multimodal Fusion framework: DNA Sequence (1D text) + 3D Physical Geometry (DNAshape) + Local Epigenetic Context (Bio-features: CpG islands, GC content, G4).
*   **Visual**: Side-by-side comparison of flat 1D sequence text vs. a u-shaped/bent 3D DNA shape structure u-bend.

---

## Slide 4: Dataset Preprocessing & Balancing
*   **Content**:
    *   **Raw Data Source**: 101 bp DNA sequences binding to SP1, SP2, and SP4 (Positive class) and non-binding controls (Negative class).
    *   **Label Mapping (Binary Classification)**:
        *   Label `1` (SP_Positive): Union of all positive sequences from SP1, SP2, and SP4.
        *   Label `0` (Negative): Non-binding control sequences.
    *   **Data Augmentation & Balancing**:
        *   **Reverse Complement (RC) Augmentation**: Doubles positive sequences from 1,696 to 3,392 per TF (total 10,176 positives) to help the model learn orientation-invariant motifs.
        *   **Dinucleotide Shuffling & Sampling**: Samples 3,392 negatives from a dinucleotide-shuffled pool to preserve CpG density without generating fake binding motifs, achieving a balanced dataset.
    *   **Data Splitting**: Group-based splitting (e.g., 80% Train, 20% Test) to ensure zero data leakage between subsets.
*   **Visual**: Flowchart of raw sequences $\rightarrow$ labeling $\rightarrow$ augmentation & shuffling $\rightarrow$ train/test splitting.

---

## Slide 5: Feature Engineering Pipeline
*   **Content**:
    *   Preprocessed inputs are split into three feature streams:
        1.  **Sequence Features**: Tokenized using the vocabulary of the pre-trained genomic model **DNABERT-2**.
        2.  **DNAshape Features**: Sliding pentamer (5-mer) lookup using the DNAshapeR database to extract 5 structural metrics: Minor Groove Width (MGW), Propeller Twist (ProT), Roll, Helix Twist (HelT), and Electrostatic Potential (EP). Output size: `[N, 5, 101]`.
        3.  **Biological Features**: Dynamic computation of **CpG Observed/Expected (O/E) ratio**, **GC content**, and regular expression (regex) search for **G-quadruplex (G4)** secondary structures.
*   **Visual**: Visual representation of the 5 DNAshape features and mathematical formulas for CpG O/E and GC%.

---

## Slide 6: Proposed Methodology: Tri-Branch Architecture
*   **Content**:
    *   Three parallel branches to process the multimodal features:
        1.  **Sequence Branch**: Fine-tuned **DNABERT-2** model (unfreezing the last 6 layers) with Mean-pooling to capture high-quality contextual sequence semantics.
        2.  **DNAshape Branch**: A **3-layer 1D CNN** with GroupNorm (multi-GPU DDP safe) to learn local spatial patterns of the DNA backbone.
        3.  **Bio-Features Branch**: A simple **MLP (Multi-Layer Perceptron)** to project biological/epigenetic features.
*   **Visual**: Block diagram of the Tri-Branch architecture showing the three branches converging.

---

## Slide 7: Cross-Modal Attention & Fusion Mechanism
*   **Content**:
    *   Bidirectional feature interaction between Sequence (Branch 1) and DNAshape (Branch 2) via a **Cross-Modal Attention Layer**:
        *   Dual-way attention: DNA sequence representation acts as a Query to extract spatial structure (Shape) details, and vice versa.
        *   Maps nucleotide sequence context directly to its physical geometry.
    *   **Fusion & Classification**:
        *   Pooled Sequence-Shape representations are combined with the biological MLP features.
        *   Fed into the **Classifier Head (MLP + Dropout)** to output binary binding probability (via Sigmoid).
*   **Visual**: Sơ đồ Attention (Q, K, V interaction between Sequence and Shape).

---

## Slide 8: Experimental Setup & Training Hyperparameters
*   **Content**:
    *   **Loss Function**: `BCEWithLogitsLoss`.
    *   **Class Imbalance Handling**: Since SP1, SP2, SP4 are grouped, the Pos:Neg ratio is 3:1 (10,176 vs 3,392). We compute dynamic class weights `pos_weight = neg_count / pos_count ≈ 0.33` to prevent class bias.
    *   **Optimization**: AdamW optimizer with Cosine Annealing learning rate schedule.
    *   **Branch-specific Learning Rates**: Backbone LR: 2e-5, Shape CNN: 3e-4, Cross-Attn: 2e-4, Bio MLP: 2e-4.
    *   **Hardware Setup**: Distributed Data Parallel (DDP) via PyTorch Accelerate on **2x GPU Tesla T4 (Kaggle)**, effective batch size = 64.

---

## Slide 9: Classical ML Baselines
*   **Content**:
    *   Training results of 10 classical ML models (SVM, Random Forest, Logistic Regression, MLP, etc.) across two feature representations:
        1.  **One-Hot Encoding**: Flat binary representations.
        2.  **k-mer TF-IDF**: Word-frequency scores of DNA sub-strings.
    *   **Performance Analysis**:
        *   Best-performing traditional model highlighted (e.g., SVM or Random Forest).
        *   Discussion: Classical ML reaches an performance ceiling due to the lack of long-range context modeling and total neglect of DNA 3D structural shape.
*   **Visual (Figures to insert)**:
    *   `figures/final/classic/classical_onehot_accuracy.png` & `classical_tfidf_accuracy.png` (Performance comparison).
    *   `figures/final/classic/classical_onehot_curves.png` & `classical_tfidf_curves.png` (ROC and PR curves).
    *   `figures/final/classic/classical_onehot_confusion_matrix.png` (Confusion matrix of the best model).

---

## Slide 10: Sequence-Only Model: DNABERT-2
*   **Content**:
    *   Fine-tuning results of the pre-trained genomic model **DNABERT-2**.
    *   **Training Dynamics**:
        *   Loss and accuracy curves for Train vs. Validation. Overfitting gap analysis.
    *   **Evaluation**: Copes well with sequence-level patterns due to Self-Attention, but struggles at sites where binding relies on 3D spatial conformations.
*   **Visual (Figures to insert)**:
    *   `figures/final/BERT_only/finetune_training_curves.png` (Convergence curves).
    *   `figures/final/BERT_only/finetune_confusion_matrix.png` (Sequence-only confusion matrix).
    *   `figures/final/BERT_only/finetune_roc_curves.png` & `finetune_pr_curves.png` (ROC and PR curves).

---

## Slide 11: Cross-Modal Model (Sequence + DNAshape)
*   **Content**:
    *   Performance of the Sequence (DNABERT-2) and Shape (DNAshape) multimodal model combined via Cross-Modal Attention.
    *   **Significant Performance Jump**: Reaches **~96.01% val_acc** at epoch 14, exhibiting a dramatic boost over sequence-only models (Only BERT).
    *   **Discussion**: Confirms that integrating 3D structural shape parameters via bidirectional cross-modal attention is the primary driver of the performance boost, showing the critical value of DNA physical shape features.
*   **Visual (Figures to insert)**:
    *   `figures/cross_modal/crossmodal_training_curves.png` (Training loss and accuracy curves).
    *   `figures/cross_modal/crossmodal_confusion_matrix.png` (Confusion matrix showing prediction counts).
    *   `figures/cross_modal/crossmodal_roc_curves.png` & `crossmodal_pr_curves.png` (ROC and PR curves showing classification capability).

---

## Slide 12: Proposed Model: Tri-Branch (G-CMAB + MSCNN + Bio-Features)
*   **Content**:
    *   Experimental results of the proposed Tri-Branch architecture combining Sequence, Shape, and Bio-features.
    *   **Detailed Performance**:
        *   Achieves the highest overall score, stable convergence, and robust gradients.
        *   Confusion matrix analysis shows excellent positive recall and negative filtering.
*   **Visual (Figures to insert)**:
    *   `figures/final/Proposed Model/gcmab_mscnn_training_curves.png` (Proposed model training curves).
    *   `figures/final/Proposed Model/gcmab_mscnn_confusion_matrix.png` (Proposed model confusion matrix).
    *   `figures/final/Proposed Model/gcmab_mscnn_roc_curves.png` & `gcmab_mscnn_pr_curves.png` (Best ROC and PR curves).

---

## Slide 13: Comparative Benchmarking & Analysis
*   **Content**:
    *   **Unified Comparison Table**: Accuracy, F1, Precision, Recall, ROC-AUC, PR-AUC of all 4 tested model groups.
    *   **Evolutionary Analysis**: Shows clear steps of improvement: Classical ML $\rightarrow$ BERT-Only $\rightarrow$ Cross-Modal $\rightarrow$ Proposed Tri-Branch.
    *   **Key Takeaway**: Highlight **Negative Recall (Ability to filter out false positives)**.
*   **Visual**:
    *   Comparison table and joint ROC/PR curves (highlighting proposed model in bold).

---

## Slide 14: Ablation Analysis & Discussion
*   **Content**:
    *   **Role of DNAshape**: Shape CNN integration causes a huge leap (~96%), proving TFs read groove geometry (MGW, Roll, HelT) rather than just sequence code.
    *   **Role of Bio-features**: CpG islands and GC content act as biological markers of promoter regions, refining the final predictions.
    *   **Biological Saturation Limit**: Explain why accuracy plateaus at ~96-97%. 101 bp sequences carry limited information, and the Proposed model has fully exhausted the sequence, structure, and epigenetic context.

---

## Slide 15: Model Interpretability: SHAP Analysis
*   **Content**:
    *   Explaining the deep learning model using **SHAP (SHapley Additive exPlanations)**.
    *   **Branch-specific SHAP Contributions**:
        *   *Sequence*: Key k-mers and binding motifs.
        *   *DNAshape*: Structural dimensions (like MGW) most crucial for SP binding.
        *   *Bio-features*: Influence of GC content and CpG density on the model's logits.
*   **Visual**: SHAP Summary/Force plots from `explain_shap_tribranch.py`.

---

## Slide 16: Conclusion & Future Work
*   **Conclusion**:
    1.  Built a state-of-the-art Tri-Branch model utilizing multimodal fusion for SP family TFBS prediction.
    2.  Multimodal fusion outperforms sequence-only LMs and classical ML baselines.
    3.  Interpretable via SHAP.
*   **Future Work**: Generalization to other TF families, ATAC-seq integration.

---

## Slide 17: Q&A
*   Acknowledgements and Q&A session.

---

# Tips for the Presentation

1.  **Storytelling Approach**: Start with biological significance $\rightarrow$ present limitations of sequence-only representation $\rightarrow$ propose shape features $\rightarrow$ implement Tri-Branch architecture $\rightarrow$ demonstrate performance boost $\rightarrow$ explain black-box using SHAP.
2.  **Animations for Slides 4, 5, 6, 7**:
    *   Use **Fade-in animations** for branches and layers to prevent audience cognitive overload. Introduce Sequence $\rightarrow$ Shape $\rightarrow$ Bio-features $\rightarrow$ Cross-Modal Attention $\rightarrow$ Classifier Head step-by-step.
3.  **Visual Focus**: Keep slides visual-heavy. Enlarge ROC/PR curves and use clean, simplified tables.
4.  **Hidden Slides Strategy (Backup slides after Slide 17)**:
    *   **Slide 18 (DNAshape Boxplots)**: Show similar structural parameter distributions across SP1, SP2, and SP4 paralogs to justify binary grouping.
    *   **Slide 19 (Efficiency Comparison)**: Compare training time and GPU memory consumption. Explain why Proposed is faster and more memory-efficient than BERT-only due to dynamic sequence lengths (length 48).
5.  **Preparation for Challenging Questions**:
    *   *Question*: Why do we need the Proposed Tri-Branch if the Dual-Branch already reaches 96% accuracy?
    *   *Answer*: While DNAshape provides the bulk of the geometric context, Bio-features (GC% and CpG islands) represent biological promoter signatures, which help regularize the network and lower false positive rates in complex genomic regions.
