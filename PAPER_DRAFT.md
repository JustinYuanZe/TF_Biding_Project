# Multimodal Deep Fusion of Genomic Sequence, DNA Shape, and Biochemical Context for Predicting SP-Family Transcription-Factor Binding Sites

*Manuscript draft — IMRaD format. All quantitative results reported here are taken directly from the saved evaluation artifacts in the repository (the `classification_report.txt` files under `figures/`). Where the project's README quotes figures that are not backed by a saved report, those figures are deliberately omitted; see the "Note on Reported Numbers" at the end.*

---

## Abstract
Accurately mapping the binding sites of transcription factors is an important approach to understanding the mechanism of gene regulation. Transcription factors of the SP family (SP1, SP2, SP4) play a central role in cell-cycle and metabolic processes, yet are challenging to fully predict due to their paralog origin. We implemented a multimodal deep-learning framework to predict SP-family binding sites from 101-bp DNA windows by using a fusion of 3 complementary methods: semantic sequence from a pre-trained genomic transformer DNABERT-2, local 3-dimensional geometry from DNAShapeR, and biochemical information (CpG O/E ratio, GC content, and G-quadruplex motifs). The branches of DNABERT-2 and DNAShapeR are coordinated through a bidirectional cross-modal attention mechanism and a multi-scale depthwise-separable CNN (msCNN), with DDP-safe Group Normalization—together forming the core G-CMAB module (GroupNorm + Cross-Modal Attention + multi-scale Branch). The branches are then concatenated before a sigmoid classification head. We initially attempted to classify the three datasets individually but found that the performance remained plateau at approximately 61% accuracy. This is due to their biological nature—they share a nearly identical GC-box motif and paralogous origin—making differentiation within a 101bp window unreliable as it doesn't adequately reflect these biological characteristics. A diagnostic binary sanity check was applied to verify the pipeline's accuracy and motivated a reformulation into binding-vs-non-binding classification. The proposed model achieved 96.35% accuracy, while the sequence-only DNABERT-2 model scored 76.24%. A SHAP analysis module was also used to extract information about sequence, shape, and biochemical features. A correct/incorrect detection was also implemented for validation.
---

## 1. Introduction

### 1.1 Biological background
SPx transcription factors significantly influence essential regulatory processes such as proliferation, metabolism, and apoptosis. Conversely, once dysregulation occurs, manifested through genetic mutations, epigenetic alterations, or even competition between transcription factors, it can lead to serious problems such as cancer and cardiovascular disease. Therefore, mapping binding sites are vital for understanding mechanisms and researching new treatments. Chromatin immunoprecipitation sequencing (ChIP-seq) provides experimentally derived binding maps. However, computational predictive models offer a more cost-effective and efficient approach.


### 1.2 Limitations of prior approaches
Previous machine learning predictive models, and even recent deep learning models, relied on the raw power of pretrained models, which were essentially semantic extensions of flat 1-D sequences (A,T,G,C). This resulted in the ignoring of physical shape elements of the double helix and surrounding biochemical factors, which are crucial features that strongly influence binding sites. Although pretrained models like DNABERT-2 capture widely spaced dependencies on 1-D sequences much better than previous baselines, they are still merely extensions of 1-D sequences. The inevitable consequence is that these models tend to overpredict binding status. Another issue concerns data leakage. Many previous results have shown extremely good performance; however, with the same architecture and dedicated fine-tuning, the reality is that with a sufficiently robust dataset, these models are mostly unable to reproduce the same good results. This inconsistency in output across different datasets leads to inconsistencies and difficulties for large-scale studies.

### 1.3 Contributions
This study was conducted with the ambition that incorporating 3-D structural information and epigenetic features into a 1-D embeddings sequence via a modern pretrained model would create a breakthrough in reducing false positive prediction rates in the SP-family Transcription Factor binding site prediction problem compared to models using only conventional sequences.

We make four significant contributions on this work:

(i) A complete data preprocessing pipeline, including the entire process of extracting data from ENCODE, balancing data, and creating motif-preserving, zero-data-leakage negative datasets.

(ii) A multimodal tri-branch architecture combining the SOTA sequence embeddings (DNABERT-2) branch, the biological geometric feature preservation (DNAShapeR) branch, and the biochemical parameter branch, and efficiently combined through bidirectional cross-modal attention and passed through a multi-scale separate CNN for maximum computational efficiency.

(iii) A comprehensive development process from distinguishing paralogs to binary prediction.

(iv) An Interpretable Model through SHAP analysis implementation.

---

## 2. Related Work
Previous works, such as the sequence-only DeepBind (Alipanahi et al.), a classic baseline for motif learning, were followed by pioneers like BERT-TFBS (Wang et al.) and the DNABERT-based MCBAM/MSCA framework (Dutta, Ghosh & Santoni) in using transformer embeddings with multi-scale convolutions and attention for TFBS prediction. Additionally, work on BERT+2D-CNN architectures (Le et al.) has shown that transformer-CNN hybrids beat traditional models despite differing problem-solving approaches. However, gaps remain regarding the lack of cross-interaction between space and sequence. The works on DNABERT-2 (Zhou et al.) and DNAshape/DNAshapeR (Chiu et al.) provide a pentamer query table for helix geometry prediction, offering powerful tools for sequence and 3D geometry information, but lacking a truly robust bridge. To fill this research gap, we propose a cross-modal attention mechanism, aiming to maximize the power of these works while creating value through the integration of 3D and sequence information.

---

## 3. Materials and Methods

### 3.1 Data source and curation
To obtain a balanced and zero-leak dataset, we first obtained datasets from the ENCODE TF ChIP-seq narrowPeak source with the same HepG2 cell line on the hg38 reference (SP1: ENCFF333SWC; SP2: ENCFF480YAW; SP4: ENCFF938KVY). To ensure that the datasets did not contain any confusion or bias related to issues such as cell line and genetic background, the selected datasets were confirmed to be consistent with the same subjects in the experiment and the CRISPR/NGS technique. From here, with information from the 10 columns of the 3 narrowPeak datasets (chrom, start, end, name, score, strand, signal, p-value, q-value, peak-offset), we performed techniques to filter and extract the data.

### 3.2 Peak filtering and quality control
There is a large difference between the datasets in terms of both quantity and quality(16,043, 11,301, and 23,574 samples for SP1,SP2,SP4 quantities and 4.03, 2.436 and 2.744 for mean -log10(q) values). Therefore, we performed a filter by choosing a threshold −log10(q) ≥ 2.0 to ensure that we obtain q ≤ 0.01 after filtering. After filtering, SP1 retained a total of 15,244 samples (95.02%), 6,453 (57.10%) for SP2, and 17,477 (74.14%). Meanwhile, mean -log10(q) improved to 4.177 (+3.6%) for SP1, 3.572 (+46.6%) for SP2, and 3.289 (+19.9%) for SP4.

Additionally, because many paralogous SPs attach to the same location, easily leading to training and prediction errors, we performed steps to remove overlaps, thereby obtaining 6,815 / 1,696 / 8,914 exclusive peaks for SP1, SP2, and SP4, respectively. From here, the baseline of 1,696 exclusive peaks for SP2 (the minority class after overlap removal) was used as a benchmark for the remaining classes. We downsampled SP1 and SP4 down to this 1,696 benchmark by selecting the top-quality peaks based on descending q-value, creating a balanced positive set of 1,696 peaks per TF class (5,088 total peaks before data augmentation).


### 3.3 Sequence extraction and windowing
We created fixed-size windows of 101bp starting from the offset (corresponding to -50bp and +50bp from the offset), ensuring that the input for the transformer and convolutional branches was fixed. Next, the sequence extraction process was performed, starting from hg38 FASTA. We used a byte-seek reader (`src/extract_fasta.py`) that tolerates chr-prefix mismatches (e.g., chr1 vs 1), with boundary peaks (summit − 50 < 0) discarded. All bases were upper-cased (`src/capitalize_fasta.py`) and N bases were preserved for downstream handling.


### 3.4 Negative-set construction
The primary negative set was produced by **dinucleotide shuffling** (Altschul–Erickson Eulerian-path algorithm, `src/generate_negatives.py`), which destroys binding motifs while exactly preserving mononucleotide and dinucleotide frequencies — and therefore GC and CpG density (verified δGC = 0.00%, `scratch/negatives_report.txt`). This design deliberately makes negatives non-trivial: they cannot be separated from positives by simple base composition, forcing the model to learn genuine motif/shape signals rather than GC content. A supplementary generator (`src/generate_negatives_v2.py`) additionally implements GC-matched random genomic negatives (with ENCODE blacklist ENCFF356LFX exclusion, ±500 bp peak padding, GC binning, and Cohen's-d quality control) and CpG-island negatives (UCSC `cpgIslandExt`) as robustness alternatives.

### 3.5 Augmentation, balancing, and splitting
Positive sequences were doubled by **reverse-complement (RC) augmentation** (1,696 → 3,392 per target) so the model learns strand-orientation-invariant motifs (`src/prepare_final_dataset.py`); negatives were not RC-augmented. The per-target imbalance (SP1 > SP4 > SP2) after overlap removal was resolved by downsampling to the SP2 exclusive peak baseline of 1,696 peaks (`src/downsample.py`), and then applying the RC augmentation to obtain 3,392 sequences per TF class. The Negative class was constructed by sampling 1,131 sequences from SP1 shuffled negatives, 1,131 from SP2 shuffled negatives, and 1,130 from SP4 shuffled negatives, producing a balanced 1:1:1:1 corpus of 3,392 sequences per class (13,568 total). Data were split 80/20 train/test with **GroupShuffleSplit**, keeping each original sequence and its reverse complement in the same fold to eliminate leakage; the held-out test set contains 2,710 sequences (SP1 726, SP2 668, SP4 634, Negative 682).

### 3.6 Feature engineering — three modalities
**(a) Sequence.** Windows were tokenized with the DNABERT-2 byte-pair-encoding (BPE) vocabulary; the maximum token length was set dynamically to the training-set p99 (clamped to [32, 96]) to save memory. **(b) DNAshape.** Five structural parameters — Minor Groove Width (MGW), Propeller Twist (ProT), Roll, Helical Twist (HelT), and Electrostatic Potential (EP) — were computed per position via a sliding pentamer (5-mer) lookup table (`src/dnashape_lookup.py`, `src/extract_dnashape.py`), with reverse-complement fallback for missing pentamers and NaN handling at window edges, producing an [N, 5, 101] tensor. **(c) Bio-features.** Three biochemical descriptors were computed per window: the CpG observed/expected ratio (promoter marker), GC content (stability marker), and presence of G-quadruplex (G4) motifs detected by regular expression.

### 3.7 Classical machine-learning baselines
As a lower bound (`notebooks/27_binary_classical_ml_baselines.py`), ten classical classifiers were trained — Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, linear SVM, k-NN, Bernoulli Naïve Bayes, and an MLP — on two sequence representations: flat one-hot encoding ([N, 404]) and character k-mer TF-IDF (n-gram range 4–6). Models were evaluated under an identical stratified 80/20 split with accuracy, F1, precision, recall, ROC-AUC, and average precision. (These baselines are saved as figures only — `figures/final/classic/` — so no numeric report is cited here.)

### 3.8 Sequence-only DNABERT-2 baseline
The sequence-only model (`notebooks/24_binary_dnabert2_finetune_kaggle.py`) fine-tunes DNABERT-2-117M with its last 3 of 12 encoder layers unfrozen. Token embeddings are pooled by concatenating the [CLS] vector with a masked mean over all tokens ([B, 1536]) and passed through LayerNorm → Linear(1536→256) → GELU → Dropout(0.3) → Linear(256→1). It is trained with BCEWithLogitsLoss (positive class weight = n_neg/n_pos), AdamW (backbone 1.5e-5, head 1e-4), linear-warmup + cosine schedule, and serves as the "sequence-only" reference point.

### 3.9 Proposed multimodal architecture (Tri-Branch / G-CMAB)
The proposed model (`notebooks/23_binary_tribranch_kaggle.py`) has three parallel branches feeding a fusion head. **Sequence branch:** DNABERT-2 with the last 6 layers unfrozen, an ELMo-style learned scalar-mix ("layer attention") over the last six hidden states, projected to a 128-d token sequence. **Shape branch:** a 1×1 Conv projection of the [B,5,101] tensor to 128 channels with GroupNorm + GELU. **Bio branch:** a small MLP (3→16→32) over the biochemical features. The 1-D sequence tokens and the spatial shape map are coupled by a **bidirectional cross-modal multi-head attention** module (d_model 128, 4 heads, 1 layer, with residual + LayerNorm + feed-forward): sequence queries shape and shape queries sequence, directly aligning nucleotide context to helix geometry.

### 3.10 Multi-scale CNN and fusion head
After cross-modal interaction, both the attended sequence and shape streams pass through a **multi-scale depthwise-separable CNN (msCNN)**: parallel kernels of sizes {7,9,11,15} (sequence) and {4,8,12,16} (shape), each with GroupNorm, GELU, and 1-max pooling, capturing motifs of multiple widths while keeping parameters low. The pooled sequence (1,024-d), pooled shape (1,024-d), and bio (32-d) vectors are concatenated (2,080-d) and classified by Linear(2080→256) → GELU → Dropout(0.7) → Linear(256→1) with a sigmoid output. The umbrella name **G-CMAB** denotes this combination of **G**roupNorm + **C**ross-**M**odal **A**ttention + multi-scale **B**ranch.

### 3.11 Training and optimization
All deep models use **BCEWithLogitsLoss** with a dynamically computed positive-class weight (n_neg/n_pos ≈ 0.33 for the merged binary task) to counteract the 3:1 positive:negative imbalance. Optimization uses **AdamW** with **branch-specific learning rates** (backbone 2e-5; sequence/shape projections and cross-attention 2e-4–3e-4; layer-attention 1e-3), weight decay 0.1, linear warmup (≈15% of steps) followed by cosine annealing, and gradient clipping (max-norm 0.5–1.0). Training ran up to 30 epochs (batch 16, gradient accumulation giving an effective batch of 32–64) with early stopping (patience 5–12) capped by a 30% train/val overfitting-gap guard, saving the best-validation checkpoint.

### 3.12 Distributed-training and systems engineering
Training used HuggingFace **Accelerate** for Distributed Data Parallel (DDP) on 2× NVIDIA Tesla T4 GPUs (Kaggle) with **bf16** mixed precision, `find_unused_parameters=True`, cross-process metric gathering, and synchronized checkpointing. **GroupNorm replaces BatchNorm** throughout the convolutional branches because BatchNorm statistics are unstable for the small per-GPU batches under DDP. Because DNABERT-2's custom Triton flash-attention and ALiBi buffers fail to load on T4/Kaggle, the backbone is loaded through a pure-PyTorch attention monkey-patch and a three-strategy loader that avoids meta-device allocation (`src/dnabert_wrapper.py`).

### 3.13 Task reformulation: 3-class → binary
The project first framed prediction as 4-way classification (SP1 / SP2 / SP4 / Negative). Because this plateaued near 61% (Section 4.2), a **binary sanity check** (`notebooks/13_binary_sanity_check_kaggle.py`) merged all positives into one class to verify the pipeline; its reasonable accuracy confirmed the difficulty was inter-paralog confusion, not a code defect. The final task was therefore reformulated as **binding (SP1∪SP2∪SP4) vs non-binding**, which is biologically justified because the three paralogs recognize an almost identical GC-box and exhibit overlapping DNAshape distributions (Section 4.1).

### 3.14 Evaluation and interpretability
Models are evaluated with accuracy, per-class precision/recall/F1, macro/weighted averages, ROC-AUC, PR-AUC, and confusion matrices, with training, ROC, and PR curves saved per experiment. Interpretability uses **SHAP GradientExplainer** (`notebooks/explain_shap_tribranch.py`; ~50 background, ~20 explained samples) to attribute logits across the three modalities — sequence tokens/k-mers, the five DNAshape parameters, and bio-features — aligned to the GC-box center over configurable flanking windows, and rendered as summary-bar, heatmap, line, and force plots. Correctly and incorrectly classified loci are exported as genomic BED tracks (`True_SP1.bed`, `True_SP4.bed`, `Confused_SP4_as_SP1.bed`) for downstream motif-enrichment and chromatin follow-up.

---

## 4. Results

### 4.1 Exploratory data analysis
EDA of the five DNAshape parameters on the merged corpus (`scratch/dnashape_eda_report.md`) showed near-symmetric MGW and HelT but strongly skewed Roll and EP, with the strongest pairwise coupling between ProT and EP (r = 0.77) and between MGW and Roll (r = 0.54). Crucially, position-wise shape distributions and GC content are very similar across SP1, SP2, and SP4 (`figures/dnashape_boxplots.png`), consistent with their shared GC-box and supporting the binary grouping. Diagnostic analysis (`src/diagnose_pipeline.py`) further showed that the discriminative core motif occupies only ~8–10 bp of the 101 bp window and that dinucleotide-shuffled negatives are GC-matched to positives — explaining why the task is genuinely hard.

### 4.2 The multiclass plateau and paralog confusion
Across every 4-way model the accuracy saturated at ~50–62% (Table 1). The consistent pattern is diagnostic: the **Negative class is recovered well** (recall ≈ 0.86–0.88, precision ≈ 0.92–0.95 in the best models), whereas the three **SP paralogs are heavily confused with one another** (SP4 recall as low as 0.33–0.57). In other words, "is this a binding site?" is learnable, but "which SP paralog?" is not, given a 101 bp window — because the paralogs share their motif and shape signature. This finding, not a pipeline bug, drove the binary reformulation.

**Table 1 — 4-way classification (held-out test, n = 2,710). Source: `figures/**/classification_report.txt`.**

| Experiment (script) | Acc. | Macro-F1 | SP1 R | SP2 R | SP4 R | Neg R | Neg P |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Balanced Dual-Branch (12) | 0.5041 | 0.4975 | 0.504 | 0.512 | 0.334 | 0.654 | 0.532 |
| KAN cross-modal (15) | 0.5343 | 0.5400 | 0.347 | 0.311 | 0.738 | 0.763 | 0.976 |
| Hierarchical KAN (20) | 0.5609 | 0.5549 | 0.546 | 0.469 | 0.420 | 0.799 | 0.701 |
| Tri-branch msCNN (22) | 0.6074 | 0.6072 | 0.657 | 0.448 | 0.450 | 0.858 | 0.950 |
| Simplified cross-modal (18) | 0.6081 | 0.6069 | 0.616 | 0.518 | 0.424 | 0.859 | 0.924 |
| Cross-modal dual (14) | 0.6133 | 0.6140 | 0.623 | 0.484 | 0.486 | 0.849 | 0.925 |
| G-CMAB msCNN (21) | 0.6137 | 0.6179 | 0.467 | 0.548 | 0.566 | 0.878 | 0.936 |
| **G-CMAB safe (19)** | **0.6151** | **0.6122** | 0.569 | 0.560 | 0.445 | 0.877 | 0.872 |

### 4.3 Binary classification (principal result)
Under the binary formulation, the multimodal G-CMAB model achieves a verified **96.35% accuracy** and **macro-F1 0.9501** (Table 2), versus **76.24%** for the fine-tuned sequence-only DNABERT-2 and **77.08%** for the merged binary sanity check. The decisive improvement is in negative rejection: **Negative recall rises from 0.324 (sequence-only) to 0.886, and Negative precision from 0.547 to 0.966**, while SP-Positive recall reaches 0.990. This confirms that the sequence-only model systematically over-predicts binding on GC-rich non-binding DNA, and that adding 3-D shape and biochemical context — coupled by cross-modal attention — is what lets the model confidently reject negatives.

**Table 2 — Binary classification (held-out test, n = 2,710; Negative 682, SP-Positive 2,028). Source: saved `classification_report.txt`.**

| Model (script) | Accuracy | Macro-F1 | Neg P | Neg R | Pos P | Pos R |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Sequence-only DNABERT-2 (24) | 0.7624 | 0.6292 | 0.547 | 0.324 | 0.800 | 0.910 |
| Binary sanity check (13) | 0.7708 | 0.6780 | 0.553 | 0.465 | 0.829 | 0.874 |
| **Proposed multimodal G-CMAB (26)** | **0.9635** | **0.9501** | **0.966** | **0.886** | **0.963** | **0.990** |

### 4.4 Architecture ablation and evolution
The 28-experiment trajectory is itself an ablation (Section 5.3). Adding DNAshape to sequence (dual-branch, script 11–12) was initially neutral until a projection bottleneck, asymmetric dropout, and differential learning rates stopped the high-dimensional BERT features from drowning out the shape branch. Replacing global pooling with a spatial shape CNN (script 14) enabled meaningful cross-modal attention, and GroupNorm + layer-attention + multi-scale CNN ("G-CMAB", scripts 19–22) gave the best and most stable multiclass models. Attempts that bundled many untested changes at once collapsed to near-random accuracy (script 16, ~25%), and a Kolmogorov–Arnold-Network classifier (ChebyKAN, script 15) over-regularized to 53%; both confirmed that single-flag, incremental changes were necessary.

### 4.5 Interpretability and genomic output
SHAP attributions decompose each prediction into sequence, DNAshape, and bio-feature contributions, localized around the GC-box, and are saved as per-modality bar/heatmap/line plots (`figures/SHAP/`, `figures/script21/`). The analysis contrasts correctly classified versus confused loci, providing a feature-level account of where and why the model errs. Correct and confused predictions are also exported as BED tracks (`True_SP1.bed`, `Confused_SP4_as_SP1.bed`), enabling genome-browser inspection and downstream motif/chromatin analysis.

---

## 5. Discussion

### 5.1 Principal findings
A 101 bp window contains enough information to decide *whether* an SP-family factor binds, but not reliably *which* paralog binds, because SP1/SP2/SP4 share an almost identical GC-box and overlapping shape profiles. Multimodal fusion of sequence, 3-D shape, and biochemical context — aligned by bidirectional cross-modal attention — raises binary accuracy from 76% (sequence-only) to a verified 96.35%. The dominant mechanism is improved specificity: shape and biochemical context let the model reject GC-rich non-binding DNA that fools a sequence-only transformer.

### 5.2 Why DNA shape and biochemical context help
Transcription factors read groove geometry (MGW, Roll, HelT) and electrostatics, not just the nucleotide string, so explicit shape features supply information the sequence stream cannot easily recover from a short window. CpG O/E ratio and GC content act as promoter/CpG-island markers that regularize the decision boundary, and the cross-modal attention maps each nucleotide context onto its physical conformation rather than treating the modalities independently. Because dinucleotide-shuffled negatives are GC-matched, these gains cannot be attributed to trivial composition shortcuts.

### 5.3 Problems solved (engineering and methodological)
The project resolved a broad set of concrete issues, summarized in Section 7. These span distributed-training stability (BatchNorm→GroupNorm; `find_unused_parameters`; bf16), backbone compatibility (Triton-free attention and ALiBi meta-device loading), modality imbalance and branch collapse (projection bottleneck, asymmetric dropout, spatial pooling, late fusion), numerical/edge issues (NaN shape values, gradient clipping), data integrity (leakage-free grouped splitting, motif-preserving negatives, cross-target overlap removal, FASTA case/prefix handling), memory/throughput on T4 (dynamic sequence length, fp16 embedding cache, gradient accumulation), and overfitting control (fine-tuning, dropout, label smoothing, early stopping with an overfitting-gap guard). The most consequential methodological fix was diagnosing the multiclass plateau and reformulating the task to binary.

### 5.4 Limitations
The model is trained on a single cell line (HepG2) and three paralogs, so generalization to other cell types and TF families is untested. The binary formulation deliberately sets aside paralog identification, which remains unsolved at 101 bp resolution; longer windows or additional assays may be required. Finally, a subset of the project's headline baseline numbers (e.g., the classical SVM and a higher BERT-only accuracy) are not backed by saved reports and are therefore excluded here pending re-evaluation.

### 5.5 Performance ceiling
The binary accuracy plateaus near 96%, consistent with a near-exhaustion of the information available in a 101 bp window once sequence, shape, and biochemical context are all used. Further gains likely require longer context, chromatin/accessibility signal (e.g., ATAC-seq), or quantitative binding affinity rather than binary peaks.

---

## 6. Conclusion
We built a multimodal "tri-branch / G-CMAB" deep-learning model that predicts SP-family TF binding by fusing DNABERT-2 sequence embeddings, DNAshape geometry, and biochemical features through bidirectional cross-modal attention and a multi-scale CNN. After an evidence-based pivot from per-paralog to binding-vs-non-binding classification, the model reaches a verified 96.35% accuracy, substantially outperforming a sequence-only transformer chiefly by rejecting GC-rich negatives, and remains interpretable via SHAP and BED-track export. Future work includes generalization to additional TF families and cell types, integration of chromatin-accessibility data, and revisiting paralog discrimination with longer genomic context.

---

## 7. Complete Inventory of Techniques Used
*(Explicit, exhaustive list as requested; each item was implemented in the project.)*

**A. Data acquisition & curation**
- ENCODE HepG2 ChIP-seq narrowPeak ingestion (SP1 ENCFF333SWC, SP2 ENCFF480YAW, SP4 ENCFF938KVY), hg38.
- q-value filtering at −log10(q) ≥ 2.0; per-target QC of counts, q-value, signal, and length.
- Summit-centered 101 bp windowing (summit ± 50).
- Cross-target exclusive-peak assignment (overlap → highest-q-value target).
- Class down-sampling to the SP2 baseline by highest q-value.
- Indexed-FASTA sequence extraction with `chr`-prefix tolerance, upper-casing, N-base preservation, boundary filtering.

**B. Negative generation**
- Dinucleotide shuffling (Altschul–Erickson Eulerian path) preserving mono/dinucleotide and GC/CpG composition.
- (Supplementary) GC-matched random genomic negatives with ENCODE-blacklist exclusion, peak padding, GC binning, Cohen's-d QC.
- (Supplementary) CpG-island negatives from UCSC `cpgIslandExt`.

**C. Augmentation, balancing, splitting**
- Reverse-complement augmentation of positives (×2).
- Balanced 1:1:1:1 corpus (3,392/class).
- Leakage-free GroupShuffleSplit keeping original+RC pairs together; stratified 80/20.

**D. Feature engineering (3 modalities)**
- Sequence: DNABERT-2 BPE tokenization; dynamic max length (p99, clamp [32,96]).
- DNAshape: 5 parameters (MGW, ProT, Roll, HelT, EP) via pentamer sliding lookup with RC fallback and edge-NaN handling → [N,5,101]; robust per-feature scaling (P1–P99).
- Bio-features: CpG O/E ratio, GC content, G-quadruplex regex detection.

**E. Models**
- Classical ML (10): LogReg, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, linear SVM, k-NN, Bernoulli NB, MLP — on one-hot and k-mer TF-IDF (4–6-mers).
- One-hot multi-scale CNN (DeepBind-style) deep baseline (`src/train_onehot_mcnn.py`, `src/mcnn_model.py`).
- Frozen-DNABERT-2-embedding + multi-scale CNN classifier (initial deep approach, `notebooks/09_*`).
- DNABERT-2-117M fine-tuning (3 or 6 layers unfrozen) with [CLS]+mean pooling and MLP head.
- Dual-branch DNABERT-2 + DNAshape CNN.
- Spatial Shape CNN (pooling deferred to preserve position).
- Bidirectional cross-modal multi-head attention (seq↔shape Q/K/V, d=128, 4 heads).
- ELMo-style layer-attention (scalar-mix over BERT hidden states).
- Multi-scale depthwise-separable CNN (msCNN) with multiple kernel sizes and 1-max pooling.
- Multi-pool aggregation (mean‖max for sequence, K-max for shape) and strided-conv downsampling.
- Tri-branch fusion (sequence + shape + bio) → sigmoid head ("G-CMAB").
- Kolmogorov–Arnold Network (ChebyKAN) classifier and a hierarchical-KAN variant (explored).

**F. Training & optimization**
- BCEWithLogitsLoss with dynamic positive-class weighting (binary); weighted cross-entropy + label smoothing (multiclass).
- AdamW with branch-specific learning rates; weight decay 0.1.
- Linear warmup + cosine-annealing schedule; gradient clipping; gradient accumulation.
- GroupNorm (DDP-safe) replacing BatchNorm.
- Early stopping with overfitting-gap guard; best-checkpoint saving.

**G. Systems / distributed training**
- HuggingFace Accelerate DDP on 2× Tesla T4; bf16 mixed precision; `find_unused_parameters`; cross-process metric gathering.
- Triton-free pure-PyTorch flash-attention replacement; three-strategy DNABERT-2 loader avoiding ALiBi meta-device errors; CUDA-safe device probe.
- fp16 embedding caching; dynamic sequence length for memory/throughput.

**H. Evaluation & interpretability**
- Accuracy, per-class P/R/F1, macro/weighted averages, ROC-AUC, PR-AUC, confusion matrices; training/ROC/PR curve plotting.
- SHAP GradientExplainer per-modality attribution with GC-box-anchored windows; summary/heatmap/line/force plots.
- BED export of correct vs confused predictions for genomic follow-up.
- EDA of DNAshape distributions/correlations and peak QC.

---

## 8. Complete Inventory of Problems Solved
*(Explicit, exhaustive list as requested.)*

1. **DNABERT-2 Triton/flash-attention incompatibility** on Kaggle T4 → replaced with a pure-PyTorch attention monkey-patch.
2. **ALiBi meta-device allocation failure** on load → three-strategy loader (direct → manual state-dict → `torch.empty` patch) that materializes buffers on CPU.
3. **BatchNorm instability under DDP** (small per-GPU batches) → GroupNorm throughout convolutional branches.
4. **DDP synchronization hangs** with conditionally-used branches → `find_unused_parameters=True`.
5. **Shape branch collapse / modality imbalance** (1,536-d BERT swamping 128-d shape) → projection bottleneck, asymmetric dropout (heavy on BERT, light on shape), and differential learning rates.
6. **Loss of positional information for attention** → removed global average pooling (Spatial Shape CNN) and deferred pooling until after cross-modal attention (late fusion).
7. **Catastrophic collapse from bundling many untested changes** (≈25% random) → disciplined single-flag incremental ablation ("safe" variants).
8. **KAN over-regularization** (53%) → reverted to a simpler, more stable multi-scale CNN.
9. **NaN DNAshape values at window edges** → edge handling and zero-fill after robust scaling; RC fallback for missing pentamers.
10. **Data leakage between original and reverse-complement sequences** → GroupShuffleSplit keeping pairs in the same fold.
11. **Trivially separable negatives** → motif-destroying, GC/CpG-preserving dinucleotide shuffling so composition cannot be used as a shortcut.
12. **Positive:negative imbalance (3:1)** after merging paralogs → dynamic `pos_weight` in BCE.
13. **Per-target imbalance (SP1 > SP4 > SP2)** → q-value-ranked down-sampling to a common baseline.
14. **Cross-target peak overlap** (one locus bound by multiple SPs) → exclusive assignment to the highest-q-value target.
15. **FASTA case/`chr`-prefix inconsistencies and out-of-bounds windows** → upper-casing, prefix auto-detection, boundary filtering.
16. **Memory/throughput limits on T4** → dynamic sequence length (p99), bf16, fp16 embedding cache, gradient accumulation (effective batch 32–64).
17. **Severe overfitting of frozen-embedding models** (train ≈100% / val ≈40%) → backbone fine-tuning, dropout, label smoothing, weight decay, early stopping with a 30% overfitting-gap cap.
18. **The 3-class accuracy plateau (paralog confusion)** → diagnosed with a binary sanity check (pipeline validated), then resolved by a biologically justified binary reformulation.
19. **Sequence-only over-prediction of binding** (Negative recall 0.324) → multimodal shape+bio fusion raising Negative recall to 0.886.
20. **Verification of sequence↔shape array alignment** (`scratch/verify_alignment.py`, `debug_shapes.py`) → confirmed 1:1 FASTA↔NPY correspondence, eliminating off-by-one bugs.

---

## 9. References (indicative)
1. Zhou, J., et al. *DNABERT-2: Efficient and Effective Foundation Model for DNA Language.* (2023/2024).
2. Chiu, T. P., et al. *DNAshapeR: an R/Bioconductor package for genome-wide DNA shape prediction.* Bioinformatics.
3. Wang, K., et al. *BERT-TFBS: a novel BERT-based model for predicting transcription factor binding sites by transfer learning.* Briefings in Bioinformatics.
4. Dutta, P., Ghosh, N., & Santoni, D. *A DNABERT-based deep-learning framework for predicting transcription factor binding sites* (MCBAM/MSCA).
5. Le, N. Q. K., et al. *A transformer architecture based on BERT and 2-D CNN to identify DNA enhancers.*
6. Alipanahi, B., et al. *Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning (DeepBind).* Nat. Biotechnol. (2015).
7. Altschul, S. F., & Erickson, B. W. *Significance of nucleotide sequence alignments: a method for random sequence permutation that preserves dinucleotide and codon usage.* Mol. Biol. Evol. (1985).
8. Liu, Z., et al. *KAN: Kolmogorov–Arnold Networks.* (2024).
9. Lundberg, S. M., & Lee, S.-I. *A Unified Approach to Interpreting Model Predictions (SHAP).* NeurIPS (2017).
10. Loshchilov, I., & Hutter, F. *Decoupled Weight Decay Regularization (AdamW).* ICLR (2019).
11. The ENCODE Project Consortium. *An integrated encyclopedia of DNA elements in the human genome.* Nature (2012). Datasets: ENCFF333SWC, ENCFF480YAW, ENCFF938KVY; blacklist ENCFF356LFX.
12. Wolf, T., et al. *Transformers / Accelerate.* HuggingFace.

---

## Note on Reported Numbers (for the author, not for submission)
Per the chosen policy, every quantitative result above is copied from a saved `classification_report.txt` in `figures/`. The repository README quotes a higher sequence-only DNABERT-2 accuracy (89.05%) and a classical-ML accuracy (78.50%) that are **not** backed by any saved report (the classical baselines exist only as PNG figures, and the saved BERT-only binary report shows 76.24%). The README's proposed-model figure (~96.4%) **is** corroborated by the saved binary report (96.35%). Before submission, reconcile the baseline numbers against the original training logs, run the classical baselines (`notebooks/27_*.py`) to emit a numeric report, and confirm which proposed-model checkpoint (`outputs_gcmab_binary`) is canonical.
