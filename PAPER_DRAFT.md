# Multimodal Deep Fusion of Genomic Sequence, DNA Shape, and Biochemical Context for Predicting SP-Family Transcription-Factor Binding Sites

*Manuscript draft — IMRaD format. All quantitative results reported here are taken directly from the saved evaluation artifacts in the repository (the `classification_report.txt` files under `figures/`). Where the project's README quotes figures that are not backed by a saved report, those figures are deliberately omitted; see the "Note on Reported Numbers" at the end.*

---

## Abstract
Accurately mapping the binding sites of transcription factors is an important approach to understanding the mechanism of gene regulation. Transcription factors of the SP family (SP1, SP2, SP4) play a central role in cell-cycle and metabolic processes, yet are challenging to fully predict due to their paralog origin. We implemented a multimodal deep-learning framework to predict SP-family binding sites from 101-bp DNA windows by using a fusion of 3 complementary methods: semantic sequence from a pre-trained genomic transformer DNABERT-2, local 3-dimensional geometry from DNAShapeR, and biochemical information (CpG O/E ratio, GC content, and G-quadruplex motifs). The branches of DNABERT-2 and DNAShapeR are coordinated through a bidirectional cross-modal attention mechanism and a multi-scale depthwise-separable CNN (msCNN), with DDP-safe Group Normalization—together forming the core G-CMAB module (GroupNorm + Cross-Modal Attention + multi-scale Branch). The branches are then concatenated before a sigmoid classification head. We initially attempted to classify the three datasets individually but found that the performance remained plateau at approximately 61% accuracy. This is due to their biological nature—they share a nearly identical GC-box motif and paralogous origin—making differentiation within a 101bp window unreliable as it doesn't adequately reflect these biological characteristics. A diagnostic binary sanity check was applied to verify the pipeline's accuracy and motivated a reformulation into binding-vs-non-binding classification. The proposed model achieved 96.35% accuracy against dinucleotide-shuffled negatives (about 80% on real GC-matched genomic negatives), while the sequence-only DNABERT-2 model scored 76.24%. A SHAP analysis module was also used to extract information about sequence, shape, and biochemical features. A correct/incorrect detection was also implemented for validation.
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
We implemented **dinucleotide shuffling** (Altschul–Erickson Eulerian-path algorithm, `src/generate_negatives.py`) to produce negative sequences that only have mononucleotide and dinucleotide frequencies preserved, which means the GC and CpG density are preserved as well, while destroying binding motifs signal. This makes negatives non-trivial so that they cannot be separated easily from base composition, forcing the model to learn biological motifs rather than being like a GC content classifier. A supplementary generator (`src/generate_negatives_v2.py`) implements GC-matched random genomic negatives (with ENCODE blacklist ENCFF356LFX exclusion, ±500 bp peak padding, GC binning, and Cohen's-d quality control) and CpG-island negatives (UCSC `cpgIslandExt`) as alternatives.


### 3.5 Augmentation, balancing, and splitting
The datasets after being filtered and balanced were significantly reduced in number, so we enriched the datasets using **reverse-complement (RC) augmentation**, a technique that will create double the number of positive sequences by taking their complements, helping the model to learn motifs regardless of the direction of the sequence. For the negative set, this method is not applicable which has been created by shuffling, resulting in it not being easily distinguishable by conventional composition. After balancing was performed, we had a final dataset of 13,568 sequences, divided into 80% train and 20% test with the GroupShuffleSplit method to ensure that each original sequence and its complement were in the same fold, avoiding data leakage.

### 3.6 Feature engineering — three modalities
**(a) Sequence.** Windows were tokenized with the DNABERT-2 byte-pair-encoding (BPE) vocabulary; the maximum token length was set dynamically to the training-set p99 (clamped to [32, 96]) to save memory. **(b) DNAshape.** Five structural parameters — Minor Groove Width (MGW), Propeller Twist (ProT), Roll, Helical Twist (HelT), and Electrostatic Potential (EP) — were computed per position via a sliding pentamer (5-mer) lookup table (`src/dnashape_lookup.py`, `src/extract_dnashape.py`), with reverse-complement fallback for missing pentamers and NaN handling at window edges, producing an [N, 5, 101] tensor. **(c) Bio-features.** Three biochemical descriptors were computed per window: the CpG observed/expected ratio (promoter marker), GC content (stability marker), and presence of G-quadruplex (G4) motifs detected by regular expression.

### 3.7 Classical machine-learning baselines
Baseline is conducted with classic models such as Gradient Boosting, Extra Trees, Random Forest, AdaBoost, SVM(Linear),Logistic Regression,Naive Bayes,KNN, Decision Tree and MLP with one-hot encoding and k-mer TF-IDF(n-gram range 4-6). Logistic achieved the best results with 80.73% accuracy, ROC-AUC 0.818 and Negative recall 0.376. Gradient Boosting reaches 75.53% accuracy but Negative recall is low (0.043). The detailed results are stored in `outputs_binary_classical_ml/{onehot,tfidf}_report.txt` and key figures are presented in Table 2.

### 3.8 Sequence-only DNABERT-2 baseline
The sequence-only baseline (`notebooks/24_binary_dnabert2_finetune_kaggle.py`) fine-tunes DNABERT-2-117M with the last 3 of its 12 encoder layers unfrozen. Token embeddings are pooled by concatenating the [CLS] vector with the masked mean, giving a [B,1536] representation. This feeds an MLP head: LayerNorm, then Linear(1536→256), GELU, Dropout(0.3), and Linear(256→1). Training uses BCEWithLogitsLoss with positive-class weight n_neg/n_pos, AdamW (backbone 1.5e-5, head 1e-4), and a linear-warmup-then-cosine schedule.

### 3.9 Proposed multimodal architecture (Tri-Branch / G-CMAB)
The proposed model (`notebooks/23_binary_tribranch_kaggle.py`) has three parallel branches feeding a fusion head. **Sequence branch:** DNABERT-2 with the last 6 layers unfrozen, an ELMo-style learned scalar-mix ("layer attention") over the last six hidden states, projected to a 128-d token sequence. **Shape branch:** a 1×1 Conv projection of the [B,5,101] tensor to 128 channels with GroupNorm + GELU. **Bio branch:** a small MLP (3→16→32) over the biochemical features. The 1-D sequence tokens and the spatial shape map are coupled by a **bidirectional cross-modal multi-head attention** module (d_model 128, 4 heads, 1 layer, with residual + LayerNorm + feed-forward): sequence queries shape and shape queries sequence, directly aligning nucleotide context to helix geometry.

### 3.10 Multi-scale CNN and fusion head
After cross-modal interaction, both the attended sequence and shape streams pass through a **multi-scale depthwise-separable CNN (msCNN)**: parallel kernels of sizes {7,9,11,15} (sequence) and {4,8,12,16} (shape), each with GroupNorm, GELU, and 1-max pooling, capturing motifs of multiple widths while keeping parameters low. The pooled sequence (1,024-d), pooled shape (1,024-d), and bio (32-d) vectors are concatenated (2,080-d) and classified by Linear(2080→256) → GELU → Dropout(0.7) → Linear(256→1) with a sigmoid output. The umbrella name **G-CMAB** denotes this combination of **G**roupNorm + **C**ross-**M**odal **A**ttention + multi-scale **B**ranch.



### 3.11 Training and optimization
All deep models use **BCEWithLogitsLoss** with a dynamically computed positive-class weight (n_neg/n_pos ≈ 0.33 for the merged binary task) to counteract the 3:1 positive:negative imbalance. Optimization uses **AdamW** with **branch-specific learning rates** (backbone 2e-5; sequence/shape projections and cross-attention 2e-4–3e-4; layer-attention 1e-3), weight decay 0.1, linear warmup (≈15% of steps) followed by cosine annealing, and gradient clipping (max-norm 0.5–1.0). Training ran up to 30 epochs (batch 16, gradient accumulation giving an effective batch of 32–64) with early stopping (patience 5–12) capped by a 30% train/val overfitting-gap guard, saving the best-validation checkpoint.

### 3.12 Distributed-training and systems engineering
Training used HuggingFace **Accelerate** for Distributed Data Parallel (DDP) on 2× NVIDIA Tesla T4 GPUs (Kaggle) with **bf16** mixed precision, `find_unused_parameters=True`, cross-process metric gathering, and synchronized checkpointing. **GroupNorm replaces BatchNorm** throughout the convolutional branches because BatchNorm statistics are unstable for the small per-GPU batches under DDP. Because DNABERT-2's custom Triton flash-attention and ALiBi buffers fail to load on T4/Kaggle, the backbone is loaded through a pure-PyTorch attention monkey-patch and a three-strategy loader that avoids meta-device allocation (`src/dnabert_wrapper.py`).

### 3.13 Task reformulation: 3-class → binary
Prediction was first framed as 4-way classification (SP1 / SP2 / SP4 / Negative). It plateaued near 61% (Section 4.2), so we ran a binary sanity check (`notebooks/13_binary_sanity_check_kaggle.py`) that merged all positives into one class to test the pipeline. Its accuracy was reasonable, which told us the difficulty was inter-paralog confusion rather than a code defect. We then reformulated the task as binding (SP1∪SP2∪SP4) vs non-binding. This is biologically reasonable: the three paralogs recognize an almost identical GC-box and have overlapping DNAshape distributions (Section 4.1).

### 3.14 Evaluation and interpretability
Models are evaluated with accuracy, per-class precision/recall/F1, macro/weighted averages, ROC-AUC, PR-AUC, and confusion matrices, with training, ROC, and PR curves saved per experiment. Interpretability uses **SHAP GradientExplainer** (`notebooks/explain_shap_tribranch.py`; ~50 background, ~20 explained samples) to attribute logits across the three modalities — sequence tokens/k-mers, the five DNAshape parameters, and bio-features — aligned to the GC-box center over configurable flanking windows, and rendered as summary-bar, heatmap, line, and force plots. Correctly and incorrectly classified loci are exported as genomic BED tracks (`True_SP1.bed`, `True_SP4.bed`, `Confused_SP4_as_SP1.bed`) for downstream motif-enrichment and chromatin follow-up.

---

## 4. Results

### 4.1 Exploratory data analysis
EDA of the five DNAshape parameters on the merged corpus (`scratch/dnashape_eda_report.md`) showed near-symmetric MGW and HelT but strongly skewed Roll and EP, with the strongest pairwise coupling between ProT and EP (r = 0.77) and between MGW and Roll (r = 0.54). The position-wise shape distributions and GC content are very similar across SP1, SP2, and SP4 (`figures/dnashape_boxplots.png`), which fits their shared GC-box and supports the binary grouping. Diagnostic analysis (`src/diagnose_pipeline.py`) adds two points that explain why the task is hard: the discriminative core motif occupies only about 8–10 bp of the 101 bp window, and the dinucleotide-shuffled negatives are GC-matched to the positives.

### 4.2 The multiclass plateau and paralog confusion
Every 4-way model saturated between roughly 50% and 62% accuracy (see Table S1 in the Supplement). The error pattern was the same in all of them: the Negative class was recovered well (recall around 0.86–0.88, precision around 0.92–0.95 in the better models), but the three SP paralogs were confused with one another, with SP4 recall falling as low as 0.33–0.57. A 101 bp window is therefore enough to answer "is this a binding site?" but not "which SP paralog?", since the paralogs share both motif and shape signature. We read this as a property of the data rather than a pipeline bug, and it is what motivated the binary reformulation.

### 4.3 Binary classification (principal result)
Under the binary formulation, and evaluated against dinucleotide-shuffled negatives, the multimodal G-CMAB model reaches 96.35% accuracy and macro-F1 0.9501 (Table 2), against 76.24% for the fine-tuned sequence-only DNABERT-2. Most of the gain is in negative rejection. Negative recall rises from 0.324 in the sequence-only model to 0.886, and Negative precision from 0.547 to 0.966; SP-Positive recall reaches 0.990. The pattern holds across every sequence-only method we tried: the fine-tuned transformer (76.2%) and the best classical k-mer model (80.7%) both stall at a Negative recall of 0.32–0.38, while the full multimodal model reaches 0.886. Section 4.4 and Section 5.2 work through which components account for this difference, and how much of the 96.35% survives a switch from shuffled to real genomic negatives; the answer to both is less tidy than "shape plus attention".

**Table 2 — Binary classification (held-out test). Deep models use the group split (n = 2,710: 682 Neg / 2,028 Pos); classical baselines (†) use a stratified split (n = 2,714: 679 / 2,035) — both ~20 %, ~3:1. Source: saved `classification_report.txt` / `{onehot,tfidf}_report.txt`.**

| Model (script) | Accuracy | Macro-F1 | Neg P | Neg R | Pos P | Pos R |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Classical one-hot · Gradient Boosting (27)† | 0.7553 | 0.4696 | 0.674 | 0.043 | 0.757 | 0.993 |
| Sequence-only DNABERT-2 (24) | 0.7624 | 0.6292 | 0.547 | 0.324 | 0.800 | 0.910 |
| Classical TF-IDF · Logistic Regression (27)† | 0.8073 | 0.6874 | 0.720 | 0.376 | 0.820 | 0.951 |
| **Proposed multimodal G-CMAB (23/28)** | **0.9635** | **0.9501** | **0.966** | **0.886** | **0.963** | **0.990** |

### 4.4 Controlled ablation
The 27-script development history (Section 5.3) was an informal architecture search, so it cannot attribute performance to any single component. To do that we ran a controlled ablation: one configurable model trained under a fixed data split, seed, and budget (15 epochs), with a single component toggled at a time (`notebooks/29_binary_ablation_kaggle.py`, output `figures/outputs_ablation_no_ckpt/ablation_results.csv`). Table 3 reports the results. Absolute accuracies sit below the 96.35% headline because the ablation runs 15 epochs rather than 30; the relative differences are what matter here.

**Table 3 — Controlled binary ablation (same split/seed, 15 epochs, n = 2,710). A1→A4 add components incrementally; A5–A7 knock out one component from the full model A4.**

| Variant | Configuration | Acc. | ROC-AUC | PR-AUC | Neg R | Pos R |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| A1 | Sequence only | 0.7446 | 0.6419 | 0.832 | 0.274 | 0.903 |
| A2 | + DNAshape (concat) | 0.7697 | 0.6495 | 0.835 | 0.191 | 0.965 |
| A3 | + Cross-modal attention | 0.7771 | 0.7013 | 0.863 | 0.277 | 0.945 |
| **A4** | **+ Bio branch (full)** | **0.8956** | **0.8930** | **0.947** | **0.666** | 0.973 |
| A5 | A4 − msCNN (mean+max pool) | 0.8985 | 0.9009 | 0.952 | 0.680 | 0.972 |
| A6 | A4 − layer-attention | 0.8963 | 0.8992 | 0.950 | 0.679 | 0.969 |
| A7 | A4, BatchNorm (no GroupNorm) | 0.8900 | 0.8956 | 0.953 | 0.606 | 0.986 |

Three things come out of this. First, cross-modal attention helps ranking more than it helps thresholded accuracy (A2→A3: ROC-AUC 0.650→0.701, accuracy +0.7 pp), which is the behaviour the bidirectional design was meant to produce. Second, the msCNN and the layer-attention add almost nothing: removing either (A5, A6) stays within 0.3 pp of the full model A4, so the architecture can be cut down considerably. GroupNorm, by contrast, gives a small but repeatable gain over BatchNorm (A4 vs A7: +0.6 pp, Negative recall 0.666 vs 0.606), which is the reason to keep it under DDP. Third, the largest single jump comes with the bio-feature branch (A3→A4: +11.9 pp accuracy, Negative recall 0.28→0.67). Section 5.2 shows that this jump is not explained by discriminative information in the bio features; it looks more like an optimization or regularization effect, and still needs multi-seed and real-negative checks. The informal search is also informative in the negative direction: bundling many untested changes at once dropped accuracy to near-random (script 16, about 25%), and a ChebyKAN classifier over-regularized to 53% (script 15). Both are arguments for changing one thing at a time.

### 4.5 Interpretability and genomic output
SHAP attributions split each prediction into sequence, DNAshape, and bio-feature contributions around the GC-box, saved as per-modality bar, heatmap, and line plots (`figures/SHAP/`, `figures/script21/`). Correct and confused predictions are also written out as BED tracks (`True_*.bed`, `False_*.bed`) for genome-browser inspection and later motif or chromatin analysis. One caveat is worth recording. When SHAP is run on confidently-correct examples, as happened in the ablation run, the sigmoid is saturated and the input gradients are near zero, so GradientExplainer attributions fall to about zero for every modality. Those flat plots are an artifact of the saturation, not evidence that the model ignores the inputs. Useful attributions have to come from samples near the decision boundary, and the interpretability routine was changed to select them in later runs.

---

## 5. Discussion

### 5.1 Principal findings
A 101 bp window holds enough information to decide whether an SP-family factor binds, but not which paralog binds, because SP1, SP2, and SP4 share an almost identical GC-box and overlapping shape profiles. Moving to binding-vs-non-binding and using the full multimodal model raises binary accuracy on the dinucleotide-shuffled benchmark from 76.2% (sequence-only DNABERT-2) and 80.7% (the best classical model, TF-IDF logistic regression) to 96.35%, with the main effect again being better negative rejection (Negative recall climbing from the 0.32–0.38 ceiling of the sequence-only methods to 0.886). Two qualifications follow in Section 5.2: which components are actually responsible, and how far the figure falls when the shuffled negatives are replaced by real genomic ones. Both make the story more careful than crediting shape or attention on their own.

### 5.2 What actually drives the gain — and an important caveat
The ablation complicates the intuitive "shape via attention" story. Cross-modal attention gives a real but modest ranking improvement (ROC-AUC 0.650→0.701), DNAshape adds a small accuracy gain, and the large jump in accuracy and specificity appears once the bio-feature branch is present (A3→A4). The problem is that the three bio features carry almost no discriminative information between positives and their dinucleotide-shuffled negatives. The Altschul–Erickson shuffle preserves mono- and dinucleotide counts exactly, so CpG O/E (0.699 for positives vs 0.694 for negatives) and GC content (0.651 vs 0.651) are nearly identical between the two classes, and G-quadruplex motifs are rare in both (4.8% vs 3.2%). The most likely reading, then, is that the bio branch helps as an optimization or regularization term rather than as a source of biological signal: with the branch attached, the network settles into a lower validation-loss basin within the same budget (about 0.20, against about 0.30 without it). This was not what we expected, and it means the original framing — that DNAshape via cross-modal attention is the primary driver — is not supported by the ablation.

A switch from shuffled to real genomic negatives confirms this caution and sharpens it. Re-evaluated against GC- and length-matched genomic loci (`notebooks/32_*`; data in `data/processed/FINAL`), the full model's accuracy falls from 96.35% to about 80%: 79.6% on the random group split, 77.9% on the dedicated real-negative run, and 79.4% ± 0.8% (ROC-AUC 0.825) for the chromosome-holdout, multi-seed robustness run (`notebooks/30_*`, chr1/6/7, seeds 42/1/2023). Almost all of the loss is in negative rejection: Negative recall drops to 0.27–0.54, and at the default operating point the multimodal model no longer clearly beats the classical TF-IDF logistic-regression baseline (80.7%). The genomic ablation reverses the component story as well — the bio branch, worth +11.9 pp on shuffles, now slightly hurts (A3→A4: 77.0%→75.4% accuracy, ROC-AUC 0.761→0.748), leaving DNAshape and cross-modal attention as the only components with a consistent positive effect (ROC-AUC 0.64→0.76 across A1–A3). A univariate test still finds CpG O/E weakly informative on genomic negatives (AUC 0.695, against about 0.50 on shuffles), but that weak signal does not carry over into the assembled model. We therefore treat 96.35% as a result on the dinucleotide-shuffled benchmark — a standard composition-matched control — while stating plainly that it is partly an evaluation artifact: the project's own real-negative check (`outputs_real_negatives/artifact_check.txt`) flags the 18-point gap. The one reassurance is that the chromosome-holdout figure (79.4%) matches the random-split genomic figure (79.6%), so the genomic number is leakage-stable rather than a further artifact.

### 5.3 Problems solved (engineering and methodological)
The project resolved a broad set of concrete issues, summarized in Section 7. These span distributed-training stability (BatchNorm→GroupNorm; `find_unused_parameters`; bf16), backbone compatibility (Triton-free attention and ALiBi meta-device loading), modality imbalance and branch collapse (projection bottleneck, asymmetric dropout, spatial pooling, late fusion), numerical/edge issues (NaN shape values, gradient clipping), data integrity (leakage-free grouped splitting, motif-preserving negatives, cross-target overlap removal, FASTA case/prefix handling), memory/throughput on T4 (dynamic sequence length, fp16 embedding cache, gradient accumulation), and overfitting control (fine-tuning, dropout, label smoothing, early stopping with an overfitting-gap guard). The most consequential methodological fix was diagnosing the multiclass plateau and reformulating the task to binary.

### 5.4 Limitations
Several limitations bound the present claims. First, the data come from a single cell line (HepG2) and a single TF family (the three SP paralogs), so generalization to other cell types and families is untested. Second, the headline 96.35% is measured against dinucleotide-shuffled negatives; on real GC-matched genomic negatives the same model reaches only about 80% (Section 5.2), so the dinuc figure should be read as a composition-matched control rather than as expected genome-wide performance. Third, the headline split is a random group split that keeps each sequence with its reverse complement, which removes augmentation leakage but not genomic-proximity or homology leakage; the chromosome-holdout, multi-seed run that controls for this gives 79.4% ± 0.8% on genomic negatives, essentially matching the random-split genomic number (79.6%), so the result is leakage-stable. Fourth, the controlled ablation uses a single seed at a 15-epoch budget; the genomic ablation has run through A6 but not A7, and the no-bio convergence control (`notebooks/31_*`) has so far completed only the sequence-only arm. Fifth, paralog identification is still unsolved at 101 bp resolution.

### 5.5 Performance ceiling
Against dinucleotide-shuffled negatives binary accuracy levels off near 96%; against real genomic negatives it sits near 80% (Section 5.2). Either way the ceiling is consistent with a 101 bp window holding only so much of what the task needs once sequence, shape, and biochemical context are all used. Pushing past it will probably need longer context, a chromatin or accessibility signal such as ATAC-seq, or quantitative binding affinities in place of binary peak calls.

---

## 6. Conclusion
We built a multimodal tri-branch model (G-CMAB) that predicts SP-family TF binding by fusing DNABERT-2 sequence embeddings, DNAshape geometry, and biochemical features through bidirectional cross-modal attention and a multi-scale CNN. After moving from per-paralog to binding-vs-non-binding classification — a change the data forced rather than one we chose freely — the model reaches 96.35% accuracy against dinucleotide-shuffled negatives, beating a sequence-only transformer mainly by rejecting those negatives. On real genomic negatives the figure is closer to 80% (Section 5.2), and narrowing that gap is the main open problem. The model stays interpretable throughout via SHAP and BED-track export. The open directions are generalization to other TF families and cell types, adding chromatin-accessibility data, and returning to paralog discrimination with longer genomic context.

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

## 9. References
*(Formatted list; entries marked [VERIFY] could not be confidently attributed from a filename alone — see `PAPER_SUPPLEMENT.md` Part 2 for the full list, the unattributed `ref/` PDFs, and resolution notes.)*

1. Zhou, Z., Ji, Y., Li, W., Dutta, P., Davuluri, R., & Liu, H. (2023). *DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome.* arXiv:2306.15006 (ICLR 2024).
2. Chiu, T.-P., Comoglio, F., Zhou, T., Yang, L., Paro, R., & Rohs, R. (2016). *DNAshapeR: an R/Bioconductor package for DNA shape prediction and feature encoding.* Bioinformatics 32(8): 1211–1213.
3. Wang, K., Zeng, X., Zhou, J., et al. *BERT-TFBS: a novel BERT-based model for predicting transcription factor binding sites by transfer learning.* Briefings in Bioinformatics. [VERIFY year/volume]
4. Dutta, P., Ghosh, N., & Santoni, D. *A DNABERT-based deep-learning framework for transcription-factor binding-site prediction (multi-scale convolution + attention).* [VERIFY title/venue/year]
5. Le, N. Q. K., et al. *A transformer architecture based on BERT and a 2-D CNN to identify DNA enhancers.* Briefings in Bioinformatics. [VERIFY]
6. Alipanahi, B., Delong, A., Weirauch, M. T., & Frey, B. J. (2015). *Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning (DeepBind).* Nature Biotechnology 33(8): 831–838.
7. Altschul, S. F., & Erickson, B. W. (1985). *Significance of nucleotide sequence alignments: a method for random sequence permutation that preserves dinucleotide and codon usage.* Molecular Biology and Evolution 2(6): 526–538.
8. Liu, Z., Wang, Y., Vaidya, S., et al. (2024). *KAN: Kolmogorov–Arnold Networks.* arXiv:2404.19756.
9. Lundberg, S. M., & Lee, S.-I. (2017). *A unified approach to interpreting model predictions (SHAP).* NeurIPS 30: 4765–4774.
10. Loshchilov, I., & Hutter, F. (2019). *Decoupled weight decay regularization (AdamW).* ICLR 2019.
11. The ENCODE Project Consortium. (2012). *An integrated encyclopedia of DNA elements in the human genome.* Nature 489(7414): 57–74. Datasets: ENCFF333SWC, ENCFF480YAW, ENCFF938KVY; blacklist ENCFF356LFX.
12. Wolf, T., Debut, L., Sanh, V., et al. (2020). *Transformers: state-of-the-art natural language processing.* EMNLP 2020 (System Demonstrations): 38–45. (HuggingFace; *Accelerate* used for distributed training.)
13. Peters, M. E., Neumann, M., Iyyer, M., et al. (2018). *Deep contextualized word representations (ELMo).* NAACL-HLT 2018: 2227–2237. (Source of the scalar-mix / layer-attention.)
14. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention is all you need.* NeurIPS 30: 5998–6008. (Scaled-dot-product / multi-head attention.)

---

## Appendices (see `PAPER_SUPPLEMENT.md`)
The companion file `PAPER_SUPPLEMENT.md` contains, ready for integration: a metadata/keywords block and Data-and-Code-Availability statement; the complete formatted reference list with `[VERIFY]` resolution notes; a 20-entry **figure inventory with captions** mapped to IMRaD sections (all paths verified to exist); a consolidated **hyperparameter table** for the proposed model; and a **mathematical-formulation appendix** (attention, scalar-mix, depthwise-separable conv, GroupNorm, BCE-with-pos-weight, AdamW, cosine schedule, CpG O/E, TF-IDF, robust scaling, Shapley/GradientExplainer, and the evaluation metrics). Reproducibility files (`requirements.txt`, `REPRODUCE.md`, `DATA_AVAILABILITY.md`) and the controlled-ablation analysis (`ABLATION_ANALYSIS.md`) accompany the repository.

---

## Note on Reported Numbers (for the author, not for submission)
Per the chosen policy, every quantitative result above is copied from a saved report (`classification_report.txt`, `{onehot,tfidf}_report.txt`, or `ablation_results.csv`) actually present in the repository. Status of previously-disputed numbers: (1) **Classical baselines** have now been re-run — the best classical model is TF-IDF logistic regression at **80.73 %** (the README's "78.50 % SVM" was close but not the best; SVM-TF-IDF was 80.43 %). (2) The **sequence-only DNABERT-2** saved report shows **76.24 %** (the README's 89.05 % is not backed by any saved report and is not used). (3) The proposed-model **96.35 %** is corroborated by the saved binary report in `outputs_gcmab_binary` — whose model file (`best_gcmab_binary.pt`) and output naming match the **tri-branch** script 23 (the reproducible trainer is `notebooks/28_*`); the "Script 19" string inside that report is a stale hard-coded title, not the architecture. (4) The exclusive-peak counts were corrected to the on-disk values (6,815 / 1,696 / 8,914). (5) The genomic re-runs are now done (saved under `figures/rerun/`) and reshape the discussion: with the sequence↔shape pairing enforced and matched genomic DNAshape, the full model scores **79.6 %** (random split, `outputs_tribranch_shap`), **79.4 % ± 0.8 %** (chromosome-holdout multi-seed, `outputs_robustness`), and **77.9 %** (real-negative eval, `outputs_real_negatives`) — so the earlier 95.24 % robustness figure was an artifact of the shape mismatch and has been replaced by 79.4 %. The project's own `artifact_check.txt` labels the 18-point dinuc→genomic gap a likely artifact, and the genomic ablation shows the bio branch reversing sign; Section 5.2 and Section 5.4 now report this. Still outstanding before submission: re-save the sequence-only genomic report (`notebooks/24_*` wrote figures + checkpoint but no `classification_report.txt`); run ablation variant A7 and finish the no-bio convergence control (`notebooks/31_*`, only A1 done); confirm the classical re-run used the genomic negatives (its TF-IDF accuracy is unchanged at 80.73 %); and resolve the `[VERIFY]` citations in `PAPER_SUPPLEMENT.md`.
