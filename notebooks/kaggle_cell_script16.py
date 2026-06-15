"""
Auto-refactored script: kaggle_cell_script16.py
Refactored to align with project standards.
"""
# Script 16: Unified Cross-Modal Attention Dual-Branch (DNABERT-2 + DNAshape)
#
# DATASETS REQUIRED (Add as Kaggle Input):
#   1. lehotrongtin/datas1 → contains sp1/sp2/sp4_positive_final.fasta + negative_genomic_matched.fasta
#   2. lehotrongtin/dataset-shape → contains dnashape_sp1/sp2/sp4/negative.npy
#
# GPU: T4 16GB (recommended) or P100
# Runtime: ~30 - 50 min on T4 (much faster than 14/15: max_length 512 -> ~48)
#
# Keeps Script 14's proven core (6-layer unfreeze, 1 direct cross-attn, MLP-256, CE)
# and adds only capacity-neutral anti-overfit gains (auto max_length, layer-wise LR
# decay, dropout 0.4, EMA) plus low-risk wins (shape pos-embed, attn-pool, mild focal).
# All advanced/risky knobs (KAN head, DropPath, R-Drop, 2 layers) are OFF by default
# behind config flags for one-knob-at-a-time ablation.

!git clone https://github.com/JustinYuanZe/TF_Biding_Project.git
%cd TF_Biding_Project
!python notebooks/16_unified_crossmodal_kaggle.py
