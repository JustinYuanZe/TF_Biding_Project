"""
Auto-refactored script: kaggle_cell_script14.py
Refactored to align with project standards.
"""
# Script 14: Cross-Modal Attention Dual-Branch (DNABERT-2 + DNAshape)
#
# DATASETS REQUIRED (Add as Kaggle Input):
#   1. lehotrongtin/datas1 → contains sp1/sp2/sp4_positive_final.fasta + negative_genomic_matched.fasta
#   2. lehotrongtin/dataset-shape → contains dnashape_sp1/sp2/sp4/negative.npy
#
# GPU: T4 16GB (recommended) or P100
# Runtime: ~60-90 min for 25 epochs on T4

!git clone https://github.com/JustinYuanZe/TF_Biding_Project.git
%cd TF_Biding_Project
!python notebooks/14_crossmodal_dual_kaggle.py
