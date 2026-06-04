# Script 15: KAN-Regularized Cross-Modal Attention Dual-Branch
#
# DATASETS REQUIRED (Add as Kaggle Input):
#   1. lehotrongtin/datas1 → contains sp1/sp2/sp4_positive_final.fasta + negative_genomic_matched.fasta
#   2. lehotrongtin/dataset-shape → contains dnashape_sp1/sp2/sp4/negative.npy
#
# GPU: T4 16GB (recommended) or P100
# Runtime: ~1.5 - 2.5 hours on T4 (due to R-Drop double forward pass)

!git clone https://github.com/JustinYuanZe/TF_Biding_Project.git
%cd TF_Biding_Project
!python notebooks/15_kan_crossmodal_kaggle.py
