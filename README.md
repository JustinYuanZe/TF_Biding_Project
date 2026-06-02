# SP1/SP2/SP4 Transcription Factor Binding Site Prediction

Multi-class TF binding site prediction for SP1, SP2, and SP4 (HepG2, ENCODE ChIP-seq, hg38).

## Data Processing Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 1 | `src/filter_and_eda.py` | Filter peaks by q-value ≤ 0.01, center 101bp on summit, remove cross-class overlaps |
| 2 | `src/downsample.py` | Balance classes to SP2 size (1,696 peaks each), prioritizing highest q-value |
| 3 | `src/extract_fasta.py` | Extract 101bp DNA sequences from hg38 reference genome |

## Quick Start (Google Colab)

```bash
# Mount Drive, clone repo, then:
python src/extract_fasta.py --fasta /content/drive/MyDrive/ML_Project/data/hg38.fa
```

Requires `hg38.fa` and `hg38.fa.fai` on Google Drive. No external dependencies needed.

## Project Structure

```
├── data/
│   ├── raw/                          # ENCODE narrowPeak files (gitignored)
│   └── processed/
│       ├── filtered_qval2/           # After q-value filtering
│       ├── filtered_centered_101bp/  # After 101bp centering
│       ├── filtered_exclusive_101bp/ # After cross-class overlap removal
│       └── downsampled_101bp/        # Final balanced BED + FASTA
├── figures/                          # EDA plots
├── notebooks/                        # Jupyter notebooks
├── ref/References/                   # Reference papers
└── src/                              # Pipeline scripts
```

## Data Sources

- **SP1**: ENCSR460YAM → ENCFF333SWC
- **SP2**: ENCSR946RZN → ENCFF480YAW
- **SP4**: ENCSR642PQK → ENCFF938KVY
