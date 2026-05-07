\# 🧬 RESEARCH LOG: Multi-modal SP-Family Binding Prediction







\## 2026-05-08 | Data



\### Objective

Establish clean 3-class dataset (SP1/SP2/SP4), eliminate batch effects.



\### Done

* \[x] Dived into ENCODE database to find good candidates
* \[x] Set SP4 (`ENCSR642PQK`) as anchor (CRISPR, HepG2, NovaSeq 6000) as there is only this one quality dataset 
* \[x] Mapped SP1/SP2 to matching tech profile
* \[x] Excluded the old datasets to prevent Cell Line Bias(standardized entirely on HepG2) and maintain genetic background(CRISPR)
* \[x] Reconstruction project structure



\### Dataset

| Protein | ENCODE ID     | Cell  | Platform     |

|---------|---------------|-------|--------------|

| SP1     | `ENCSR460YAM` | HepG2 | NextSeq 500  |

| SP2     | `ENCSR946RZN` | HepG2 | NovaSeq 6000 |

| SP4     | `ENCSR642PQK` | HepG2 | NovaSeq 6000 |



\### Next

* \[ ] Sketch data preparation
* \[ ] Sketch for multi-modal mCNN modelling architecture.
* \[ ] Further searching for related works

