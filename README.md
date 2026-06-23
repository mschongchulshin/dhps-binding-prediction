# AI-guided De Novo Design of DHPS Inhibitors

This repository provides the code and dataset for benchmarking AI-based binding affinity prediction and de novo molecular generation targeting Deoxyhypusine Synthase (DHPS).

Hypusine modification of eIF5A by DHPS is essential for the replication of a broad range of viruses, making DHPS an attractive host-directed therapy (HDT) target for antiviral drug discovery. We present an integrated AI benchmarking pipeline evaluating 13 MM-GBSA binding energy prediction models and 10 de novo molecular generation methods under identical conditions. For the prediction task, we propose an Out-of-Fold stacking ensemble integrating gradient boosting, graph neural networks, and chemical language models, achieving a Spearman rank correlation of 0.861. For the generation task, we propose RL-Design, a reinforcement learning-based framework using the ensemble predictor as a scoring oracle, which achieved a hit rate of 92.02% while maintaining drug-like physicochemical properties. Experimental DHPS inhibitory activity assays confirmed complete concordance between predicted and experimental activity categories across all six tested compounds.

## Dataset

`data/dataset.csv` — 1,920 DHPS fragment compounds with MM-GBSA binding energies (kcal/mol), docked using Glide XP (Schrödinger Suite 2025-4) against human DHPS (PDB: 6PGR).

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

**Forward prediction (31-seed cross-validation):**
```bash
cd scripts
python run_ridge.py        # Ridge regression
python run_ml.py           # LightGBM, XGBoost, SVR
python run_gnn.py          # AttentiveFP (Graph Neural Network)
python run_bilstm.py       # BiLSTM
```

**Reverse design (5 methods × 5 seeds):**
```bash
cd scripts
python run_reverse.py      # GA, BO, ScaffoldHop, Fragment, Retrieval
```

## Citation

Shin et al., *AI-guided De Novo Design of DHPS Inhibitors*, Journal of Cheminformatics, 2026. (under review)

Contact: saekomi5@korea.ac.kr

## Code Availability

<ins>**The full code for RL-Design and the OOF Stacking Ensemble is currently under patent application**</ins> and is not publicly available. The code will be made available upon reasonable request.
