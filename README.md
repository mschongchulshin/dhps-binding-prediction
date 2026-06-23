# DHPS MMGBSA Binding Prediction

Code and data for "AI-guided De Novo Design of DHPS Inhibitors" (Journal of Cheminformatics, 2026).

## Dataset

`dataset.csv` — 1,920 DHPS fragment compounds with MM-GBSA binding energies (kcal/mol), docked using Glide XP (Schrödinger Suite 2025-4) against human DHPS (PDB: 6PGR).

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

**Forward prediction (benchmarking 4 models over 31 seeds):**
```bash
python run_ridge_31seed.py
python run_ml_31seed.py       # LightGBM, XGBoost, SVR
python run_gnn_31seed.py      # AttentiveFP
python run_bilstm_31seed.py   # BiLSTM (requires results/augmented_full.pkl)
```

**Reverse design (5 methods × 5 seeds):**
```bash
python run_reverse_5method_5seed.py
```

Results are saved to `results/`.

## Citation

Shin et al., *AI-guided De Novo Design of DHPS Inhibitors*, Journal of Cheminformatics, 2026.
