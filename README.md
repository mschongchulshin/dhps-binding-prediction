# DHPS MMGBSA Binding Prediction

Code and data for "AI-guided De Novo Design of DHPS Inhibitors" (Journal of Cheminformatics, 2026).

## Dataset

`data/dataset.csv` — 1,920 DHPS fragment compounds with MM-GBSA binding energies (kcal/mol), docked using Glide XP (Schrödinger Suite 2025-4) against human DHPS (PDB: 6PGR).

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

**Forward prediction (4 models, 31-seed cross-validation):**
```bash
cd scripts
python run_ridge.py
python run_ml.py        # LightGBM, XGBoost, SVR
python run_gnn.py       # AttentiveFP
python run_bilstm.py    # BiLSTM
```

**Reverse design (5 methods × 5 seeds):**
```bash
cd scripts
python run_reverse.py
```

Results are saved to `results/`.

## Citation

Shin et al., *AI-guided De Novo Design of DHPS Inhibitors*, Journal of Cheminformatics, 2026.
