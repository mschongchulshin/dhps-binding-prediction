#!/usr/bin/env python3
"""
BiLSTM 31-seed evaluation (seeds 0-30)
"""
import resource, json, sys, os
import numpy as np
import torch
import optuna
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# RAM 10GB 제한
try:
    MEM = 10 * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_RSS, (MEM, MEM))
except:
    pass

optuna.logging.set_verbosity(optuna.logging.WARNING)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import load_data, split_data
import model4_bilstm as m4

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[BiLSTM-31] Device: {DEVICE}")

print("[BiLSTM-31] 데이터 로드...")
df = load_data()
df["mol_id"] = df.index

aug_df = pd.read_pickle(f"{RESULTS_DIR}/augmented_full.pkl")

print("[BiLSTM-31] Best params 로드...")
study = optuna.load_study(study_name="bilstm_forward",
                          storage=f"sqlite:///{RESULTS_DIR}/optuna_studies.db")
bp = study.best_params
print(f"  {bp}")

results = []
for seed in range(31):
    print(f"\n[BiLSTM-31] Seed {seed:2d}...", flush=True)
    train_df, val_df, test_df = split_data(df, seed=seed)

    train_ids = set(train_df["mol_id"])
    val_ids   = set(val_df["mol_id"])
    test_ids  = set(test_df["mol_id"])

    train_aug  = aug_df[aug_df["mol_id"].isin(train_ids)].reset_index(drop=True)
    val_orig   = aug_df[aug_df["mol_id"].isin(val_ids)   & (~aug_df["is_augmented"])].reset_index(drop=True)
    test_orig  = aug_df[aug_df["mol_id"].isin(test_ids)  & (~aug_df["is_augmented"])].reset_index(drop=True)

    model, _ = m4.train_bilstm(
        train_aug, val_orig,
        embed_dim=bp["embed_dim"],
        hidden_dim=bp["hidden_dim"],
        n_layers=bp["n_layers"],
        dropout=bp["dropout"],
        lr=bp["lr"],
        batch_size=bp["batch_size"],
        epochs=10,
        device=DEVICE, seed=seed,
    )

    preds  = m4.predict_forward_bilstm(model, test_orig["canonical_smiles"].tolist(), DEVICE)
    labels = test_orig["MMGBSA dG Bind"].values
    mask   = ~np.isnan(preds) & ~np.isnan(labels)
    preds, labels = preds[mask], labels[mask]

    rmse = float(np.sqrt(mean_squared_error(labels, preds)))
    r2   = float(r2_score(labels, preds))
    rho  = float(spearmanr(labels, preds)[0])
    pcc  = float(pearsonr(labels, preds)[0])
    mae  = float(mean_absolute_error(labels, preds))

    results.append({"seed": seed, "RMSE": rmse, "MAE": mae, "R2": r2, "PCC": pcc, "Spearman": rho})
    print(f"  RMSE={rmse:.4f}  R²={r2:.4f}  ρ={rho:.4f}", flush=True)

mean = {k: float(np.mean([r[k] for r in results])) for k in ["RMSE","MAE","R2","PCC","Spearman"]}
std  = {k: float(np.std( [r[k] for r in results])) for k in ["RMSE","MAE","R2","PCC","Spearman"]}
print(f"\n[BiLSTM-31] RMSE={mean['RMSE']:.4f}±{std['RMSE']:.4f}  R²={mean['R2']:.4f}±{std['R2']:.4f}  ρ={mean['Spearman']:.4f}±{std['Spearman']:.4f}")

out = {"mean": mean, "std": std, "seeds": results}
with open(f"{RESULTS_DIR}/bilstm_result.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"저장: {RESULTS_DIR}/bilstm_result.json")
