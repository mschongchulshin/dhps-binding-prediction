"""
Ridge 31-seed evaluation (seeds 0-30)
"""
import json, sys, os
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))
from data_utils import load_data, split_data, get_rdkit_features

df = load_data()
results = []

for seed in range(31):
    train_df, val_df, test_df = split_data(df, seed=seed)
    X_train = get_rdkit_features(train_df["canonical_smiles"].tolist())
    X_test  = get_rdkit_features(test_df["canonical_smiles"].tolist())
    y_train = train_df["MMGBSA dG Bind"].values
    y_test  = test_df["MMGBSA dG Bind"].values

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = Ridge(alpha=3708.369222848137)
    model.fit(X_train_sc, y_train)
    preds = model.predict(X_test_sc)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2   = float(r2_score(y_test, preds))
    sp   = float(spearmanr(y_test, preds)[0])
    mae  = float(np.mean(np.abs(y_test - preds)))
    pcc  = float(pearsonr(y_test, preds)[0])

    results.append({"seed": seed, "RMSE": rmse, "R2": r2, "Spearman": sp, "MAE": mae, "PCC": pcc})
    print(f"Seed {seed:2d}: RMSE={rmse:.4f}  R²={r2:.4f}  ρ={sp:.4f}")

mean = {k: float(np.mean([r[k] for r in results])) for k in ["RMSE","R2","Spearman","MAE","PCC"]}
std  = {k: float(np.std( [r[k] for r in results])) for k in ["RMSE","R2","Spearman","MAE","PCC"]}
print(f"\nMean RMSE={mean['RMSE']:.4f}±{std['RMSE']:.4f}")
print(f"Mean R²  ={mean['R2']:.4f}±{std['R2']:.4f}")
print(f"Mean ρ   ={mean['Spearman']:.4f}±{std['Spearman']:.4f}")

out = {"mean": mean, "std": std, "seeds": results}
import os as _os; _os.makedirs(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results"), exist_ok=True)
with open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results", "ridge_result.json"), "w") as f:
    json.dump(out, f, indent=2)
print("저장: results/ridge_result.json")
