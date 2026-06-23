"""
LightGBM, XGBoost, SVR — 31-seed evaluation (seeds 0-30)
"""
import json, sys, os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
import lightgbm as lgb
from sklearn.svm import SVR
from xgboost import XGBRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))
from data_utils import load_data, split_data, get_rdkit_features

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
df = load_data()

# Load best params from Optuna
lgb_params_raw = optuna.load_study(study_name="lgbm_forward",
    storage=f"sqlite:///{RESULTS_DIR}/optuna_studies.db").best_params
lgb_p = {k: v for k, v in lgb_params_raw.items()}
if "lr" in lgb_p: lgb_p["learning_rate"] = lgb_p.pop("lr")
lgb_p["n_estimators"] = min(lgb_p.get("n_estimators", 500), 300)

xgb_params = optuna.load_study(study_name="xgboost_forward",
    storage=f"sqlite:///{RESULTS_DIR}/optuna_studies.db").best_params

all_results = {"LightGBM": [], "XGBoost": [], "SVR": []}

for seed in range(31):
    train_df, val_df, test_df = split_data(df, seed=seed)
    X_train = get_rdkit_features(train_df["canonical_smiles"].tolist())
    X_test  = get_rdkit_features(test_df["canonical_smiles"].tolist())
    y_train = train_df["MMGBSA dG Bind"].values
    y_test  = test_df["MMGBSA dG Bind"].values

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    # LightGBM
    lgb_model = lgb.LGBMRegressor(**lgb_p, random_state=seed, verbosity=-1, n_jobs=-1)
    lgb_model.fit(X_train, y_train)
    p = lgb_model.predict(X_test)
    all_results["LightGBM"].append({
        "seed": seed,
        "RMSE": float(np.sqrt(mean_squared_error(y_test, p))),
        "R2": float(r2_score(y_test, p)),
        "Spearman": float(spearmanr(y_test, p)[0])
    })

    # XGBoost
    xgb_model = XGBRegressor(**xgb_params, random_state=seed, verbosity=0, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    p = xgb_model.predict(X_test)
    all_results["XGBoost"].append({
        "seed": seed,
        "RMSE": float(np.sqrt(mean_squared_error(y_test, p))),
        "R2": float(r2_score(y_test, p)),
        "Spearman": float(spearmanr(y_test, p)[0])
    })

    # SVR
    svr = SVR(kernel="rbf", C=10.0, epsilon=0.1)
    svr.fit(X_tr_sc, y_train)
    p = svr.predict(X_te_sc)
    all_results["SVR"].append({
        "seed": seed,
        "RMSE": float(np.sqrt(mean_squared_error(y_test, p))),
        "R2": float(r2_score(y_test, p)),
        "Spearman": float(spearmanr(y_test, p)[0])
    })

    print(f"Seed {seed:2d}: "
          f"LGB RMSE={all_results['LightGBM'][-1]['RMSE']:.3f}  "
          f"XGB RMSE={all_results['XGBoost'][-1]['RMSE']:.3f}  "
          f"SVR RMSE={all_results['SVR'][-1]['RMSE']:.3f}")

out = {}
for model, runs in all_results.items():
    mean = {k: float(np.mean([r[k] for r in runs])) for k in ["RMSE","R2","Spearman"]}
    std  = {k: float(np.std( [r[k] for r in runs])) for k in ["RMSE","R2","Spearman"]}
    out[model] = {"mean": mean, "std": std, "seeds": runs}
    print(f"{model}: RMSE={mean['RMSE']:.4f}±{std['RMSE']:.4f}  "
          f"R²={mean['R2']:.4f}±{std['R2']:.4f}  ρ={mean['Spearman']:.4f}±{std['Spearman']:.4f}")

with open(f"{RESULTS_DIR}/ml_result.json", "w") as f:
    json.dump(out, f, indent=2)
print("저장: results/ml_result.json")
