#!/usr/bin/env python3
"""
GNN (AttentiveFP) 31-seed evaluation (seeds 0-30)
"""
import resource, json, sys, os
import numpy as np
import torch
import torch.nn as nn
import optuna
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGLoader
from torch_geometric.nn import AttentiveFP

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

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[GNN-31] Device: {DEVICE}")

# ── 노드/엣지 피처 상수 (run_gnn.py와 동일) ─────────────────────────────────
NODE_DIM = 39
EDGE_DIM = 10

def mol_to_graph(smi, y=0.0):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    mol = Chem.RemoveHs(mol)

    atom_features = []
    for atom in mol.GetAtoms():
        f = [
            atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
            int(atom.GetHybridization()), int(atom.GetIsAromatic()),
            atom.GetNumImplicitHs(), int(atom.IsInRing()),
            int(atom.IsInRingSize(3)), int(atom.IsInRingSize(4)),
            int(atom.IsInRingSize(5)), int(atom.IsInRingSize(6)),
        ]
        # one-hot atomic num (top 28 elements + other)
        common = [1,5,6,7,8,9,14,15,16,17,34,35,53]
        ohe = [int(atom.GetAtomicNum() == a) for a in common]
        f = ohe + [
            atom.GetDegree() / 6.0,
            atom.GetFormalCharge(),
            atom.GetNumImplicitHs() / 4.0,
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
        ]
        # pad/truncate to NODE_DIM
        f = f[:NODE_DIM] + [0.0] * max(0, NODE_DIM - len(f))
        atom_features.append(f)

    if not atom_features: return None
    x = torch.tensor(atom_features, dtype=torch.float)

    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond.GetBondTypeAsDouble()
        ring = int(bond.IsInRing())
        conj = int(bond.GetIsConjugated())
        stereo = int(bond.GetStereo())
        ef = [bt/3.0, ring, conj, stereo/5.0,
              int(bt==1), int(bt==1.5), int(bt==2), int(bt==3),
              int(ring), int(conj)]
        ef = ef[:EDGE_DIM] + [0.0]*max(0, EDGE_DIM - len(ef))
        for src, dst in [(i,j),(j,i)]:
            edge_index.append([src, dst])
            edge_attr.append(ef)

    if not edge_index:
        edge_index = torch.zeros((2,0), dtype=torch.long)
        edge_attr  = torch.zeros((0, EDGE_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_attr,  dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor([y], dtype=torch.float))


def make_graphs(df_part):
    graphs = []
    for smi, y in zip(df_part["canonical_smiles"].tolist(),
                      df_part["MMGBSA dG Bind"].tolist()):
        g = mol_to_graph(smi, y)
        if g is not None:
            graphs.append(g)
    return graphs


class GNNRegressor(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_timesteps, dropout):
        super().__init__()
        self.gnn = AttentiveFP(
            in_channels=NODE_DIM,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_dim=EDGE_DIM,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    def forward(self, data):
        x = self.gnn(data.x, data.edge_index, data.edge_attr, data.batch)
        return self.head(x).squeeze(-1)


@torch.no_grad()
def _eval(model, dl, device):
    model.eval()
    preds, labels = [], []
    for batch in dl:
        batch = batch.to(device)
        preds.extend(model(batch).cpu().numpy().tolist())
        labels.extend(batch.y.cpu().numpy().tolist())
    return np.array(preds), np.array(labels)


def train_and_eval(train_graphs, val_graphs, test_graphs, params, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    train_dl = PyGLoader(train_graphs, batch_size=params["batch_size"], shuffle=True)
    val_dl   = PyGLoader(val_graphs,   batch_size=128, shuffle=False)
    test_dl  = PyGLoader(test_graphs,  batch_size=128, shuffle=False)

    model = GNNRegressor(params["hidden_dim"], params["num_layers"],
                         params["num_timesteps"], params["dropout"]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=1e-4)
    from transformers import get_cosine_schedule_with_warmup
    total_steps = len(train_dl) * 100
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps*0.05), total_steps)
    criterion = nn.MSELoss()

    best_rmse, best_state, patience_cnt = float("inf"), None, 0
    PATIENCE = 20

    for epoch in range(100):
        model.train()
        for batch in train_dl:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
        preds_v, labels_v = _eval(model, val_dl, DEVICE)
        rmse = float(np.sqrt(mean_squared_error(labels_v, preds_v)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE: break

    model.load_state_dict(best_state)
    preds_t, labels_t = _eval(model, test_dl, DEVICE)
    rmse = float(np.sqrt(mean_squared_error(labels_t, preds_t)))
    r2   = float(r2_score(labels_t, preds_t))
    rho  = float(spearmanr(labels_t, preds_t)[0])
    pcc  = float(pearsonr(labels_t, preds_t)[0])
    mae  = float(mean_absolute_error(labels_t, preds_t))
    return {"seed": seed, "RMSE": rmse, "MAE": mae, "R2": r2, "PCC": pcc, "Spearman": rho}


# ── 메인 ─────────────────────────────────────────────────────────────────────
print("[GNN-31] 데이터 로드 및 best params 로드...")
df = load_data()
df["mol_id"] = df.index

study = optuna.load_study(study_name="gnn_attentivefp_forward",
                          storage=f"sqlite:///{RESULTS_DIR}/optuna_studies.db")
params = study.best_params
print(f"[GNN-31] Best params: {params}")

results = []
for seed in range(31):
    print(f"\n[GNN-31] Seed {seed:2d}...", flush=True)
    train_df, val_df, test_df = split_data(df, seed=seed)
    train_g = make_graphs(train_df)
    val_g   = make_graphs(val_df)
    test_g  = make_graphs(test_df)
    r = train_and_eval(train_g, val_g, test_g, params, seed)
    results.append(r)
    print(f"  RMSE={r['RMSE']:.4f}  R²={r['R2']:.4f}  ρ={r['Spearman']:.4f}", flush=True)

mean = {k: float(np.mean([r[k] for r in results])) for k in ["RMSE","MAE","R2","PCC","Spearman"]}
std  = {k: float(np.std( [r[k] for r in results])) for k in ["RMSE","MAE","R2","PCC","Spearman"]}
print(f"\n[GNN-31] RMSE={mean['RMSE']:.4f}±{std['RMSE']:.4f}  R²={mean['R2']:.4f}±{std['R2']:.4f}  ρ={mean['Spearman']:.4f}±{std['Spearman']:.4f}")

out = {"mean": mean, "std": std, "seeds": results}
with open(f"{RESULTS_DIR}/gnn_result.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"저장: {RESULTS_DIR}/gnn_result.json")
