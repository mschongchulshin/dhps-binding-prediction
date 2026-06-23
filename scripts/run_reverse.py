#!/usr/bin/env python3
"""
역설계 5-method 5-seed 평가
Methods: GA, BO, RNN, Fragment, SimilarityRetrieval
Seeds: 0-4
Targets: -30, -40
"""
import os, sys, json, warnings, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import load_data, split_data, get_rdkit_features
from rdkit import Chem, RDLogger, RDConfig
from rdkit.Chem import Descriptors, QED, AllChem, BRICS, RWMol, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

import os as _os, sys as _sys
_sys.path.append(_os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import lightgbm as lgb
import optuna

# GNN imports
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGLoader
from torch_geometric.nn import AttentiveFP

# BiLSTM imports
import model4_bilstm as m4

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
log_device = lambda: print(f"[5Method] Device: {DEVICE}", flush=True)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
TARGETS = [-30, -40]
SEEDS = [0, 1, 2, 3, 4]
N_GENERATE = 50
BASE_SEED = 42
METHODS = ["GA", "BO", "ScaffoldHop", "Fragment", "Retrieval"]

def log(msg): print(msg, flush=True)

# ── GNN 정의 ──────────────────────────────────────────────────────────────────
NODE_DIM = 39
EDGE_DIM = 10

def mol_to_graph(smi, y=0.0):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    # Take largest fragment for disconnected SMILES (e.g. Fragment method output)
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumAtoms())
    atom_features = []
    for atom in mol.GetAtoms():
        common = [1,5,6,7,8,9,14,15,16,17,34,35,53]
        ohe = [int(atom.GetAtomicNum() == a) for a in common]
        f = ohe + [atom.GetDegree()/6.0, atom.GetFormalCharge(),
                   atom.GetNumImplicitHs()/4.0, int(atom.GetIsAromatic()), int(atom.IsInRing())]
        f = f[:NODE_DIM] + [0.0]*max(0, NODE_DIM - len(f))
        atom_features.append(f)
    if not atom_features: return None
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond.GetBondTypeAsDouble()
        ring = int(bond.IsInRing()); conj = int(bond.GetIsConjugated()); stereo = int(bond.GetStereo())
        ef = [bt/3.0, ring, conj, stereo/5.0, int(bt==1), int(bt==1.5), int(bt==2), int(bt==3), int(ring), int(conj)]
        ef = ef[:EDGE_DIM] + [0.0]*max(0, EDGE_DIM - len(ef))
        for src, dst in [(i,j),(j,i)]:
            edge_index.append([src,dst]); edge_attr.append(ef)
    if not edge_index:
        edge_index = torch.zeros((2,0), dtype=torch.long)
        edge_attr  = torch.zeros((0,EDGE_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_attr,  dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor([y], dtype=torch.float))

class GNNRegressor(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_timesteps, dropout):
        super().__init__()
        self.gnn = AttentiveFP(in_channels=NODE_DIM, hidden_channels=hidden_dim,
                               out_channels=hidden_dim, edge_dim=EDGE_DIM,
                               num_layers=num_layers, num_timesteps=num_timesteps, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim//2),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim//2, 1))
    def forward(self, data):
        x = self.gnn(data.x, data.edge_index, data.edge_attr, data.batch)
        return self.head(x).squeeze(-1)

@torch.no_grad()
def _gnn_predict(model, smiles_list, device):
    model.eval()
    graphs = [mol_to_graph(s) for s in smiles_list]
    graphs = [g for g in graphs if g is not None]
    if not graphs: return np.array([])
    dl = PyGLoader(graphs, batch_size=128, shuffle=False)
    preds = []
    for batch in dl:
        batch = batch.to(device)
        preds.extend(model(batch).cpu().numpy().tolist())
    return np.array(preds)

# ── 데이터 & 모델 ─────────────────────────────────────────────────────────────
log_device()
log("[5Method] 데이터 로드...")
df = load_data()
df["mol_id"] = df.index
train_df, val_df, test_df = split_data(df, seed=BASE_SEED)
X_train = get_rdkit_features(train_df["canonical_smiles"].tolist())
y_train = train_df["MMGBSA dG Bind"].values
train_smiles_set = set(train_df["canonical_smiles"].tolist())
all_smiles = df["canonical_smiles"].tolist()

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)

log("[5Method] Ridge 학습...")
ridge = Ridge(alpha=3708.369222848137)
ridge.fit(X_train_sc, y_train)

log("[5Method] LightGBM 학습 (독립 교차검증용)...")
optuna.logging.set_verbosity(optuna.logging.WARNING)
lgb_params = optuna.load_study(study_name="lgbm_forward",
    storage=f"sqlite:///{RESULTS_DIR}/optuna_studies.db").best_params
p = {k: v for k, v in lgb_params.items()}
if "lr" in p: p["learning_rate"] = p.pop("lr")
p["n_estimators"] = min(p.get("n_estimators", 500), 300)
lgb_model = lgb.LGBMRegressor(**p, random_state=BASE_SEED, verbosity=-1, n_jobs=-1)
lgb_model.fit(X_train, y_train)

log("[5Method] GNN 학습 (독립 교차검증용)...")
gnn_params = optuna.load_study(study_name="gnn_attentivefp_forward",
    storage=f"sqlite:///{RESULTS_DIR}/optuna_studies.db").best_params
train_graphs_gnn = [g for g in [mol_to_graph(s, y) for s, y in
    zip(train_df["canonical_smiles"].tolist(), train_df["MMGBSA dG Bind"].tolist())] if g is not None]
val_graphs_gnn   = [g for g in [mol_to_graph(s, y) for s, y in
    zip(val_df["canonical_smiles"].tolist(), val_df["MMGBSA dG Bind"].tolist())] if g is not None]
torch.manual_seed(BASE_SEED); np.random.seed(BASE_SEED)
gnn_model = GNNRegressor(gnn_params["hidden_dim"], gnn_params["num_layers"],
                          gnn_params["num_timesteps"], gnn_params["dropout"]).to(DEVICE)
gnn_optimizer = torch.optim.AdamW(gnn_model.parameters(), lr=gnn_params["lr"], weight_decay=1e-4)
from transformers import get_cosine_schedule_with_warmup
_gnn_train_dl = PyGLoader(train_graphs_gnn, batch_size=gnn_params["batch_size"], shuffle=True)
_gnn_val_dl   = PyGLoader(val_graphs_gnn,   batch_size=128, shuffle=False)
_total = len(_gnn_train_dl) * 50
_sched = get_cosine_schedule_with_warmup(gnn_optimizer, int(_total*0.05), _total)
_criterion = nn.MSELoss()
_best_rmse_gnn, _best_state_gnn, _pat = float("inf"), None, 0
for _ep in range(50):
    gnn_model.train()
    for _b in _gnn_train_dl:
        _b = _b.to(DEVICE); gnn_optimizer.zero_grad()
        _loss = _criterion(gnn_model(_b), _b.y)
        _loss.backward(); nn.utils.clip_grad_norm_(gnn_model.parameters(), 1.0)
        gnn_optimizer.step(); _sched.step()
    with torch.no_grad():
        gnn_model.eval()
        _vp, _vl = [], []
        for _b in _gnn_val_dl:
            _b = _b.to(DEVICE); _vp.extend(gnn_model(_b).cpu().numpy()); _vl.extend(_b.y.cpu().numpy())
        _vrmse = float(np.sqrt(np.mean((np.array(_vp)-np.array(_vl))**2)))
    if _vrmse < _best_rmse_gnn:
        _best_rmse_gnn = _vrmse
        _best_state_gnn = {k: v.cpu().clone() for k,v in gnn_model.state_dict().items()}; _pat = 0
    else:
        _pat += 1
        if _pat >= 15: break
gnn_model.load_state_dict(_best_state_gnn)
log(f"  GNN best val RMSE={_best_rmse_gnn:.4f}")

log("[5Method] BiLSTM 학습 (독립 교차검증용)...")
aug_df = pd.read_pickle(f"{RESULTS_DIR}/augmented_full.pkl")
train_ids = set(train_df["mol_id"])
val_ids   = set(val_df["mol_id"])
train_aug = aug_df[aug_df["mol_id"].isin(train_ids)].reset_index(drop=True)
val_orig  = aug_df[aug_df["mol_id"].isin(val_ids) & (~aug_df["is_augmented"])].reset_index(drop=True)
bilstm_params = optuna.load_study(study_name="bilstm_forward",
    storage=f"sqlite:///{RESULTS_DIR}/optuna_studies.db").best_params
bilstm_model, _ = m4.train_bilstm(
    train_aug, val_orig,
    embed_dim=bilstm_params["embed_dim"], hidden_dim=bilstm_params["hidden_dim"],
    n_layers=bilstm_params["n_layers"], dropout=bilstm_params["dropout"],
    lr=bilstm_params["lr"], batch_size=bilstm_params["batch_size"],
    epochs=10, device=DEVICE, seed=BASE_SEED,
)
log("[5Method] 모든 모델 준비 완료")

# 전체 데이터 Ridge 예측 (Retrieval용)
X_all = get_rdkit_features(all_smiles)
X_all_sc = scaler.transform(X_all)
pred_all = ridge.predict(X_all_sc)
log("[5Method] 준비 완료")


# ── 평가 함수 ─────────────────────────────────────────────────────────────────
def evaluate_molecules(smiles_list, target_dG):
    valid_smiles, valid_mols = [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            can = Chem.MolToSmiles(mol, canonical=True)
            valid_smiles.append(can)
            valid_mols.append(mol)
    if not valid_mols:
        return {"n": 0, "Ridge_hit3": 0.0, "LGB_hit3": 0.0,
                "GNN_hit3": 0.0, "BiLSTM_hit3": 0.0,
                "QED": 0.0, "SA_score": 0.0, "Lipinski": 0.0, "novelty": 0.0}

    X = get_rdkit_features(valid_smiles)
    X_sc = scaler.transform(X)
    pred_ridge  = ridge.predict(X_sc)
    pred_lgb    = lgb_model.predict(X)
    pred_gnn    = _gnn_predict(gnn_model, valid_smiles, DEVICE)
    pred_bilstm = m4.predict_forward_bilstm(bilstm_model, valid_smiles, DEVICE)
    qed_scores  = [QED.qed(m) for m in valid_mols]
    sa_scores   = [sascorer.calculateScore(m) for m in valid_mols]
    lipinski_ok = [
        int(Descriptors.MolWt(m) <= 500 and Descriptors.MolLogP(m) <= 5 and
            rdMolDescriptors.CalcNumHBD(m) <= 5 and rdMolDescriptors.CalcNumHBA(m) <= 10)
        for m in valid_mols
    ]
    novelty = sum(1 for s in valid_smiles if s not in train_smiles_set) / len(valid_smiles) * 100

    def hit_rate(preds):
        arr = np.array(preds, dtype=float)
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0: return 0.0
        return round(sum(abs(p - target_dG) <= 3 for p in valid) / len(valid) * 100, 1)

    return {
        "n": len(valid_mols),
        "Ridge_hit3":  hit_rate(pred_ridge),
        "LGB_hit3":    hit_rate(pred_lgb),
        "GNN_hit3":    hit_rate(pred_gnn),
        "BiLSTM_hit3": hit_rate(pred_bilstm),
        "QED":      round(float(np.mean(qed_scores)), 3),
        "SA_score": round(float(np.mean(sa_scores)), 3),
        "Lipinski": round(float(np.mean(lipinski_ok)) * 100, 1),
        "novelty":  round(novelty, 1),
    }


# ── GA ────────────────────────────────────────────────────────────────────────
def run_ga(target_dG, seed, pop_size=100, n_gen=150):
    random.seed(seed)
    train_mols = [Chem.MolFromSmiles(s) for s in train_df["canonical_smiles"]
                  if Chem.MolFromSmiles(s)]

    def fitness(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return -999
        X = get_rdkit_features([smi])
        return -abs(ridge.predict(scaler.transform(X))[0] - target_dG)

    def mutate(mol):
        try:
            frags = BRICS.BRICSDecompose(mol)
            if len(frags) > 1:
                donor = random.choice(train_mols)
                donor_frags = BRICS.BRICSDecompose(donor)
                if donor_frags:
                    frag = random.choice(list(donor_frags))
                    new_mol = Chem.MolFromSmiles(
                        frag.replace('[*]','').replace('[1*]','').replace('[2*]','').replace('[3*]',''))
                    if new_mol: return new_mol
        except: pass
        try:
            rwmol = RWMol(mol)
            atoms = [a for a in rwmol.GetAtoms() if a.GetAtomicNum() in [6,7,8]]
            if atoms:
                random.choice(atoms).SetAtomicNum(random.choice([6,7,8,9,16]))
            Chem.SanitizeMol(rwmol)
            return rwmol.GetMol()
        except: pass
        return mol

    pop = random.sample(train_mols, min(pop_size, len(train_mols)))
    pop_smi = [Chem.MolToSmiles(m) for m in pop]
    pop_fit = [fitness(s) for s in pop_smi]

    for _ in range(n_gen):
        children_smi = []
        for _ in range(pop_size // 2):
            i, j = random.sample(range(len(pop)), 2)
            parent = pop[i] if pop_fit[i] > pop_fit[j] else pop[j]
            children_smi.append(Chem.MolToSmiles(mutate(parent)))
        children_fit = [fitness(s) for s in children_smi]
        all_s = pop_smi + children_smi
        all_f = pop_fit + children_fit
        idx = np.argsort(all_f)[::-1][:pop_size]
        pop_smi = [all_s[i] for i in idx]
        pop_fit = [all_f[i] for i in idx]
        pop = [Chem.MolFromSmiles(s) or random.choice(train_mols) for s in pop_smi]

    top_idx = np.argsort(pop_fit)[::-1][:N_GENERATE]
    return [pop_smi[i] for i in top_idx]


# ── BO ────────────────────────────────────────────────────────────────────────
def run_bo(target_dG, seed):
    random.seed(seed)
    close_idx = np.argsort(np.abs(pred_all - target_dG))[:20]
    seed_mols = [Chem.MolFromSmiles(all_smiles[i]) for i in close_idx]
    seed_mols = [m for m in seed_mols if m]

    results = []
    for mol in seed_mols:
        for _ in range(50):
            try:
                rwmol = RWMol(mol)
                atoms = [a for a in rwmol.GetAtoms() if a.GetAtomicNum() in [6,7,8]]
                if atoms:
                    random.choice(atoms).SetAtomicNum(random.choice([6,7,8,9,16,17]))
                Chem.SanitizeMol(rwmol)
                new_smi = Chem.MolToSmiles(rwmol.GetMol())
                if Chem.MolFromSmiles(new_smi):
                    X = get_rdkit_features([new_smi])
                    pred = ridge.predict(scaler.transform(X))[0]
                    results.append((new_smi, abs(pred - target_dG)))
            except: continue
    for i in close_idx:
        smi = all_smiles[i]
        results.append((smi, abs(pred_all[i] - target_dG)))

    results.sort(key=lambda x: x[1])
    seen, unique = set(), []
    for smi, _ in results:
        if smi not in seen:
            seen.add(smi); unique.append(smi)
        if len(unique) >= N_GENERATE: break
    return unique


# ── Scaffold Hopping ─────────────────────────────────────────────────────────
def run_scaffoldhop(target_dG, seed):
    """
    훈련셋에서 target 근접 분자의 스캐폴드 추출 후
    다른 훈련 분자의 치환기 조합으로 호핑 (Murcko scaffold + R-group swap)
    """
    from rdkit.Chem.Scaffolds import MurckoScaffold
    random.seed(seed)
    np.random.seed(seed)

    close_idx = np.argsort(np.abs(pred_all - target_dG))[:30]
    seed_smiles = [all_smiles[i] for i in close_idx]

    # 훈련 분자에서 치환기 라이브러리 구축 (단순: BRICS 말단 프래그먼트)
    substituents = set()
    for smi in random.sample(train_df["canonical_smiles"].tolist(),
                             min(300, len(train_df))):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                frags = BRICS.BRICSDecompose(mol)
                for f in frags:
                    clean = f.replace('[*]','').replace('[1*]','').replace('[2*]','') \
                             .replace('[3*]','').replace('[4*]','')
                    m = Chem.MolFromSmiles(clean)
                    if m and m.GetNumAtoms() <= 8:
                        substituents.add(Chem.MolToSmiles(m, canonical=True))
            except: pass
    sub_list = list(substituents) if substituents else ["C", "N", "O", "F", "Cl"]

    results = []
    for smi in seed_smiles:
        mol = Chem.MolFromSmiles(smi)
        if not mol: continue
        # 스캐폴드 추출
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaf_smi = Chem.MolToSmiles(scaffold, canonical=True)
        except:
            scaf_smi = smi

        # 치환기를 랜덤으로 붙이기
        for _ in range(60):
            try:
                sub = random.choice(sub_list)
                combined = scaf_smi + "." + sub
                new_mol = Chem.MolFromSmiles(combined)
                if new_mol:
                    new_smi = Chem.MolToSmiles(new_mol, canonical=True)
                    X = get_rdkit_features([new_smi])
                    pred = ridge.predict(scaler.transform(X))[0]
                    results.append((new_smi, abs(pred - target_dG)))
            except: continue
        # 원본도 포함
        try:
            X = get_rdkit_features([smi])
            pred = ridge.predict(scaler.transform(X))[0]
            results.append((smi, abs(pred - target_dG)))
        except: pass

    results.sort(key=lambda x: x[1])
    seen, unique = set(), []
    for smi, _ in results:
        if smi not in seen:
            seen.add(smi); unique.append(smi)
        if len(unique) >= N_GENERATE: break
    return unique


# ── Fragment Assembly ─────────────────────────────────────────────────────────
def run_fragment(target_dG, seed):
    random.seed(seed)
    np.random.seed(seed)

    # BRICS 분해
    all_frags = set()
    for smi in train_df["canonical_smiles"].tolist():
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                frags = BRICS.BRICSDecompose(mol)
                all_frags.update(frags)
            except: pass
    frag_list = list(all_frags)
    if not frag_list: return []

    # 정제 (와일드카드 제거 → 유효 SMILES)
    clean_frags = []
    for f in frag_list:
        s = f.replace('[*]','').replace('[1*]','').replace('[2*]','').replace('[3*]','').replace('[4*]','')
        mol = Chem.MolFromSmiles(s)
        if mol: clean_frags.append(Chem.MolToSmiles(mol, canonical=True))
    clean_frags = list(set(clean_frags))
    if not clean_frags: return []

    results = []
    for _ in range(10000):
        try:
            # 1~3개 프래그먼트 연결 (단순 SMILES 연결)
            n = random.randint(1, 3)
            parts = random.sample(clean_frags, min(n, len(clean_frags)))
            combined = ".".join(parts)
            mol = Chem.MolFromSmiles(combined)
            if mol:
                smi = Chem.MolToSmiles(mol, canonical=True)
                X = get_rdkit_features([smi])
                pred = ridge.predict(scaler.transform(X))[0]
                results.append((smi, abs(pred - target_dG)))
        except: pass

    results.sort(key=lambda x: x[1])
    seen, unique = set(), []
    for smi, _ in results:
        if smi not in seen:
            seen.add(smi); unique.append(smi)
        if len(unique) >= N_GENERATE: break
    return unique


# ── Similarity Retrieval (baseline) ──────────────────────────────────────────
def run_retrieval(target_dG, seed):
    """Ridge 예측값 기준으로 훈련셋에서 가장 가까운 분자 N개 반환"""
    random.seed(seed)
    np.random.seed(seed)

    diffs = np.abs(pred_all - target_dG)
    idx = np.argsort(diffs)[:N_GENERATE * 3]  # 여유있게 추출

    candidates = []
    for i in idx:
        smi = all_smiles[i]
        mol = Chem.MolFromSmiles(smi)
        if mol:
            candidates.append(Chem.MolToSmiles(mol, canonical=True))
        if len(candidates) >= N_GENERATE: break

    # seed에 따라 약간 다른 순서 (tie-breaking)
    # 동일 Ridge 점수면 random ordering
    random.shuffle(candidates)
    return candidates[:N_GENERATE]


# ── 5-seed 실행 ───────────────────────────────────────────────────────────────
method_fns = {"GA": run_ga, "BO": run_bo, "ScaffoldHop": run_scaffoldhop,
              "Fragment": run_fragment, "Retrieval": run_retrieval}

seed_results = {str(t): {m: [] for m in METHODS} for t in TARGETS}
METHOD_COLORS_DISPLAY = {"GA":"GA", "BO":"BO★", "ScaffoldHop":"ScHop",
                          "Fragment":"Frag", "Retrieval":"Retr"}

for seed in SEEDS:
    log(f"\n{'═'*70}")
    log(f"[Seed {seed}]")
    log(f"{'═'*70}")
    for target in TARGETS:
        log(f"  target={target}")
        for method in METHODS:
            t0 = time.time()
            try:
                smi_list = method_fns[method](target, seed)
                r = evaluate_molecules(smi_list, target)
            except Exception as e:
                log(f"    {method} ERROR: {e}")
                r = {"n": 0, "Ridge_hit3": 0.0, "LGB_hit3": 0.0,
                     "GNN_hit3": 0.0, "BiLSTM_hit3": 0.0,
                     "QED": 0.0, "SA_score": 0.0, "Lipinski": 0.0, "novelty": 0.0}
            seed_results[str(target)][method].append(r)
            log(f"    {method:<12} {time.time()-t0:.1f}s  "
                f"Ridge={r['Ridge_hit3']}%  LGB={r['LGB_hit3']}%  "
                f"GNN={r['GNN_hit3']}%  BiLSTM={r['BiLSTM_hit3']}%  "
                f"QED={r['QED']:.3f}  novelty={r['novelty']}%")


# ── 통계 집계 ─────────────────────────────────────────────────────────────────
METRICS = ["Ridge_hit3", "LGB_hit3", "GNN_hit3", "BiLSTM_hit3", "QED", "SA_score", "Lipinski", "novelty"]
summary = {}
for tkey in seed_results:
    summary[tkey] = {}
    for method in METHODS:
        runs = seed_results[tkey][method]
        mean = {k: float(np.mean([r[k] for r in runs])) for k in METRICS}
        std  = {k: float(np.std( [r[k] for r in runs])) for k in METRICS}
        summary[tkey][method] = {"mean": mean, "std": std, "seeds": runs}

out_path = f"{RESULTS_DIR}/reverse_result.json"
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
log(f"\n저장: {out_path}")

# ── 요약 출력 ─────────────────────────────────────────────────────────────────
log(f"\n{'═'*80}")
log(f"{'Target':>7} {'Method':<14} {'Ridge':>10} {'LGB':>10} {'GNN':>10} {'BiLSTM':>10} {'QED':>8} {'Novelty':>8}")
log(f"{'─'*100}")
for t in TARGETS:
    for m in METHODS:
        s = summary[str(t)][m]
        log(f"{t:>7} {m:<14} "
            f"{s['mean']['Ridge_hit3']:>5.1f}±{s['std']['Ridge_hit3']:>4.1f}  "
            f"{s['mean']['LGB_hit3']:>5.1f}±{s['std']['LGB_hit3']:>4.1f}  "
            f"{s['mean']['GNN_hit3']:>5.1f}±{s['std']['GNN_hit3']:>4.1f}  "
            f"{s['mean']['BiLSTM_hit3']:>5.1f}±{s['std']['BiLSTM_hit3']:>4.1f}  "
            f"{s['mean']['QED']:>5.3f}±{s['std']['QED']:>4.3f}  "
            f"{s['mean']['novelty']:>5.1f}±{s['std']['novelty']:>4.1f}")
log(f"{'═'*80}")
