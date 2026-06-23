"""
Data loading, validation, and splitting utilities.
SMILES validation uses RDKit; fallback to regex if not installed.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CSV_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "data", "dataset.csv"
)
SMILES_COL = "Smiles"
TARGET_COL = "MMGBSA dG Bind"
SEEDS = [42]  # 빠른 탐색용: 단일 seed

# ── RDKit optional ──────────────────────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    def canonicalize(smi: str):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)

    def is_valid_smiles(smi: str) -> bool:
        return Chem.MolFromSmiles(smi) is not None

    RDKIT_AVAILABLE = True
except ImportError:
    import re
    RDKIT_AVAILABLE = False

    _SMILES_RE = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#\\\/.%]*$")

    def canonicalize(smi: str):
        return smi.strip()

    def is_valid_smiles(smi: str) -> bool:
        s = smi.strip()
        return bool(s) and bool(_SMILES_RE.match(s))


# ── RDKit 분자 기술자 (Morgan FP + 183개 유효 기술자) ──────────────────────────
# 전체 217개 중 NaN 없고 분산>0인 183개 (데이터셋 전체 기준 필터링)
VALID_DESC_NAMES = [
    'MaxAbsEStateIndex','MaxEStateIndex','MinAbsEStateIndex','MinEStateIndex',
    'qed','SPS','MolWt','HeavyAtomMolWt','ExactMolWt','NumValenceElectrons',
    'MaxPartialCharge','MinPartialCharge','MaxAbsPartialCharge','MinAbsPartialCharge',
    'FpDensityMorgan1','FpDensityMorgan2','FpDensityMorgan3',
    'BCUT2D_MWHI','BCUT2D_MWLOW','BCUT2D_CHGHI','BCUT2D_CHGLO',
    'BCUT2D_LOGPHI','BCUT2D_LOGPLOW','BCUT2D_MRHI','BCUT2D_MRLOW',
    'AvgIpc','BalabanJ','BertzCT','Chi0','Chi0n','Chi0v','Chi1','Chi1n','Chi1v',
    'Chi2n','Chi2v','Chi3n','Chi3v','Chi4n','Chi4v','HallKierAlpha','Ipc',
    'Kappa1','Kappa2','Kappa3','LabuteASA',
    'PEOE_VSA1','PEOE_VSA10','PEOE_VSA11','PEOE_VSA12','PEOE_VSA13','PEOE_VSA14',
    'PEOE_VSA2','PEOE_VSA3','PEOE_VSA4','PEOE_VSA5','PEOE_VSA6','PEOE_VSA7',
    'PEOE_VSA8','PEOE_VSA9',
    'SMR_VSA1','SMR_VSA10','SMR_VSA2','SMR_VSA3','SMR_VSA4','SMR_VSA5',
    'SMR_VSA6','SMR_VSA7','SMR_VSA9',
    'SlogP_VSA1','SlogP_VSA10','SlogP_VSA11','SlogP_VSA12','SlogP_VSA2',
    'SlogP_VSA3','SlogP_VSA4','SlogP_VSA5','SlogP_VSA6','SlogP_VSA7','SlogP_VSA8',
    'TPSA',
    'EState_VSA1','EState_VSA10','EState_VSA2','EState_VSA3','EState_VSA4',
    'EState_VSA5','EState_VSA6','EState_VSA7','EState_VSA8','EState_VSA9',
    'VSA_EState1','VSA_EState10','VSA_EState2','VSA_EState3','VSA_EState4',
    'VSA_EState5','VSA_EState6','VSA_EState7','VSA_EState8','VSA_EState9',
    'FractionCSP3','HeavyAtomCount','NHOHCount','NOCount',
    'NumAliphaticCarbocycles','NumAliphaticHeterocycles','NumAliphaticRings',
    'NumAmideBonds','NumAromaticCarbocycles','NumAromaticHeterocycles','NumAromaticRings',
    'NumAtomStereoCenters','NumBridgeheadAtoms','NumHAcceptors','NumHDonors',
    'NumHeteroatoms','NumHeterocycles','NumRotatableBonds',
    'NumSaturatedCarbocycles','NumSaturatedHeterocycles','NumSaturatedRings',
    'NumSpiroAtoms','NumUnspecifiedAtomStereoCenters','Phi','RingCount',
    'MolLogP','MolMR',
    'fr_Al_COO','fr_Al_OH','fr_Al_OH_noTert','fr_ArN','fr_Ar_COO','fr_Ar_N',
    'fr_Ar_NH','fr_Ar_OH','fr_COO','fr_COO2','fr_C_O','fr_C_O_noCOO',
    'fr_HOCCN','fr_Imine','fr_NH0','fr_NH1','fr_NH2',
    'fr_Ndealkylation1','fr_Ndealkylation2','fr_Nhpyrrole','fr_alkyl_halide',
    'fr_allylic_oxid','fr_amide','fr_amidine','fr_aniline','fr_aryl_methyl',
    'fr_benzene','fr_bicyclic','fr_ether','fr_furan','fr_guanido','fr_halogen',
    'fr_hdrzine','fr_hdrzone','fr_imidazole','fr_imide','fr_ketone',
    'fr_ketone_Topliss','fr_lactam','fr_methoxy','fr_morpholine','fr_nitrile',
    'fr_oxazole','fr_para_hydroxylation','fr_piperdine','fr_piperzine',
    'fr_priamide','fr_pyridine','fr_sulfide','fr_sulfonamd','fr_sulfone',
    'fr_tetrazole','fr_thiazole','fr_thiophene','fr_urea',
]


def get_rdkit_features(smiles_list):
    """
    Morgan ECFP4 (2048 bits) + 유효 기술자 183개를 추출.
    반환: (N, 2231) float32 numpy array
    """
    from rdkit.Chem import AllChem, Descriptors
    FP_BITS = 2048

    desc_funcs = {name: fn for name, fn in Descriptors.descList if name in VALID_DESC_NAMES}
    ordered = [(name, desc_funcs[name]) for name in VALID_DESC_NAMES if name in desc_funcs]

    feats = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            feats.append(np.zeros(FP_BITS + len(ordered), dtype=np.float32))
            continue
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_BITS), dtype=np.float32)
        desc = np.array([fn(mol) for _, fn in ordered], dtype=np.float32)
        desc = np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)
        feats.append(np.concatenate([fp, desc]))
    return np.array(feats, dtype=np.float32)


# ── SMILES 증강 (Enumeration Augmentation) ───────────────────────────────────
def augment_smiles(df, n_augment=10, seed=42):
    """
    훈련 데이터 증강: 각 분자에 대해 최대 n_augment개의 랜덤 SMILES 생성.
    val/test에는 사용하지 않음.
    반환: 증강된 DataFrame (canonical_smiles 컬럼이 랜덤 SMILES로 대체)
    """
    import random
    random.seed(seed)

    rows = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row["canonical_smiles"]))
        if mol is None:
            rows.append(row)
            continue

        smiles_set = {row["canonical_smiles"]}  # 캐노니컬 포함
        attempts = 0
        while len(smiles_set) < n_augment + 1 and attempts < n_augment * 5:
            try:
                rand_smi = Chem.MolToSmiles(mol, doRandom=True)
                smiles_set.add(rand_smi)
            except Exception:
                pass
            attempts += 1

        for smi in smiles_set:
            new_row = row.copy()
            new_row["canonical_smiles"] = smi
            rows.append(new_row)

    aug_df = pd.DataFrame(rows).reset_index(drop=True)
    return aug_df


def build_augmented_dataset(df, n_augment=50, seed=42, save_path=None):
    """
    전체 데이터셋을 최대 증강하여 저장.

    각 분자에 mol_id(원본 분자 인덱스)와 is_augmented 컬럼을 추가.
    CV split 시 mol_id 기준으로 split하면 데이터 누수 방지 가능.

    반환: 증강된 DataFrame
    """
    import random
    random.seed(seed)

    rows = []
    for mol_id, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row["canonical_smiles"]))
        if mol is None:
            new_row = row.copy()
            new_row["mol_id"] = mol_id
            new_row["is_augmented"] = False
            rows.append(new_row)
            continue

        smiles_set = {row["canonical_smiles"]}
        attempts = 0
        while len(smiles_set) < n_augment + 1 and attempts < n_augment * 10:
            try:
                rand_smi = Chem.MolToSmiles(mol, doRandom=True)
                smiles_set.add(rand_smi)
            except Exception:
                pass
            attempts += 1

        for smi in smiles_set:
            new_row = row.copy()
            new_row["canonical_smiles"] = smi
            new_row["mol_id"] = mol_id
            new_row["is_augmented"] = (smi != row["canonical_smiles"])
            rows.append(new_row)

    aug_df = pd.DataFrame(rows).reset_index(drop=True)

    if save_path:
        aug_df.to_pickle(save_path)
        print(f"[Augmentation] 저장 완료: {save_path}")
        print(f"[Augmentation] 전체 행: {len(aug_df)} (원본 분자: {df['canonical_smiles'].nunique()})")
        print(f"[Augmentation] 평균 증강 수: {len(aug_df) / len(df):.1f}배")

    return aug_df


# ── MMGBSA binning helpers ───────────────────────────────────────────────────
def value_to_bin_label(value: float, bins: list) -> str:
    """Convert MMGBSA value to a discretized bin label string."""
    for lo, hi in bins:
        if lo <= value < hi:
            return f"{(lo + hi) / 2:.1f}"
    # edge case: at max boundary
    return f"{bins[-1][1]:.1f}"


def build_bins(values: np.ndarray, n_bins: int = 10):
    """Build uniform-width bins over the value range."""
    lo, hi = values.min(), values.max()
    edges = np.linspace(lo, hi + 1e-9, n_bins + 1)
    return list(zip(edges[:-1], edges[1:]))


# ── Core data loading ─────────────────────────────────────────────────────────
def load_data(csv_path: str = CSV_PATH):
    df = pd.read_csv(csv_path)
    df = df[[SMILES_COL, TARGET_COL]].dropna()
    df[SMILES_COL] = df[SMILES_COL].astype(str).str.strip()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna()

    # Validate SMILES
    mask = df[SMILES_COL].apply(is_valid_smiles)
    n_invalid = (~mask).sum()
    if n_invalid > 0:
        print(f"[data_utils] Dropping {n_invalid} invalid SMILES rows.")
    df = df[mask].reset_index(drop=True)

    # Canonicalize
    df["canonical_smiles"] = df[SMILES_COL].apply(canonicalize)
    df = df.dropna(subset=["canonical_smiles"]).reset_index(drop=True)

    print(f"[data_utils] Loaded {len(df)} valid molecules.")
    print(f"[data_utils] MMGBSA range: {df[TARGET_COL].min():.2f} ~ {df[TARGET_COL].max():.2f}")
    print(f"[data_utils] RDKit available: {RDKIT_AVAILABLE}")
    return df


def split_data(df: pd.DataFrame, seed: int = 42, val_ratio: float = 0.15, test_ratio: float = 0.15):
    """Stratified split by MMGBSA quantile bins."""
    bins = pd.qcut(df[TARGET_COL], q=10, labels=False, duplicates="drop")
    train_df, tmp_df = train_test_split(
        df, test_size=val_ratio + test_ratio, random_state=seed, stratify=bins
    )
    bins_tmp = pd.qcut(tmp_df[TARGET_COL], q=5, labels=False, duplicates="drop")
    val_df, test_df = train_test_split(
        tmp_df,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=seed,
        stratify=bins_tmp,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ── SMILES character vocabulary ───────────────────────────────────────────────
SMILES_CHARS = (
    ["<pad>", "<sos>", "<eos>", "<unk>"]
    + list("CNOSPFClBrI")                        # heavy atoms
    + list("cnos")                               # aromatic
    + list("0123456789")
    + list("@+-=#\\/[](). %")                   # structural
)

CHAR2IDX = {c: i for i, c in enumerate(SMILES_CHARS)}
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}
VOCAB_SIZE = len(SMILES_CHARS)
PAD_IDX = CHAR2IDX["<pad>"]
SOS_IDX = CHAR2IDX["<sos>"]
EOS_IDX = CHAR2IDX["<eos>"]
UNK_IDX = CHAR2IDX["<unk>"]


def smiles_to_ids(smi: str, max_len: int = 128) -> list:
    ids = [SOS_IDX]
    for ch in smi[:max_len - 2]:
        ids.append(CHAR2IDX.get(ch, UNK_IDX))
    ids.append(EOS_IDX)
    return ids


def ids_to_smiles(ids: list) -> str:
    chars = []
    for idx in ids:
        if idx == EOS_IDX:
            break
        if idx in (SOS_IDX, PAD_IDX):
            continue
        chars.append(IDX2CHAR.get(idx, ""))
    return "".join(chars)


# ── Normalizer ────────────────────────────────────────────────────────────────
class TargetNormalizer:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, values: np.ndarray):
        self.mean = float(np.mean(values))
        self.std = float(np.std(values))

    def transform(self, values):
        return (np.array(values) - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, values):
        return np.array(values) * (self.std + 1e-8) + self.mean


if __name__ == "__main__":
    df = load_data()
    train_df, val_df, test_df = split_data(df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
