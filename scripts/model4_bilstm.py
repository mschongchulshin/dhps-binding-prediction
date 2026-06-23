"""
Model 4: BiLSTM Forward-only (SMILES → MMGBSA regression)

캐릭터 레벨 토크나이저 (data_utils.SMILES_CHARS) 사용.
사전학습 없음 → 파라미터 ~1-5M, RAM ~0.5GB 이하.
Forward 전용 (Reverse 생성 없음).
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from scipy.stats import spearmanr

from data_utils import SMILES_CHARS, CHAR2IDX, PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX

MAX_LEN = 128


# ── Dataset ───────────────────────────────────────────────────────────────────

class SMILESDataset(Dataset):
    def __init__(self, smiles_list, targets, max_len=MAX_LEN):
        self.ids     = [_encode(s, max_len) for s in smiles_list]
        self.targets = torch.tensor(list(targets), dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {"input_ids": self.ids[idx], "labels": self.targets[idx]}


def _encode(smi: str, max_len: int) -> torch.Tensor:
    ids = [SOS_IDX]
    for ch in str(smi)[: max_len - 2]:
        ids.append(CHAR2IDX.get(ch, UNK_IDX))
    ids.append(EOS_IDX)
    # pad
    ids += [PAD_IDX] * (max_len - len(ids))
    return torch.tensor(ids[:max_len], dtype=torch.long)


# ── Model ─────────────────────────────────────────────────────────────────────

class BiLSTMForward(nn.Module):
    def __init__(self, vocab_size=len(SMILES_CHARS), embed_dim=64,
                 hidden_dim=256, n_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=n_layers,
            batch_first=True, bidirectional=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_ids):
        # input_ids: (B, L)
        x = self.embedding(input_ids)           # (B, L, E)
        out, _ = self.lstm(x)                   # (B, L, 2H)
        # mean pool (ignore padding)
        mask = (input_ids != PAD_IDX).unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.regressor(pooled).squeeze(-1)


# ── Training ──────────────────────────────────────────────────────────────────

def train_bilstm(
    train_df, val_df,
    embed_dim=64, hidden_dim=256, n_layers=2, dropout=0.2,
    lr=1e-3, batch_size=64, epochs=100, warmup_ratio=0.05,
    weight_decay=0.01, device="cpu", seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = SMILESDataset(train_df["canonical_smiles"], train_df["MMGBSA dG Bind"])
    val_ds   = SMILESDataset(val_df["canonical_smiles"],   val_df["MMGBSA dG Bind"])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size * 2)

    model = BiLSTMForward(
        embed_dim=embed_dim, hidden_dim=hidden_dim,
        n_layers=n_layers, dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_dl) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * warmup_ratio), total_steps
    )
    criterion = nn.MSELoss()

    best_rmse = float("inf")
    best_state = None
    patience, patience_count = 30, 0

    for epoch in range(epochs):
        model.train()
        for batch in train_dl:
            optimizer.zero_grad()
            preds = model(batch["input_ids"].to(device))
            loss  = criterion(preds, batch["labels"].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        rmse, sign_acc, spear = _evaluate(model, val_dl, device)
        marker = "*" if rmse < best_rmse else " "
        print(f"    [BiLSTM] Ep {epoch+1:3d}  val_RMSE={rmse:.4f}"
              f"  (spearman={spear:.4f}, sign_acc={sign_acc:.4f}) {marker}", flush=True)

        if rmse < best_rmse:
            best_rmse = rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"    [BiLSTM] Early stop @ ep {epoch+1}", flush=True)
                break

    model.load_state_dict(best_state)
    return model, best_rmse


@torch.no_grad()
def _evaluate(model, dataloader, device):
    model.eval()
    preds_all, labels_all = [], []
    for batch in dataloader:
        p = model(batch["input_ids"].to(device)).cpu().numpy()
        preds_all.append(p)
        labels_all.append(batch["labels"].numpy())
    preds_all  = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    rmse     = float(np.sqrt(np.mean((preds_all - labels_all) ** 2)))
    sign_acc = float(np.mean((preds_all < 0) == (labels_all < 0)))
    spear, _ = spearmanr(preds_all, labels_all)
    return rmse, sign_acc, float(spear)


@torch.no_grad()
def predict_forward_bilstm(model, smiles_list, device, batch_size=128):
    model.eval()
    all_preds = []
    for i in range(0, len(smiles_list), batch_size):
        chunk = list(smiles_list[i: i + batch_size])
        ids = torch.stack([_encode(s, MAX_LEN) for s in chunk]).to(device)
        all_preds.extend(model(ids).cpu().numpy().tolist())
    return np.array(all_preds)


# ── Optuna objective ───────────────────────────────────────────────────────────

def bilstm_objective(trial, train_df, val_df, device, epochs=10):
    import gc
    embed_dim  = trial.suggest_categorical("embed_dim",  [32, 64, 128])
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    n_layers   = trial.suggest_int("n_layers", 1, 3)
    dropout    = trial.suggest_float("dropout", 0.1, 0.4)
    lr         = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    model, best_rmse = train_bilstm(
        train_df, val_df,
        embed_dim=embed_dim, hidden_dim=hidden_dim, n_layers=n_layers,
        dropout=dropout, lr=lr, batch_size=batch_size,
        epochs=epochs, device=device, seed=42,
    )
    del model; gc.collect()
    return best_rmse
