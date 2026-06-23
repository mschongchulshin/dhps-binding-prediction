"""
Model 1: BART-based Bidirectional Molecular Model (MolBART/ChemBART style)

Forward  (SMILES → MMGBSA): BartModel encoder + regression MLP head
Reverse  (MMGBSA → SMILES): BartForConditionalGeneration seq2seq
  - Input text : "Generate SMILES for dG = <VALUE> :"
  - Target text: canonical SMILES

Pretrained backbone: facebook/bart-base (140M params)
"""
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BartTokenizer,
    BartModel,
    BartForConditionalGeneration,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

MODEL_NAME = "facebook/bart-base"


# ── Datasets ─────────────────────────────────────────────────────────────────

class ForwardDataset(Dataset):
    """SMILES → MMGBSA regression dataset."""

    def __init__(self, smiles_list, targets, tokenizer, max_len=128):
        self.encodings = tokenizer(
            list(smiles_list),
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.targets = torch.tensor(list(targets), dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.targets[idx],
        }


class ReverseDataset(Dataset):
    """MMGBSA → SMILES seq2seq dataset."""

    def __init__(self, smiles_list, targets, tokenizer, max_src=32, max_tgt=128):
        # Build source text: "Generate SMILES for dG = -27.50 :"
        src_texts = [f"Generate SMILES for dG = {v:.2f} :" for v in targets]
        self.src = tokenizer(
            src_texts,
            max_length=max_src,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.tgt = tokenizer(
            list(smiles_list),
            max_length=max_tgt,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1

    def __len__(self):
        return len(self.src["input_ids"])

    def __getitem__(self, idx):
        labels = self.tgt["input_ids"][idx].clone()
        labels[labels == self.pad_id] = -100  # mask padding tokens
        return {
            "input_ids": self.src["input_ids"][idx],
            "attention_mask": self.src["attention_mask"][idx],
            "labels": labels,
        }


# ── Forward model ─────────────────────────────────────────────────────────────

class BARTForwardModel(nn.Module):
    """BART encoder + mean-pool + MLP regression head."""

    def __init__(self, dropout: float = 0.1, hidden_dim: int = 256):
        super().__init__()
        self.encoder = BartModel.from_pretrained(MODEL_NAME).encoder
        enc_dim = self.encoder.config.d_model  # 768 for bart-base
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(enc_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pool over non-padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()
        hidden = (out.last_hidden_state * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
        return self.regressor(hidden).squeeze(-1)


def train_bart_forward(
    train_df, val_df, tokenizer,
    lr=2e-5, batch_size=16, epochs=15, warmup_ratio=0.1,
    weight_decay=0.01, dropout=0.1, hidden_dim=256,
    device="cpu", seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = ForwardDataset(train_df["canonical_smiles"], train_df["MMGBSA dG Bind"], tokenizer)
    val_ds   = ForwardDataset(val_df["canonical_smiles"],   val_df["MMGBSA dG Bind"],   tokenizer)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size * 2)

    model = BARTForwardModel(dropout=dropout, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_dl) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * warmup_ratio), total_steps
    )
    criterion = nn.MSELoss()

    best_val_spear = float("-inf")
    best_state = None
    patience, patience_count = 30, 0

    for epoch in range(epochs):
        model.train()
        for batch in train_dl:
            optimizer.zero_grad()
            preds = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            loss = criterion(preds, batch["labels"].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        val_rmse, val_acc, val_spear = evaluate_forward(model, val_dl, device)
        marker = "*" if val_spear > best_val_spear else " "
        print(f"    [BART Fwd] Ep {epoch+1:3d}  spearman={val_spear:.4f}  (RMSE={val_rmse:.4f}, sign_acc={val_acc:.4f}) {marker}", flush=True)
        if val_spear > best_val_spear:
            best_val_spear = val_spear
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"    [BART Fwd] Early stop @ ep {epoch+1}", flush=True)
                break

    model.load_state_dict(best_state)
    return model, best_val_spear


@torch.no_grad()
def evaluate_forward(model, dataloader, device):
    model.eval()
    preds_all, labels_all = [], []
    for batch in dataloader:
        preds = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        ).cpu().numpy()
        preds_all.append(preds)
        labels_all.append(batch["labels"].numpy())
    from scipy.stats import spearmanr
    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    rmse = float(np.sqrt(np.mean((preds_all - labels_all) ** 2)))
    sign_acc = float(np.mean((preds_all < 0) == (labels_all < 0)))
    spearman, _ = spearmanr(preds_all, labels_all)
    return rmse, sign_acc, float(spearman)


@torch.no_grad()
def predict_forward(model, smiles_list, tokenizer, device, batch_size=32):
    model.eval()
    all_preds = []
    for i in range(0, len(smiles_list), batch_size):
        chunk = list(smiles_list[i : i + batch_size])
        enc = tokenizer(chunk, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        preds = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        all_preds.extend(preds.cpu().numpy().tolist())
    return np.array(all_preds)


# ── Reverse model ─────────────────────────────────────────────────────────────

def train_bart_reverse(
    train_df, val_df, tokenizer,
    lr=2e-5, batch_size=8, epochs=15, warmup_ratio=0.1,
    weight_decay=0.01,
    device="cpu", seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = ReverseDataset(train_df["canonical_smiles"], train_df["MMGBSA dG Bind"], tokenizer)
    val_ds   = ReverseDataset(val_df["canonical_smiles"],   val_df["MMGBSA dG Bind"],   tokenizer)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size * 2)

    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_dl) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * warmup_ratio), total_steps
    )

    best_val_loss = float("inf")
    best_state = None
    patience, patience_count = 30, 0

    for epoch in range(epochs):
        model.train()
        for batch in train_dl:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            out.loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        val_loss = _val_loss_reverse(model, val_dl, device)
        marker = "*" if val_loss < best_val_loss else " "
        print(f"    [BART Rev] Ep {epoch+1:3d}  val_loss={val_loss:.4f} {marker}", flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"    [BART Rev] Early stop @ ep {epoch+1}", flush=True)
                break

    model.load_state_dict(best_state)
    return model, best_val_loss


@torch.no_grad()
def _val_loss_reverse(model, dataloader, device):
    model.eval()
    total_loss, n = 0.0, 0
    for batch in dataloader:
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
        total_loss += out.loss.item() * len(batch["input_ids"])
        n += len(batch["input_ids"])
    return total_loss / n if n > 0 else float("inf")


@torch.no_grad()
def generate_smiles_bart(model, tokenizer, target_values, num_beams=4, max_new_tokens=100, device="cpu"):
    """Generate SMILES for each target MMGBSA value."""
    model.eval()
    src_texts = [f"Generate SMILES for dG = {v:.2f} :" for v in target_values]
    enc = tokenizer(
        src_texts,
        max_length=32,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    outputs = model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        early_stopping=True,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


# ── Optuna objective helpers ──────────────────────────────────────────────────

def bart_forward_objective(trial, train_df, val_df, tokenizer, device):
    import gc
    lr          = trial.suggest_float("lr", 5e-6, 1e-4, log=True)
    batch_size  = trial.suggest_categorical("batch_size", [8, 16, 32])
    warmup_ratio= trial.suggest_float("warmup_ratio", 0.0, 0.2)
    weight_decay= trial.suggest_float("weight_decay", 0.0, 0.1)
    dropout     = trial.suggest_float("dropout", 0.05, 0.3)
    hidden_dim  = trial.suggest_categorical("hidden_dim", [256, 512, 1024])
    model, val_rmse = train_bart_forward(
        train_df, val_df, tokenizer,
        lr=lr, batch_size=batch_size, warmup_ratio=warmup_ratio,
        weight_decay=weight_decay, dropout=dropout, hidden_dim=hidden_dim,
        device=device, epochs=12, seed=42,  # Optuna 탐색용 에폭
    )
    del model
    gc.collect()
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    return val_rmse


def bart_reverse_objective(trial, train_df, val_df, tokenizer, device):
    import gc
    lr          = trial.suggest_float("lr", 5e-6, 1e-4, log=True)
    batch_size  = trial.suggest_categorical("batch_size", [8, 16, 32])
    warmup_ratio= trial.suggest_float("warmup_ratio", 0.0, 0.2)
    weight_decay= trial.suggest_float("weight_decay", 0.0, 0.1)
    model, val_loss = train_bart_reverse(
        train_df, val_df, tokenizer,
        lr=lr, batch_size=batch_size, warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        device=device, epochs=10, seed=42,  # Optuna 탐색용 에폭
    )
    del model
    gc.collect()
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    return val_loss


def get_tokenizer():
    return BartTokenizer.from_pretrained(MODEL_NAME)
