"""
Model 2: MolT5 – Text-to-Text Transformer (Google T5 기반)

모든 태스크를 "텍스트 번역"으로 취급합니다.
하나의 T5ForConditionalGeneration 모델로 두 방향을 동시 학습합니다.

Forward  프롬프트: "Predict MMGBSA for SMILES: <SMILES>"  → "<value>"
Reverse  프롬프트: "Design molecule with MMGBSA: <value>"  → "<SMILES>"

Pretrained backbone: laituan245/molt5-small (77M params)
"""
import math
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

MODEL_NAME = "laituan245/molt5-small"

# ── Prompt builders ────────────────────────────────────────────────────────────

def forward_prompt(smiles: str) -> str:
    return f"Predict MMGBSA for SMILES: {smiles}"

def reverse_prompt(value: float) -> str:
    return f"Design molecule with MMGBSA: {value:.2f}"

def format_value(value: float) -> str:
    return f"{value:.2f}"

def parse_value(text: str) -> float | None:
    """Extract first numeric value from generated text."""
    m = re.search(r"[-+]?\d*\.?\d+", text)
    return float(m.group()) if m else None


# ── Datasets ──────────────────────────────────────────────────────────────────

class ForwardT5Dataset(Dataset):
    """SMILES → MMGBSA value (as text)."""

    def __init__(self, smiles_list, targets, tokenizer, max_src=150, max_tgt=16):
        src_texts = [forward_prompt(s) for s in smiles_list]
        tgt_texts = [format_value(v) for v in targets]

        self.src = tokenizer(
            src_texts, max_length=max_src, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        self.tgt = tokenizer(
            tgt_texts, max_length=max_tgt, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __len__(self):
        return len(self.src["input_ids"])

    def __getitem__(self, idx):
        labels = self.tgt["input_ids"][idx].clone()
        labels[labels == self.pad_id] = -100
        return {
            "input_ids": self.src["input_ids"][idx],
            "attention_mask": self.src["attention_mask"][idx],
            "labels": labels,
        }


class ReverseT5Dataset(Dataset):
    """MMGBSA value (as text) → SMILES."""

    def __init__(self, smiles_list, targets, tokenizer, max_src=32, max_tgt=150):
        src_texts = [reverse_prompt(v) for v in targets]
        tgt_texts = list(smiles_list)

        self.src = tokenizer(
            src_texts, max_length=max_src, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        self.tgt = tokenizer(
            tgt_texts, max_length=max_tgt, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __len__(self):
        return len(self.src["input_ids"])

    def __getitem__(self, idx):
        labels = self.tgt["input_ids"][idx].clone()
        labels[labels == self.pad_id] = -100
        return {
            "input_ids": self.src["input_ids"][idx],
            "attention_mask": self.src["attention_mask"][idx],
            "labels": labels,
        }


class MultiTaskT5Dataset(Dataset):
    """Forward + Reverse 동시 학습용 멀티태스크 데이터셋.
    두 태스크의 max_src/max_tgt를 동일하게 맞춰 배치 충돌을 방지합니다.
    """

    def __init__(self, smiles_list, targets, tokenizer,
                 max_src=150, max_fwd_tgt=150, max_rev_tgt=150):
        # 동일한 max_len으로 통일하여 DataLoader collation 오류 방지
        self.forward_ds = ForwardT5Dataset(smiles_list, targets, tokenizer, max_src, max_fwd_tgt)
        self.reverse_ds = ReverseT5Dataset(smiles_list, targets, tokenizer, max_src, max_rev_tgt)

    def __len__(self):
        return len(self.forward_ds) + len(self.reverse_ds)

    def __getitem__(self, idx):
        if idx < len(self.forward_ds):
            return self.forward_ds[idx]
        else:
            return self.reverse_ds[idx - len(self.forward_ds)]


# ── Training ──────────────────────────────────────────────────────────────────

def _build_dataset(smiles_list, targets, tokenizer, task: str):
    if task == "forward":
        return ForwardT5Dataset(smiles_list, targets, tokenizer)
    elif task == "reverse":
        return ReverseT5Dataset(smiles_list, targets, tokenizer)
    else:  # multitask
        return MultiTaskT5Dataset(smiles_list, targets, tokenizer)


def train_molt5(
    train_df, val_df, tokenizer,
    task: str = "multitask",
    lr: float = 1e-4,
    batch_size: int = 16,
    epochs: int = 20,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    device: str = "cpu",
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_smiles  = train_df["canonical_smiles"].tolist()
    train_targets = train_df["MMGBSA dG Bind"].tolist()
    val_smiles    = val_df["canonical_smiles"].tolist()
    val_targets   = val_df["MMGBSA dG Bind"].tolist()

    train_ds = _build_dataset(train_smiles, train_targets, tokenizer, task)
    val_fwd  = ForwardT5Dataset(val_smiles, val_targets, tokenizer)
    val_rev  = ReverseT5Dataset(val_smiles, val_targets, tokenizer)

    train_dl   = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_fwd_dl = DataLoader(val_fwd,  batch_size=batch_size * 2)
    val_rev_dl = DataLoader(val_rev,  batch_size=batch_size * 2)

    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_dl) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * warmup_ratio), total_steps
    )

    best_val_spear = float("-inf")
    best_state = None
    patience, patience_count = 30, 0

    for epoch in range(epochs):
        model.train()
        for batch in train_dl:
            optimizer.zero_grad()
            loss = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            ).loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Validation: forward RMSE + reverse loss (equal weight)
        fwd_rmse, fwd_acc, fwd_spear = _forward_rmse(model, val_fwd_dl, tokenizer, device)
        rev_loss = _seq2seq_loss(model, val_rev_dl, device)
        marker = "*" if fwd_spear > best_val_spear else " "
        print(f"    [MolT5] Ep {epoch+1:3d}  spearman={fwd_spear:.4f}  (RMSE={fwd_rmse:.4f}, sign_acc={fwd_acc:.4f}, rev_loss={rev_loss:.4f}) {marker}", flush=True)

        if fwd_spear > best_val_spear:
            best_val_spear = fwd_spear
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"    [MolT5] Early stop @ ep {epoch+1}", flush=True)
                break

    model.load_state_dict(best_state)
    return model, best_val_spear


# ── Evaluation helpers ────────────────────────────────────────────────────────

@torch.no_grad()
def _seq2seq_loss(model, dataloader, device):
    model.eval()
    total, n = 0.0, 0
    for batch in dataloader:
        loss = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        ).loss.item()
        total += loss * len(batch["input_ids"])
        n += len(batch["input_ids"])
    return total / n if n > 0 else float("inf")


@torch.no_grad()
def _forward_rmse(model, dataloader, tokenizer, device, num_beams=1):
    """Generate text predictions → parse float → compute RMSE."""
    model.eval()
    preds_all, labels_all = [], []
    for batch in dataloader:
        outputs = model.generate(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            max_new_tokens=16,
            num_beams=num_beams,
        )
        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for text, label_ids in zip(texts, batch["labels"]):
            val = parse_value(text)
            if val is not None:
                preds_all.append(val)
            else:
                preds_all.append(0.0)
            # decode label
            real_ids = label_ids[label_ids != -100]
            real_text = tokenizer.decode(real_ids, skip_special_tokens=True)
            real_val = parse_value(real_text)
            labels_all.append(real_val if real_val is not None else 0.0)

    from scipy.stats import spearmanr
    preds_all  = np.array(preds_all)
    labels_all = np.array(labels_all)
    sign_acc = float(np.mean((preds_all < 0) == (labels_all < 0)))
    rmse = float(np.sqrt(np.mean((preds_all - labels_all) ** 2)))
    spearman, _ = spearmanr(preds_all, labels_all)
    return rmse, sign_acc, float(spearman)


@torch.no_grad()
def predict_forward_molt5(model, smiles_list, tokenizer, device,
                           batch_size=32, num_beams=2):
    """SMILES 리스트 → 예측 MMGBSA 값 배열 반환."""
    model.eval()
    all_preds = []
    for i in range(0, len(smiles_list), batch_size):
        chunk = list(smiles_list[i : i + batch_size])
        src_texts = [forward_prompt(s) for s in chunk]
        enc = tokenizer(src_texts, max_length=150, padding="max_length",
                        truncation=True, return_tensors="pt").to(device)
        outputs = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=16,
            num_beams=num_beams,
        )
        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for t in texts:
            v = parse_value(t)
            all_preds.append(v if v is not None else float("nan"))
    return np.array(all_preds)


@torch.no_grad()
def generate_smiles_molt5(model, tokenizer, target_values,
                           num_beams=4, max_new_tokens=150, device="cpu"):
    """목표 MMGBSA 값 리스트 → 생성된 SMILES 리스트 반환."""
    model.eval()
    src_texts = [reverse_prompt(v) for v in target_values]
    all_smiles = []
    batch_size = 16
    for i in range(0, len(src_texts), batch_size):
        chunk = src_texts[i : i + batch_size]
        enc = tokenizer(chunk, max_length=32, padding="max_length",
                        truncation=True, return_tensors="pt").to(device)
        outputs = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            early_stopping=True,
        )
        all_smiles.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return all_smiles


# ── Optuna objectives ─────────────────────────────────────────────────────────

def molt5_objective(trial, train_df, val_df, tokenizer, device):
    lr           = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [8, 16, 32])
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)

    import gc
    model, val_combined = train_molt5(
        train_df, val_df, tokenizer,
        task="multitask",
        lr=lr, batch_size=batch_size, warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        device=device, epochs=8, seed=42,   # Optuna 탐색용 에폭
    )
    del model; gc.collect()
    if hasattr(torch.mps, 'empty_cache'): torch.mps.empty_cache()
    return val_combined


def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)
