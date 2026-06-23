"""
Model 3: ChemBERTa + Conditional SMILES Decoder

Forward  (SMILES → MMGBSA): ChemBERTa encoder (BERT) + MLP regression head
Reverse  (MMGBSA → SMILES): 경량 GPT-스타일 Transformer 디코더 (character-level)
                             MMGBSA 값을 조건(condition)으로 주입

Pretrained backbone (forward): seyonec/ChemBERTa-zinc-base-v1 (125M)
Reverse decoder: 처음부터 학습하는 조건부 소형 Transformer
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from data_utils import (
    VOCAB_SIZE, PAD_IDX, SOS_IDX, EOS_IDX,
    smiles_to_ids, ids_to_smiles, is_valid_smiles,
)

CHEMBERTA_NAME = "seyonec/ChemBERTa-zinc-base-v1"

# ═══════════════════════════════════════════════════════════════════════════════
# FORWARD: ChemBERTa Encoder + MLP Regression Head
# ═══════════════════════════════════════════════════════════════════════════════

class ChemBERTaForwardModel(nn.Module):
    """ChemBERTa mean-pool + Morgan FP/기술자 concat + MLP regression head."""

    def __init__(self, dropout: float = 0.1, hidden_dim: int = 256,
                 freeze_layers: int = 0, extra_dim: int = 0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(CHEMBERTA_NAME)
        enc_dim = self.encoder.config.hidden_size  # 384

        for i, layer in enumerate(self.encoder.encoder.layer):
            if i < freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False

        total_dim = enc_dim + extra_dim
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, input_ids, attention_mask, extra_features=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        if extra_features is not None:
            pooled = torch.cat([pooled, extra_features], dim=-1)
        return self.head(pooled).squeeze(-1)


class ChemBERTaForwardDataset(Dataset):
    def __init__(self, smiles_list, targets, tokenizer, max_len=128, extra_features=None):
        self.encodings = tokenizer(
            list(smiles_list),
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.targets = torch.tensor(list(targets), dtype=torch.float32)
        self.extra = torch.tensor(extra_features, dtype=torch.float32) if extra_features is not None else None

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.targets[idx],
        }
        if self.extra is not None:
            item["extra_features"] = self.extra[idx]
        return item


def train_chembert_forward(
    train_df, val_df, tokenizer,
    lr=2e-5, batch_size=16, epochs=20, warmup_ratio=0.1,
    weight_decay=0.01, dropout=0.1, hidden_dim=256, freeze_layers=0,
    train_extra=None, val_extra=None,
    device="cpu", seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    extra_dim = train_extra.shape[1] if train_extra is not None else 0
    train_ds = ChemBERTaForwardDataset(
        train_df["canonical_smiles"], train_df["MMGBSA dG Bind"], tokenizer, extra_features=train_extra)
    val_ds = ChemBERTaForwardDataset(
        val_df["canonical_smiles"], val_df["MMGBSA dG Bind"], tokenizer, extra_features=val_extra)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size * 2)

    model = ChemBERTaForwardModel(
        dropout=dropout, hidden_dim=hidden_dim, freeze_layers=freeze_layers, extra_dim=extra_dim
    ).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
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
            ef = batch.get("extra_features")
            ef = ef.to(device) if ef is not None else None
            preds = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), ef)
            loss = criterion(preds, batch["labels"].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        val_rmse, val_acc, val_spear = _eval_forward(model, val_dl, device)
        marker = "*" if val_spear > best_val_spear else " "
        print(f"    [ChemBERTa Fwd] Ep {epoch+1:3d}  spearman={val_spear:.4f}  (RMSE={val_rmse:.4f}, sign_acc={val_acc:.4f}) {marker}", flush=True)
        if val_spear > best_val_spear:
            best_val_spear = val_spear
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"    [ChemBERTa Fwd] Early stop @ ep {epoch+1}", flush=True)
                break

    model.load_state_dict(best_state)
    return model, best_val_spear


@torch.no_grad()
def _eval_forward(model, dataloader, device):
    model.eval()
    preds_all, labels_all = [], []
    for batch in dataloader:
        ef = batch.get("extra_features")
        ef = ef.to(device) if ef is not None else None
        p = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), ef).cpu().numpy()
        preds_all.append(p)
        labels_all.append(batch["labels"].numpy())
    from scipy.stats import spearmanr
    p = np.concatenate(preds_all)
    l = np.concatenate(labels_all)
    sign_acc = float(np.mean((p < 0) == (l < 0)))
    rmse = float(np.sqrt(np.mean((p - l) ** 2)))
    spearman, _ = spearmanr(p, l)
    return rmse, sign_acc, float(spearman)


@torch.no_grad()
def predict_forward_chembert(model, smiles_list, tokenizer, device, batch_size=32, extra_features=None):
    from data_utils import get_rdkit_features
    model.eval()
    all_preds = []
    if extra_features is None:
        extra_features = get_rdkit_features(smiles_list)
    for i in range(0, len(smiles_list), batch_size):
        chunk = list(smiles_list[i : i + batch_size])
        enc = tokenizer(chunk, max_length=128, padding="max_length",
                        truncation=True, return_tensors="pt")
        ef = torch.tensor(extra_features[i : i + batch_size], dtype=torch.float32).to(device)
        preds = model(enc["input_ids"].to(device), enc["attention_mask"].to(device), ef)
        all_preds.extend(preds.cpu().numpy().tolist())
    return np.array(all_preds)


# ═══════════════════════════════════════════════════════════════════════════════
# REVERSE: 조건부 Character-level Transformer Decoder
# ═══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class ConditionalSMILESDecoder(nn.Module):
    """
    GPT-스타일 Causal Transformer Decoder.
    MMGBSA 값을 embedding하여 첫 번째 token 위치에 주입합니다.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        n_head: int = 8,
        n_layers: int = 4,
        dim_ff: int = 512,
        dropout: float = 0.1,
        max_len: int = 130,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # MMGBSA 값 → 조건 벡터
        self.condition_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt_ids, condition_values, tgt_key_padding_mask=None):
        """
        tgt_ids        : (B, T) int64  – teacher-forced target token ids
        condition_values: (B,)  float  – normalized MMGBSA values
        """
        B, T = tgt_ids.shape

        # 조건 벡터: (B, 1, d_model) → memory로 사용
        cond = self.condition_proj(condition_values.unsqueeze(-1))  # (B, d_model)
        memory = cond.unsqueeze(1)  # (B, 1, d_model)

        # 토큰 임베딩 + 위치 인코딩
        tgt_emb = self.pos_enc(self.token_emb(tgt_ids) * math.sqrt(self.d_model))

        # 인과적 마스크 (causal mask)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=tgt_ids.device)

        out = self.transformer(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.output_proj(out)  # (B, T, vocab_size)

    @torch.no_grad()
    def generate(self, condition_value: float, device: str,
                 max_len: int = 128, temperature: float = 1.0, top_k: int = 10) -> str:
        """단일 MMGBSA 값 → SMILES 생성 (greedy/top-k sampling)."""
        self.eval()
        cond = torch.tensor([[condition_value]], dtype=torch.float32, device=device)
        memory = self.condition_proj(cond).unsqueeze(1)  # (1, 1, d_model)

        ids = [SOS_IDX]
        for _ in range(max_len):
            tgt = torch.tensor([ids], dtype=torch.long, device=device)
            tgt_emb = self.pos_enc(self.token_emb(tgt) * math.sqrt(self.d_model))
            T = tgt.size(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
            out = self.transformer(tgt_emb, memory, tgt_mask=causal_mask)
            logits = self.output_proj(out[:, -1, :]) / temperature  # (1, vocab)
            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, -1:]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            if next_id == EOS_IDX:
                break
            ids.append(next_id)

        return ids_to_smiles(ids)


# ── Reverse Dataset ───────────────────────────────────────────────────────────

class ReverseCharDataset(Dataset):
    """MMGBSA 값 → character-level SMILES 토큰 시퀀스."""

    def __init__(self, smiles_list, targets, max_len=130,
                 target_mean=0.0, target_std=1.0):
        self.ids_list = [smiles_to_ids(s, max_len) for s in smiles_list]
        self.targets  = [(v - target_mean) / (target_std + 1e-8) for v in targets]
        self.max_len  = max_len

    def __len__(self):
        return len(self.ids_list)

    def __getitem__(self, idx):
        ids = self.ids_list[idx]
        # pad to max_len
        padded = ids + [PAD_IDX] * (self.max_len - len(ids))
        padded = padded[: self.max_len]
        return {
            "token_ids": torch.tensor(padded, dtype=torch.long),
            "condition": torch.tensor(self.targets[idx], dtype=torch.float32),
        }


def train_chembert_reverse(
    train_df, val_df,
    target_mean: float = 0.0, target_std: float = 1.0,
    lr=5e-4, batch_size=64, epochs=50, warmup_ratio=0.1,
    weight_decay=0.01, dropout=0.1, d_model=256, n_layers=4,
    device="cpu", seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = ReverseCharDataset(
        train_df["canonical_smiles"], train_df["MMGBSA dG Bind"],
        target_mean=target_mean, target_std=target_std,
    )
    val_ds = ReverseCharDataset(
        val_df["canonical_smiles"], val_df["MMGBSA dG Bind"],
        target_mean=target_mean, target_std=target_std,
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size * 2)

    model = ConditionalSMILESDecoder(
        d_model=d_model, n_layers=n_layers, dropout=dropout
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_dl) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * warmup_ratio), total_steps
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_val_loss = float("inf")
    best_state = None
    patience, patience_count = 30, 0

    for epoch in range(epochs):
        model.train()
        for batch in train_dl:
            ids      = batch["token_ids"].to(device)  # (B, T)
            cond     = batch["condition"].to(device)   # (B,)
            src_ids  = ids[:, :-1]
            tgt_ids  = ids[:, 1:]
            pad_mask = (src_ids == PAD_IDX)

            optimizer.zero_grad()
            logits = model(src_ids, cond, tgt_key_padding_mask=pad_mask)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_ids.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        val_loss = _eval_reverse(model, val_dl, device)
        marker = "*" if val_loss < best_val_loss else " "
        print(f"    [ChemBERTa Rev] Ep {epoch+1:3d}  val_loss={val_loss:.4f} {marker}", flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"    [ChemBERTa Rev] Early stop @ ep {epoch+1}", flush=True)
                break

    model.load_state_dict(best_state)
    return model, best_val_loss


@torch.no_grad()
def _eval_reverse(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    total, n = 0.0, 0
    for batch in dataloader:
        ids     = batch["token_ids"].to(device)
        cond    = batch["condition"].to(device)
        src_ids = ids[:, :-1]
        tgt_ids = ids[:, 1:]
        pad_mask = (src_ids == PAD_IDX)
        logits   = model(src_ids, cond, tgt_key_padding_mask=pad_mask)
        loss     = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_ids.reshape(-1))
        total += loss.item() * len(ids)
        n += len(ids)
    return total / n if n > 0 else float("inf")


def generate_smiles_chembert(
    model, target_values, target_mean, target_std,
    device="cpu", n_per_target=1, temperature=1.0, top_k=10,
):
    """목표 MMGBSA 값 리스트 → 생성된 SMILES 리스트."""
    results = []
    for v in target_values:
        norm_v = (v - target_mean) / (target_std + 1e-8)
        for _ in range(n_per_target):
            smi = model.generate(norm_v, device=device, temperature=temperature, top_k=top_k)
            results.append(smi)
    return results


# ── Optuna objectives ─────────────────────────────────────────────────────────

def chembert_forward_objective(trial, train_df, val_df, tokenizer, device,
                               train_extra=None, val_extra=None):
    lr           = trial.suggest_float("lr", 5e-6, 1e-4, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [8, 16, 32])
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    dropout      = trial.suggest_float("dropout", 0.05, 0.3)
    hidden_dim   = trial.suggest_categorical("hidden_dim", [256, 512, 1024])

    import gc
    model, val_rmse = train_chembert_forward(
        train_df, val_df, tokenizer,
        lr=lr, batch_size=batch_size, warmup_ratio=warmup_ratio,
        weight_decay=weight_decay, dropout=dropout,
        hidden_dim=hidden_dim, freeze_layers=0,
        train_extra=train_extra, val_extra=val_extra,
        device=device, epochs=10, seed=42,
    )
    del model; gc.collect()
    if hasattr(torch.mps, 'empty_cache'): torch.mps.empty_cache()
    return val_rmse


def chembert_reverse_objective(trial, train_df, val_df, target_mean, target_std, device):
    import gc
    lr           = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [32, 64, 128])
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.15)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.05)
    dropout      = trial.suggest_float("dropout", 0.05, 0.2)
    d_model      = trial.suggest_categorical("d_model", [128, 256, 512])
    n_layers     = trial.suggest_int("n_layers", 2, 8)

    model, val_loss = train_chembert_reverse(
        train_df, val_df,
        target_mean=target_mean, target_std=target_std,
        lr=lr, batch_size=batch_size, warmup_ratio=warmup_ratio,
        weight_decay=weight_decay, dropout=dropout,
        d_model=d_model, n_layers=n_layers,
        device=device, epochs=20, seed=42,  # Optuna 탐색용 에폭
    )
    del model; gc.collect()
    if hasattr(torch.mps, 'empty_cache'): torch.mps.empty_cache()
    return val_loss


def get_tokenizer():
    return AutoTokenizer.from_pretrained(CHEMBERTA_NAME)
