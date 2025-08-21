# train.py
import json
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

from models.fantasy_transformer import FantasyTransformer

DATA_DIR = Path("data/processed")
NPZ_PATH = DATA_DIR / "player_sequences_npz.npz"
META_PATH = DATA_DIR / "feature_meta.json"
CKPT_DIR = Path("checkpoints"); CKPT_DIR.mkdir(parents=True, exist_ok=True)


# Repro (optional)
def set_seed(seed=42):
    import random, os
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# Dataset
class PlayerSeqDataset(Dataset):
    def __init__(self, npz_path: Path, split: str):
        data = np.load(npz_path, allow_pickle=True)
        self.X_num = data["X_num"]        # (N, T, F_num), standardized
        self.X_cat = data["X_cat"]        # (N, T, F_cat)
        self.X_mask= data["X_mask"]       # (N, T)
        self.y     = data["y"].astype(np.float32)

        tr = data["train_idx"].astype(bool)
        va = data["val_idx"].astype(bool)
        if split == "train":
            self.sel = np.where(tr)[0]
        elif split == "val":
            self.sel = np.where(va)[0]
        else:
            raise ValueError("split must be 'train' or 'val'")

    def __len__(self): return len(self.sel)

    def __getitem__(self, i):
        j = self.sel[i]
        return {
            "x_num": torch.from_numpy(self.X_num[j]).float(),
            "x_cat": torch.from_numpy(self.X_cat[j]).long(),
            "mask":  torch.from_numpy(self.X_mask[j]).float(),
            "y":     torch.tensor(self.y[j]).float(),
        }

# Metrics
def mae(pred, target):
    return (pred - target).abs().mean().item()

@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    total_abs, m = 0.0, 0
    for batch in dl:
        x_num = batch["x_num"].to(device, non_blocking=True)
        x_cat = batch["x_cat"].to(device, non_blocking=True)
        mask  = batch["mask"].to(device, non_blocking=True)
        y     = batch["y"].to(device, non_blocking=True)
        yhat  = model(x_num, x_cat, mask)
        total_abs += (yhat - y).abs().sum().item()
        m        += y.numel()
    return total_abs / m


# Warmup + Cosine scheduler
def make_warmup_cosine(optimizer, warmup_steps, total_steps, min_lr=1e-6):
    base_lrs = [g['lr'] for g in optimizer.param_groups]
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# Train
def main():
    feat = json.load(open(META_PATH))
    num_feats = len(feat["num_feats"])
    team_vocab = len(feat["team_index"])
    opp_vocab  = len(feat["opp_index"])
    pos_vocab  = len(feat["pos_index"])

    train_ds = PlayerSeqDataset(NPZ_PATH, "train")
    val_ds   = PlayerSeqDataset(NPZ_PATH, "val")
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FantasyTransformer(
        num_feats=num_feats,
        cat_vocab_sizes=[team_vocab, opp_vocab, pos_vocab],
        d_model=192, nhead=4, num_layers=3, ff_mult=4, dropout=0.1,
        use_pos_encoding=True,
    ).to(device)

    base_lr = 2e-4
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(beta=1.0)  # Huber
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    epochs = 12
    total_steps = epochs * max(1, len(train_dl))
    warmup_steps = 500 if total_steps > 1000 else int(0.05 * total_steps)
    sched = make_warmup_cosine(opt, warmup_steps, total_steps)

    best_val = float("inf")
    global_step = 0

    for ep in range(1, epochs + 1):
        model.train()
        run_abs, n_obs = 0.0, 0

        for batch in train_dl:
            x_num = batch["x_num"].to(device, non_blocking=True)
            x_cat = batch["x_cat"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            y_clipped = torch.clamp(y, 0.0, 50.0)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                yhat = model(x_num, x_cat, mask)
                loss = loss_fn(yhat, y_clipped)

            scaler.scale(loss).backward()
            # gradient clipping
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            sched.step()
            global_step += 1

            run_abs += (yhat.detach() - y).abs().sum().item()
            n_obs   += y.numel()

        tr_mae = run_abs / max(1, n_obs)
        val_mae = evaluate(model, val_dl, device)
        print(f"Epoch {ep:02d} | Train MAE: {tr_mae:.3f} | Val MAE: {val_mae:.3f} | LR: {sched.get_last_lr()[0]:.2e}")

        # Save best
        if val_mae < best_val:
            best_val = val_mae
            ckpt_path = CKPT_DIR / "fantasy_transformer_best.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "meta": {
                    "num_feats": num_feats,
                    "team_vocab": team_vocab,
                    "opp_vocab": opp_vocab,
                    "pos_vocab": pos_vocab
                }
            }, ckpt_path)
            print(f"saved {ckpt_path} (best_val={best_val:.3f})")

if __name__ == "__main__":
    main()
