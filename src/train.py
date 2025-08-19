# train.py
import json
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

# -------- Dataset wrapper --------
class PlayerSeqDataset(Dataset):
    def __init__(self, npz_path: Path, split: str):
        data = np.load(npz_path, allow_pickle=True)
        self.X_num = data["X_num"]        # (N, T, F_num)
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

# -------- Metrics --------
def mae(pred, target):
    return (pred - target).abs().mean().item()

@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    total, m = 0.0, 0
    for batch in dl:
        x_num = batch["x_num"].to(device)
        x_cat = batch["x_cat"].to(device)
        mask  = batch["mask"].to(device)
        y     = batch["y"].to(device)
        yhat  = model(x_num, x_cat, mask)
        total += (yhat - y).abs().sum().item()
        m     += y.numel()
    return total / m

# -------- Training --------
def main():
    # Load feature meta
    feat = json.load(open(META_PATH))
    num_feats = len(feat["num_feats"])
    team_vocab = len(feat["team_index"])
    opp_vocab  = len(feat["opp_index"])
    pos_vocab  = len(feat["pos_index"])

    # Datasets / loaders
    train_ds = PlayerSeqDataset(NPZ_PATH, "train")
    val_ds   = PlayerSeqDataset(NPZ_PATH, "val")
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FantasyTransformer(
        num_feats=num_feats,
        cat_vocab_sizes=[team_vocab, opp_vocab, pos_vocab],
        d_model=128, nhead=4, num_layers=2, ff_mult=4, dropout=0.1, use_pos_encoding=True
    ).to(device)

    # Optim / loss
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    epochs = 10

    for ep in range(1, epochs+1):
        model.train()
        run_loss = 0.0; n = 0
        for batch in train_dl:
            x_num = batch["x_num"].to(device)
            x_cat = batch["x_cat"].to(device)
            mask  = batch["mask"].to(device)
            y     = batch["y"].to(device)

            opt.zero_grad()
            yhat = model(x_num, x_cat, mask)
            loss = loss_fn(yhat, y)
            loss.backward()
            opt.step()

            run_loss += loss.item() * y.numel()
            n += y.numel()

        tr_mae = run_loss / n
        val_mae = evaluate(model, val_dl, device)

        print(f"Epoch {ep:02d} | Train MAE: {tr_mae:.3f} | Val MAE: {val_mae:.3f}")

        # Save best
        if val_mae < best_val:
            best_val = val_mae
            ckpt_path = CKPT_DIR / "fantasy_transformer_best.pt"
            torch.save({"state_dict": model.state_dict(),
                        "meta": {"num_feats": num_feats,
                                 "team_vocab": team_vocab,
                                 "opp_vocab": opp_vocab,
                                 "pos_vocab": pos_vocab}}, ckpt_path)
            print(f"  ↳ saved {ckpt_path} (best_val={best_val:.3f})")

if __name__ == "__main__":
    main()
