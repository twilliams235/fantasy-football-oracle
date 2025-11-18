import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

from models.fantasy_transformer import FantasyTransformer


DATA_DIR      = Path("data/processed")
NPZ_PATH      = DATA_DIR / "player_sequences_npz.npz"
META_PATH     = DATA_DIR / "feature_meta.json"
META_PARQUET  = DATA_DIR / "meta.parquet"
CKPT_PATH     = Path("checkpoints") / "fantasy_transformer_best.pt"
FIG_DIR       = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PLAYERS = [
    "Bijan Robinson",
    "Josh Downs",
    "James Cook III",
    "Drake London",
    "George Kittle",
    "David Montgomery",
    "Jordan Addison",
    "Zach Charbonnet",
    "Breece Hall",
    "Jaylen Waddle",
    "Bo Nix",
]
TARGET_PLAYERS_LOWER = [p.lower() for p in TARGET_PLAYERS]
TARGET_SEASON = 2025
MAX_WEEK = 10


class PlayerSeqDataset(Dataset):
    def __init__(self, npz_path: Path, split: str):
        data = np.load(npz_path, allow_pickle=True)
        self.X_num  = data["X_num"]
        self.X_cat  = data["X_cat"]
        self.X_mask = data["X_mask"]
        self.y      = data["y"].astype(np.float32)

        tr = data["train_idx"].astype(bool)
        va = data["val_idx"].astype(bool)
        if split == "train":
            self.sel = np.where(tr)[0]
        elif split == "val":
            self.sel = np.where(va)[0]
        else:
            raise ValueError("split must be 'train' or 'val'")

    def __len__(self): 
        return len(self.sel)

    def __getitem__(self, i):
        j = self.sel[i]
        return {
            "idx":   j,
            "x_num": torch.from_numpy(self.X_num[j]).float(),
            "x_cat": torch.from_numpy(self.X_cat[j]).long(),
            "mask":  torch.from_numpy(self.X_mask[j]).float(),
            "y":     torch.tensor(self.y[j]).float(),
        }

def load_model_and_meta():
    feat = json.load(open(META_PATH))
    num_feats = feat["num_feats"]
    team_vocab = len(feat["team_index"])
    opp_vocab  = len(feat["opp_index"])
    pos_vocab  = len(feat["pos_index"])

    ckpt = torch.load(CKPT_PATH, map_location="cpu")

    model = FantasyTransformer(
        num_feats=len(num_feats),
        cat_vocab_sizes=[team_vocab, opp_vocab, pos_vocab],
        d_model=192,
        nhead=4,
        num_layers=3,
        ff_mult=4,
        dropout=0.1,
        use_pos_encoding=True,
    )
    model.load_state_dict(ckpt["state_dict"])
    return model, feat


def get_2025_player_subset():
    model, feat = load_model_and_meta()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    arr = np.load(NPZ_PATH, allow_pickle=True)
    y_all   = arr["y"].astype(np.float32)
    val_idx = arr["val_idx"].astype(bool)
    val_sel = np.where(val_idx)[0]

    val_ds = PlayerSeqDataset(NPZ_PATH, split="val")
    val_dl = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=2)

    y_pred_list = []
    for batch in val_dl:
        x_num = batch["x_num"].to(device, non_blocking=True)
        x_cat = batch["x_cat"].to(device, non_blocking=True)
        mask  = batch["mask"].to(device, non_blocking=True)
        with torch.no_grad():
            yhat = model(x_num, x_cat, mask)
        y_pred_list.append(yhat.cpu().numpy())
    y_pred_val = np.concatenate(y_pred_list, axis=0)

    y_true_val = y_all[val_idx]

    meta = pd.read_parquet(META_PARQUET)
    meta_val = meta.iloc[val_sel].copy()

    meta_val["player_name_lower"] = meta_val["player_name"].str.lower()
    mask = (
        (meta_val["season"] == TARGET_SEASON) &
        (meta_val["week"] <= MAX_WEEK) &
        (meta_val["player_name_lower"].isin(TARGET_PLAYERS_LOWER))
    )

    meta_sub = meta_val[mask].copy()

    sub_idx = meta_sub.index.to_numpy()

    pos_in_val = meta_val.index.get_indexer(sub_idx)

    y_true_sub = y_true_val[pos_in_val]
    y_pred_sub = y_pred_val[pos_in_val]


    meta_sub["y_true"] = y_true_sub
    meta_sub["y_pred_model"] = y_pred_sub

    return meta_sub, y_true_sub, y_pred_sub


def plot_pred_vs_actual(meta_sub, save_path: Path):
    y_true = meta_sub["y_true"].to_numpy()
    y_pred = meta_sub["y_pred_model"].to_numpy()

    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.6)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual PPR")
    plt.ylabel("Model Predicted PPR")
    plt.title("Predicted vs Actual PPR (2025 Weeks 1–10)\nSelected Players")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_player_lines(meta_sub, save_dir: Path):
    for player in TARGET_PLAYERS:
        sub = meta_sub[meta_sub["player_name"] == player].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("week")
        weeks = sub["week"].to_numpy()
        y_true = sub["y_true"].to_numpy()
        y_pred = sub["y_pred_model"].to_numpy()

        plt.figure()
        plt.plot(weeks, y_true, marker="o", label="Actual PPR")
        plt.plot(weeks, y_pred, marker="o", linestyle="--", label="Model PPR")
        plt.xlabel("Week")
        plt.ylabel("PPR")
        plt.title(f"{player} – 2025 Weeks 1–10")
        plt.legend()
        plt.tight_layout()
        safe_name = player.replace(" ", "_").replace("'", "")
        out_path = save_dir / f"{safe_name}_2025_w1_10.png"


        plt.savefig(out_path, dpi=200)
        plt.close()


def main():
    print("Collecting 2025 weeks 1–10 data for target players...")
    meta_sub, y_true_sub, y_pred_sub = get_2025_player_subset()

    mae = np.mean(np.abs(y_pred_sub - y_true_sub))
    print(f"MAE (model) for selected players, weeks 1–10, 2025: {mae:.3f}")

    plot_pred_vs_actual(meta_sub, FIG_DIR / "pred_vs_actual_selected_players_2025_w1_10.png")
    plot_player_lines(meta_sub, FIG_DIR)

    print("Saved plots in:", FIG_DIR)

if __name__ == "__main__":
    main()

