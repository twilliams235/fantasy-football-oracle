# predict_players.py
import argparse, json, numpy as np, pandas as pd, torch
from pathlib import Path
from models.fantasy_transformer import FantasyTransformer

DATA_DIR = Path("data/processed")
NPZ_PATH = DATA_DIR / "player_sequences_npz.npz"
META_PATH = DATA_DIR / "meta.parquet"
FEAT_PATH = DATA_DIR / "feature_meta.json"
CKPT_PATH = Path("checkpoints/fantasy_transformer_best.pt")

def load_model(device):
    feat = json.load(open(FEAT_PATH))
    model = FantasyTransformer(
        num_feats=len(feat["num_feats"]),
        cat_vocab_sizes=[len(feat["team_index"]), len(feat["opp_index"]), len(feat["pos_index"])],
        d_model=192, nhead=4, num_layers=3, ff_mult=4, dropout=0.1, use_pos_encoding=True
    ).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

@torch.no_grad()
def predict_for_names(query: str, split="val", device="cpu"):
    data = np.load(NPZ_PATH, allow_pickle=True)
    meta = pd.read_parquet(META_PATH).reset_index(drop=True)

    idx = data[f"{split}_idx"].astype(bool)
    ids = np.where(idx)[0]
    sub = meta.loc[ids].copy()

    # name match (case-insensitive substring)
    mask = sub["player_name"].str.contains(query, case=False, na=False)
    sel_ids = ids[mask.values]
    if len(sel_ids) == 0:
        return pd.DataFrame()

    X_num = torch.from_numpy(data["X_num"][sel_ids]).float().to(device)
    X_cat = torch.from_numpy(data["X_cat"][sel_ids]).long().to(device)
    X_mask= torch.from_numpy(data["X_mask"][sel_ids]).float().to(device)

    model = load_model(device)
    yhat = model(X_num, X_cat, X_mask).cpu().numpy()

    out = meta.loc[sel_ids, ["player_name","position","team","season","week","y_next_ppr"]].copy()
    out["pred_ppr"] = yhat
    return out.sort_values(["season","week"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="substring to match, e.g. 'McCaffrey'")
    ap.add_argument("--split", default="val", choices=["train","val"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = predict_for_names(args.name, split=args.split, device=device)
    if df.empty:
        print(f"No matches for '{args.name}' in {args.split} split.")
    else:
        print(df.to_string(index=False))

