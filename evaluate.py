# evaluate.py
import argparse, json, math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.fantasy_transformer import FantasyTransformer

DATA_DIR   = Path("data/processed")
NPZ_PATH   = DATA_DIR / "player_sequences_npz.npz"
META_PATH  = DATA_DIR / "meta.parquet"
FEAT_PATH  = DATA_DIR / "feature_meta.json"
CKPT_PATH  = Path("checkpoints/fantasy_transformer_best.pt")
OUT_DIR    = Path("data/eval"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------- helpers -----------------------
def mae(a,b): return np.mean(np.abs(a-b))
def rmse(a,b): return math.sqrt(np.mean((a-b)**2))

def spearman(x, y):
    # rank then Pearson on ranks (no SciPy dependency)
    xr = pd.Series(x).rank(method="average")
    yr = pd.Series(y).rank(method="average")
    return xr.corr(yr)

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
def batched_preds(model, X_num, X_cat, X_mask, device="cpu", bs=1024):
    n = X_num.shape[0]
    out = np.empty((n,), dtype=np.float32)
    for i in range(0, n, bs):
        j = slice(i, min(i+bs, n))
        xn = torch.from_numpy(X_num[j]).float().to(device)
        xc = torch.from_numpy(X_cat[j]).long().to(device)
        xm = torch.from_numpy(X_mask[j]).float().to(device)
        out[j] = model(xn, xc, xm).detach().cpu().numpy().astype(np.float32)
    return out

def add_baseline(df, kind):
    if kind is None or kind == "none":
        df["baseline"] = np.nan
        return df
    if kind == "last":
        # previous week's fantasy points for that player
        df = df.sort_values(["player_id","season","week"])
        df["baseline"] = df.groupby("player_id")["fantasy_points_ppr"].shift(1)
    elif kind == "ma3":
        df = df.sort_values(["player_id","season","week"])
        df["baseline"] = (
            df.groupby("player_id")["fantasy_points_ppr"].shift(1).rolling(3, min_periods=1).mean()
        )
    else:
        raise ValueError(f"Unknown baseline: {kind}")
    return df

def merge_espn(df, csv_path):
    if not csv_path: 
        df["espn_proj"] = np.nan
        return df
    espn = pd.read_csv(csv_path)
    # expected columns: season, week, player_name, proj_ppr
    cols = {"Proj","Projection","proj","ppr","ppr_proj","pred","prediction"}
    if "proj_ppr" not in espn.columns:
        # attempt to auto-detect a projection column
        cand = [c for c in espn.columns if c.lower() in {c.lower() for c in cols}]
        if cand:
            espn = espn.rename(columns={cand[0]: "proj_ppr"})
        else:
            raise ValueError("ESPN CSV must contain 'proj_ppr' column (or obvious equivalent).")
    on = ["season","week","player_name"]
    espn = espn[on + ["proj_ppr"]].copy()
    return df.merge(espn, on=on, how="left").rename(columns={"proj_ppr":"espn_proj"})

# ----------------------- main -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train","val"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--baseline", default="ma3", choices=["none","last","ma3"])
    ap.add_argument("--espn-csv", default="", help="Optional CSV with columns: season,week,player_name,proj_ppr")
    args = ap.parse_args()

    device = args.device
    data = np.load(NPZ_PATH, allow_pickle=True)
    meta = pd.read_parquet(META_PATH).reset_index(drop=True)

    idx = data[f"{args.split}_idx"].astype(bool)
    sel = np.where(idx)[0]

    X_num = data["X_num"][sel]
    X_cat = data["X_cat"][sel]
    X_mask= data["X_mask"][sel]
    y     = data["y"][sel].astype(np.float32)

    model = load_model(device)
    pred = batched_preds(model, X_num, X_cat, X_mask, device=device, bs=args.batch_size)

    # row-wise frame for analysis
    df = meta.loc[sel, ["player_id","player_name","position","team","season","week","y_next_ppr"]].copy()
    df = df.rename(columns={"y_next_ppr":"actual"})
    # bring original last-week value for baselines
    # (merge minimal column from meta by index if available)
    if "fantasy_points_ppr" in meta.columns:
        df["fantasy_points_ppr"] = meta.loc[sel, "fantasy_points_ppr"].values
    else:
        df["fantasy_points_ppr"] = np.nan

    df["pred"] = pred

    # baselines
    df = add_baseline(df, args.baseline)
    df = merge_espn(df, args.espn_csv) if args.espn_csv else df.assign(espn_proj=np.nan)

    # --- summary metrics ---
    overall = {
        "split": args.split,
        "N": int(df.shape[0]),
        "MAE": float(mae(df["pred"], df["actual"])),
        "RMSE": float(rmse(df["pred"], df["actual"])),
        "Spearman": float(spearman(df["pred"], df["actual"])),
    }
    # per-position MAE
    by_pos = (
        df.groupby("position")
          .apply(lambda g: pd.Series({
              "N": g.shape[0],
              "MAE": mae(g["pred"], g["actual"]),
              "RMSE": rmse(g["pred"], g["actual"]),
              "Spearman": spearman(g["pred"], g["actual"]),
          }))
          .reset_index()
    )

    # weekly within-position rank correlation (lineup usefulness)
    wk_pos = (
        df.groupby(["season","week","position"])
          .apply(lambda g: spearman(g["pred"], g["actual"]))
          .reset_index(name="spearman_wkpos")
    )
    lineup_spearman_mean = float(wk_pos["spearman_wkpos"].mean())

    # calibration (prediction deciles)
    dec = pd.qcut(df["pred"], q=10, duplicates="drop")
    calib = (
        df.groupby(dec)
          .agg(mean_pred=("pred","mean"), mean_actual=("actual","mean"), count=("actual","size"))
          .reset_index(drop=True)
    )

    # baselines/ESPN comparisons (if available)
    comps = []
    if df["baseline"].notna().any():
        comps.append({
            "name":"baseline_"+args.baseline,
            "MAE": float(mae(df["baseline"].dropna(), df.loc[df["baseline"].notna(),"actual"])),
            "RMSE": float(rmse(df["baseline"].dropna(), df.loc[df["baseline"].notna(),"actual"])),
            "Spearman": float(spearman(df.loc[df["baseline"].notna(),"baseline"],
                                       df.loc[df["baseline"].notna(),"actual"])),
        })
    if df["espn_proj"].notna().any():
        mask = df["espn_proj"].notna()
        comps.append({
            "name":"espn",
            "MAE": float(mae(df.loc[mask,"espn_proj"], df.loc[mask,"actual"])),
            "RMSE": float(rmse(df.loc[mask,"espn_proj"], df.loc[mask,"actual"])),
            "Spearman": float(spearman(df.loc[mask,"espn_proj"], df.loc[mask,"actual"])),
        })
    comps = pd.DataFrame(comps) if comps else pd.DataFrame(columns=["name","MAE","RMSE","Spearman"])

    # save artifacts
    df_out_path = OUT_DIR / f"rows_{args.split}.parquet"
    df.to_parquet(df_out_path, index=False)

    calib_path = OUT_DIR / f"calibration_{args.split}.csv"
    calib.to_csv(calib_path, index=False)

    bypos_path = OUT_DIR / f"bypos_{args.split}.csv"
    by_pos.to_csv(bypos_path, index=False)

    comps_path = OUT_DIR / f"comparisons_{args.split}.csv"
    comps.to_csv(comps_path, index=False)

    summary = {
        **overall,
        "LineupSpearmanMean": lineup_spearman_mean,
        "outputs": {
            "rows": str(df_out_path),
            "calibration": str(calib_path),
            "by_position": str(bypos_path),
            "comparisons": str(comps_path),
        }
    }
    (OUT_DIR / f"summary_{args.split}.json").write_text(json.dumps(summary, indent=2))

    # pretty print
    print("\n=== Evaluation Summary ===")
    for k,v in overall.items():
        print(f"{k:>16}: {v}")
    print(f"{'LineupSpearmanMean':>16}: {lineup_spearman_mean:.4f}")
    if not by_pos.empty:
        print("\nPer-Position:")
        print(by_pos.to_string(index=False, formatters={"MAE":"{:.3f}".format, "RMSE":"{:.3f}".format, "Spearman":"{:.3f}".format}))
    if not comps.empty:
        print("\nBaselines / External:")
        print(comps.to_string(index=False, formatters={"MAE":"{:.3f}".format, "RMSE":"{:.3f}".format, "Spearman":"{:.3f}".format}))
    print("\nSaved artifacts to:", OUT_DIR)

if __name__ == "__main__":
    main()
