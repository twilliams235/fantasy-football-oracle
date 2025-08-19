# pip install -U nfl_data_py pandas numpy pyarrow torch
import os
import json
import numpy as np
import pandas as pd
import nfl_data_py as nfl
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset

# -------------------------
# Config
# -------------------------
YEARS = list(range(2017, 2025))           # adjust
SEQ_LEN = 10                               # weeks of history per sample
MIN_WEEKS = 3                              # require at least this much past data
KEEP_POS = {"QB","RB","WR","TE"}          # fantasy positions to keep
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# 1) Load & normalize
# -------------------------
weekly = nfl.import_weekly_data(YEARS)
rosters = nfl.import_seasonal_rosters(YEARS)
ids_x = nfl.import_ids()

# normalize roster name/team columns across versions
name_col = "player_name" if "player_name" in rosters.columns else ("full_name" if "full_name" in rosters.columns else None)
team_cols = [c for c in ["team","recent_team","team_abbr"] if c in rosters.columns]
team_col = team_cols[0] if team_cols else None

ro_map_cols = ["player_id"]
if name_col: ro_map_cols.append(name_col)
if team_col: ro_map_cols.append(team_col)
if "position" in rosters.columns: ro_map_cols.append("position")

ro_map = rosters[ro_map_cols].drop_duplicates("player_id").copy()
if name_col: ro_map = ro_map.rename(columns={name_col:"player_name"})
if team_col: ro_map = ro_map.rename(columns={team_col:"team"})

# ensure standard fields exist in weekly
if "recent_team" in weekly.columns and "team" not in weekly.columns:
    weekly = weekly.rename(columns={"recent_team":"team"})
if "player_name" not in weekly.columns and "name" in weekly.columns:
    weekly = weekly.rename(columns={"name":"player_name"})

# Keep only skill positions
if "position" not in weekly.columns:
    weekly = weekly.merge(ro_map[["player_id","position"]], on="player_id", how="left")
weekly = weekly[weekly["position"].isin(KEEP_POS)]

# opponent column name harmonization
opp_col = "opponent_team" if "opponent_team" in weekly.columns else ("opponent" if "opponent" in weekly.columns else None)
if opp_col is None:
    raise RuntimeError("Opponent column not found in weekly data.")

weekly["season"] = weekly["season"].astype(int)
weekly["week"]   = weekly["week"].astype(int)

# -------------------------
# 2) Basic feature tables
# -------------------------
# A) Player-week core stats (subset; add more later as needed)
num_keep = [
    "passing_yards","passing_tds","interceptions",
    "rushing_yards","rushing_tds","carries",
    "receiving_yards","receiving_tds","receptions","targets",
]
for c in num_keep:
    if c not in weekly.columns:
        weekly[c] = np.nan

# fantasy points (PPR) if available, else compute basic PPR
if "fantasy_points_ppr" not in weekly.columns:
    weekly["fantasy_points_ppr"] = (
        weekly["receptions"].fillna(0)
        + weekly["rushing_yards"].fillna(0)/10
        + weekly["receiving_yards"].fillna(0)/10
        + weekly["passing_yards"].fillna(0)/25
        + 6*(weekly["rushing_tds"].fillna(0) + weekly["receiving_tds"].fillna(0))
        + 4*weekly["passing_tds"].fillna(0)
        - 2*weekly["interceptions"].fillna(0)
        # ignoring fumbles lost / bonuses for simplicity
    )

# attach canonical name/position/team from roster map
weekly = weekly.merge(ro_map[["player_id","player_name","position","team"]], on="player_id", how="left", suffixes=("","_ro"))
weekly["team"] = weekly["team"].fillna(weekly.get("team_ro"))

# B) Team-week context: pass rate
agg = weekly.groupby(["season","week","team"], as_index=False).agg(
    pass_att=("attempts","sum") if "attempts" in weekly.columns else ("passing_yards","count"),
    rush_att=("carries","sum"),
)
agg["plays"] = agg["pass_att"] + agg["rush_att"]
agg["pass_rate"] = agg["pass_att"] / agg["plays"].replace(0,np.nan)
team_week = agg[["season","week","team","pass_rate"]]

# C) Defense-vs-position (fantasy points allowed in that week)
dvp = (weekly.groupby(["season","week",opp_col,"position"], as_index=False)
              .agg(fp_allowed=("fantasy_points_ppr","sum")))
dvp_wide = dvp.pivot_table(index=["season","week",opp_col], columns="position", values="fp_allowed", fill_value=0).reset_index()
dvp_wide.columns = ["season","week","team_def"] + [f"def_fp_allowed_{c}" for c in dvp_wide.columns[3:]]

# -------------------------
# 3) Join + rolling (leak-free)
# -------------------------
X = (weekly.merge(team_week, on=["season","week","team"], how="left")
            .merge(dvp_wide, left_on=["season","week",opp_col],
                          right_on=["season","week","team_def"], how="left"))

# Sort for rolling windows
X = X.sort_values(["player_id","season","week"])

# Rolling recency features (shift BEFORE rolling to avoid leakage)
by_pid = X.groupby("player_id", group_keys=False)
base_roll = ["fantasy_points_ppr","targets","carries","receptions","rushing_yards","receiving_yards"]
for col in base_roll:
    X[f"{col}_lag1"] = by_pid[col].shift(1)
    X[f"{col}_ma3"]  = by_pid[col].shift(1).rolling(3, min_periods=1).mean()
    X[f"{col}_ma5"]  = by_pid[col].shift(1).rolling(5, min_periods=1).mean()

# Label: next-week PPR
X["y_next_ppr"] = by_pid["fantasy_points_ppr"].shift(-1)

# require some history
X["hist_weeks"] = by_pid.cumcount()
X = X[X["hist_weeks"] >= MIN_WEEKS]

# -------------------------
# 4) Categorical encoders
# -------------------------
def make_indexer(values: pd.Series) -> Dict[str,int]:
    uniq = ["<PAD>"] + sorted(set([str(v) for v in values.dropna().unique()]))
    return {v:i for i,v in enumerate(uniq)}

team_index = make_indexer(X["team"])
opp_index  = make_indexer(X[opp_col])
pos_index  = make_indexer(X["position"])

def map_idx(series: pd.Series, mapping: Dict[str,int]) -> np.ndarray:
    return series.fillna("<PAD>").astype(str).map(lambda x: mapping.get(x, 0)).astype("int32").values

X["team_id"] = map_idx(X["team"], team_index)
X["opp_id"]  = map_idx(X[opp_col], opp_index)
X["pos_id"]  = map_idx(X["position"], pos_index)

# -------------------------
# 5) Select final feature set
# -------------------------
num_feats = [
    # team/matchup
    "pass_rate",
    "def_fp_allowed_QB","def_fp_allowed_RB","def_fp_allowed_WR","def_fp_allowed_TE",
    # recency
    "fantasy_points_ppr_lag1","fantasy_points_ppr_ma3","fantasy_points_ppr_ma5",
    "targets_lag1","targets_ma3","targets_ma5",
    "carries_lag1","carries_ma3","carries_ma5",
    "receptions_lag1","receptions_ma3","receptions_ma5",
    "rushing_yards_lag1","rushing_yards_ma3","rushing_yards_ma5",
    "receiving_yards_lag1","receiving_yards_ma3","receiving_yards_ma5",
]
for c in num_feats:
    if c not in X.columns:
        X[c] = np.nan
num_feats = [c for c in num_feats if c in X.columns]

cat_feats = ["team_id","opp_id","pos_id"]

keep_cols = ["season","week","player_id","player_name","position","team",opp_col,"y_next_ppr"] + num_feats + cat_feats
X = X[keep_cols].reset_index(drop=True)

# -------------------------
# 6) Build sequences per (player_id, week)
# -------------------------
def build_sequences(df: pd.DataFrame, seq_len: int) -> Tuple[np.ndarray,np.ndarray,np.ndarray,pd.DataFrame]:
    """
    Returns:
      X_num:  [N, seq_len, F_num]
      X_cat:  [N, seq_len, F_cat]
      mask:   [N, seq_len]  (1 = valid, 0 = pad)
      meta:   DataFrame with keys for each sample
    """
    rows = []
    Xn, Xc, Mm = [], [], []
    for pid, g in df.groupby("player_id"):
        g = g.sort_values(["season","week"]).reset_index(drop=True)
        # slide over each week (use all but the last row that has label)
        for i in range(len(g)):
            # stop if label missing
            if pd.isna(g.loc[i, "y_next_ppr"]): 
                continue
            # take up to seq_len rows ending at i (i inclusive)
            start = max(0, i - seq_len + 1)
            hist  = g.iloc[start:i+1]
            # pad on the left if needed
            pad = seq_len - len(hist)
            num = hist[num_feats].to_numpy(dtype=np.float32)
            cat = hist[cat_feats].to_numpy(dtype=np.int64)
            if pad > 0:
                num = np.vstack([np.zeros((pad, num.shape[1]), dtype=np.float32), num])
                cat = np.vstack([np.zeros((pad, cat.shape[1]), dtype=np.int64),   cat])
            mask = np.zeros((seq_len,), dtype=np.float32); mask[pad:] = 1.0
            Xn.append(num); Xc.append(cat); Mm.append(mask)

            rows.append({
                "player_id": pid,
                "player_name": g.loc[i,"player_name"],
                "position": g.loc[i,"position"],
                "team": g.loc[i,"team"],
                "season": g.loc[i,"season"],
                "week": g.loc[i,"week"],
                "y_next_ppr": g.loc[i,"y_next_ppr"]
            })

    meta = pd.DataFrame(rows)
    return np.stack(Xn), np.stack(Xc), np.stack(Mm), meta

X_num, X_cat, X_mask, META = build_sequences(X, SEQ_LEN)

# standardize numeric features (fit on train later; here quick global z-score)
mu = np.nanmean(X_num, axis=(0,1))
sd = np.nanstd(X_num, axis=(0,1)) + 1e-6
X_num = (np.nan_to_num(X_num, nan=0.0) - mu) / sd
X_cat = np.nan_to_num(X_cat, nan=0).astype(np.int64)

# Targets
y = META["y_next_ppr"].values.astype(np.float32)

# -------------------------
# 7) Train/Val split (leak-free by season)
# -------------------------
VAL_SEASON = 2024
train_idx = META["season"] < VAL_SEASON
val_idx   = META["season"] == VAL_SEASON

np.savez_compressed(
    os.path.join(OUT_DIR, "player_sequences_npz.npz"),
    X_num=X_num, X_cat=X_cat, X_mask=X_mask, y=y,
    train_idx=train_idx.values, val_idx=val_idx.values
)

META.to_parquet(os.path.join(OUT_DIR, "meta.parquet"), index=False)

# Save feature metadata (for model input sizes & embedding vocab)
feat_meta = {
    "num_feats": num_feats,
    "cat_feats": cat_feats,
    "seq_len": SEQ_LEN,
    "team_index": team_index,
    "opp_index": opp_index,
    "pos_index": pos_index,
    "num_mu": mu.tolist(),
    "num_sd": sd.tolist(),
}
json.dump(feat_meta, open(os.path.join(OUT_DIR,"feature_meta.json"), "w"))

print("Saved:",
      os.path.join(OUT_DIR, "player_sequences_npz.npz"),
      os.path.join(OUT_DIR, "meta.parquet"))

# -------------------------
# 8) PyTorch Dataset (import from this file or copy to your training script)
# -------------------------
class PlayerWeekSequenceDataset(Dataset):
    def __init__(self, npz_path: str, split: str = "train"):
        data = np.load(npz_path, allow_pickle=True)
        self.X_num = data["X_num"]    # [N, T, F_num]
        self.X_cat = data["X_cat"]    # [N, T, F_cat]
        self.X_mask= data["X_mask"]   # [N, T]
        self.y     = data["y"]        # [N]
        train_idx  = data["train_idx"].astype(bool)
        val_idx    = data["val_idx"].astype(bool)
        idx = train_idx if split=="train" else val_idx
        self.sel = np.where(idx)[0]

    def __len__(self): return len(self.sel)

    def __getitem__(self, i):
        j = self.sel[i]
        return {
            "x_num": torch.from_numpy(self.X_num[j]),   # (T, F_num)
            "x_cat": torch.from_numpy(self.X_cat[j]),   # (T, F_cat)
            "mask":  torch.from_numpy(self.X_mask[j]),  # (T,)
            "y":     torch.tensor(self.y[j])            # ()
        }
