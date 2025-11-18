import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("Yahoo_scores.csv")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"{CSV_PATH} not found!")

    df = pd.read_csv(CSV_PATH)

    required_cols = [
        "player_name",
        "season",
        "week",
        "yahoo_proj_ppr",
        "actual_ppr",
        "model_proj_ppr",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column in CSV: {c}")

    df["yahoo_proj_ppr"] = pd.to_numeric(df["yahoo_proj_ppr"], errors="coerce")
    df["actual_ppr"]     = pd.to_numeric(df["actual_ppr"], errors="coerce")
    df["model_proj_ppr"] = pd.to_numeric(df["model_proj_ppr"], errors="coerce")

    y_true   = df["actual_ppr"].values
    y_yahoo  = df["yahoo_proj_ppr"].values
    y_model  = df["model_proj_ppr"].values

    overall_mae_yahoo = mae(y_true, y_yahoo)
    overall_mae_model = mae(y_true, y_model)
    overall_rmse_yahoo = rmse(y_true, y_yahoo)
    overall_rmse_model = rmse(y_true, y_model)

    print("=== Overall accuracy (all CSV rows) ===")
    print(f"Yahoo MAE:  {overall_mae_yahoo:.3f}")
    print(f"Model MAE:  {overall_mae_model:.3f}")
    print(f"Yahoo RMSE: {overall_rmse_yahoo:.3f}")
    print(f"Model RMSE: {overall_rmse_model:.3f}")
    print()

    rows = []
    for player, sub in df.groupby("player_name"):
        y_true_p   = sub["actual_ppr"].values
        y_yahoo_p  = sub["yahoo_proj_ppr"].values
        y_model_p  = sub["model_proj_ppr"].values
        rows.append({
            "player_name": player,
            "mae_yahoo": mae(y_true_p, y_yahoo_p),
            "mae_model": mae(y_true_p, y_model_p),
            "n_samples": len(sub),
        })

    per_player = pd.DataFrame(rows).sort_values("player_name")
    print("=== Per-player MAE ===")
    print(per_player.to_string(index=False))
    print()

    per_player.to_csv("per_player_mae_yahoo_vs_model.csv", index=False)
    print("Saved per-player MAE table to per_player_mae_yahoo_vs_model.csv")

    x = np.arange(len(per_player))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, per_player["mae_yahoo"], width, label="Yahoo")
    plt.bar(x + width/2, per_player["mae_model"], width, label="Model")
    plt.xticks(x, per_player["player_name"], rotation=45, ha="right")
    plt.ylabel("MAE (PPR)")
    plt.title("MAE by Player – Yahoo vs Model")
    plt.legend()
    plt.tight_layout()

    out_path = FIG_DIR / "mae_by_player_yahoo_vs_model.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved bar chart to {out_path}")

if __name__ == "__main__":
    main()
