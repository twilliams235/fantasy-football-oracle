import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nflreadpy as nfl

# -----------------------------
# Config
# -----------------------------
YEARS = list(range(2017, 2026))  # or [2024, 2025] if you want smaller
OUT_DIR = Path("figures_nflreadpy")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Choose a player name to plot weekly time series for (if found)
EXAMPLE_PLAYER = "Breece Hall"  # change this to whoever you want

# -----------------------------
# Helper: ensure PPR exists
# -----------------------------
def ensure_ppr(df: pd.DataFrame) -> pd.DataFrame:
    if "fantasy_points_ppr" in df.columns:
        return df

    # try to build it from common stat columns
    for col in ["receptions", "rushing_yards", "receiving_yards",
                "passing_yards", "rushing_tds", "receiving_tds",
                "passing_tds", "interceptions"]:
        if col not in df.columns:
            df[col] = 0.0

    df["fantasy_points_ppr"] = (
        df["receptions"].fillna(0)
        + df["rushing_yards"].fillna(0) / 10.0
        + df["receiving_yards"].fillna(0) / 10.0
        + df["passing_yards"].fillna(0) / 25.0
        + 6.0 * (df["rushing_tds"].fillna(0) + df["receiving_tds"].fillna(0))
        + 4.0 * df["passing_tds"].fillna(0)
        - 2.0 * df["interceptions"].fillna(0)
    )
    return df

# -----------------------------
# Load data from nflreadpy
# -----------------------------
print("Loading player stats from nflreadpy...")
weekly_pl = nfl.load_player_stats(YEARS)
weekly = weekly_pl.to_pandas()

print("Loading rosters from nflreadpy...")
rosters_pl = nfl.load_rosters(YEARS)
rosters = rosters_pl.to_pandas()

# Normalize IDs if needed
if "player_id" not in weekly.columns and "gsis_id" in weekly.columns:
    weekly = weekly.rename(columns={"gsis_id": "player_id"})
if "player_id" not in rosters.columns and "gsis_id" in rosters.columns:
    rosters = rosters.rename(columns={"gsis_id": "player_id"})

# Attach positions (QB/RB/WR/TE)
if "position" not in weekly.columns and "position" in rosters.columns:
    weekly = weekly.merge(
        rosters[["player_id", "position"]],
        on="player_id",
        how="left"
    )

# Filter to skill positions only
KEEP_POS = {"QB", "RB", "WR", "TE"}
weekly = weekly[weekly["position"].isin(KEEP_POS)].copy()

# Ensure season/week are ints
weekly["season"] = weekly["season"].astype(int)
weekly["week"] = weekly["week"].astype(int)

# Add PPR if missing
weekly = ensure_ppr(weekly)

print("Finished loading. Shape:", weekly.shape)

# -----------------------------
# 1. Distribution of fantasy points (all players)
# -----------------------------
def plot_fp_distribution(df: pd.DataFrame, out_dir: Path):
    plt.figure(figsize=(8,5))
    df["fantasy_points_ppr"].hist(bins=40)
    plt.xlabel("Fantasy Points (PPR)")
    plt.ylabel("Count")
    plt.title("Distribution of Weekly Fantasy Points (PPR)")
    plt.tight_layout()
    out_path = out_dir / "fp_ppr_distribution.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved:", out_path)

# -----------------------------
# 2. Fantasy points by position (boxplot)
# -----------------------------
def plot_fp_by_position(df: pd.DataFrame, out_dir: Path):
    plt.figure(figsize=(8,5))
    # Use a simple boxplot grouped by position
    data = [df.loc[df["position"] == pos, "fantasy_points_ppr"] for pos in sorted(df["position"].unique())]
    positions = sorted(df["position"].unique())
    plt.boxplot(data, labels=positions, showfliers=False)
    plt.xlabel("Position")
    plt.ylabel("Fantasy Points (PPR)")
    plt.title("Weekly Fantasy Points by Position")
    plt.tight_layout()
    out_path = out_dir / "fp_ppr_by_position.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved:", out_path)

# -----------------------------
# 3. Example player weekly time series
# -----------------------------
def plot_player_timeseries(df: pd.DataFrame, player_name: str, out_dir: Path):
    if "player_name" not in df.columns:
        print("No player_name column in df; skipping player time series plot.")
        return

    sub = df[df["player_name"] == player_name].copy()
    if sub.empty:
        print(f"No rows found for player '{player_name}'; picking a random player instead.")
        # choose a random player
        any_player = df["player_name"].dropna().sample(1).iloc[0]
        print("Random player:", any_player)
        sub = df[df["player_name"] == any_player].copy()

    sub = sub.sort_values(["season", "week"])
    # Build a continuous index as "season-week" for plotting
    sub["season_week"] = sub["season"].astype(str) + "-W" + sub["week"].astype(str)

    plt.figure(figsize=(10,4))
    plt.plot(sub["season_week"], sub["fantasy_points_ppr"], marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Fantasy Points (PPR)")
    plt.title(f"Weekly Fantasy Points – {sub['player_name'].iloc[0]}")
    plt.tight_layout()
    out_path = out_dir / "example_player_timeseries.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved:", out_path)

# -----------------------------
# 4. Team pass rate over time (example)
# -----------------------------
def plot_team_pass_rate(df: pd.DataFrame, out_dir: Path):
    # Some nflreadpy schemas may use "attempts" or similar – handle lightly
    # If attempts doesn't exist, approximate via passing_yards>0
    if "attempts" in df.columns:
        df["pass_att"] = df["attempts"]
    else:
        # crude approximation
        df["pass_att"] = (df["passing_yards"] > 0).astype(int)

    if "carries" not in df.columns:
        df["carries"] = 0

    # team column present?
    team_col_candidates = [c for c in ["team", "recent_team", "team_abbr"] if c in df.columns]
    if not team_col_candidates:
        print("No team column found; skipping team pass rate plot.")
        return
    team_col = team_col_candidates[0]

    team_week = df.groupby(["season", "week", team_col], as_index=False).agg(
        pass_att=("pass_att", "sum"),
        rush_att=("carries", "sum"),
    )
    team_week["plays"] = team_week["pass_att"] + team_week["rush_att"]
    team_week["pass_rate"] = team_week["pass_att"] / team_week["plays"].replace(0, np.nan)

    # pick a random team
    team = team_week[team_col].dropna().sample(1).iloc[0]
    sub = team_week[team_week[team_col] == team].copy().sort_values(["season", "week"])
    sub["season_week"] = sub["season"].astype(str) + "-W" + sub["week"].astype(str)

    plt.figure(figsize=(10,4))
    plt.plot(sub["season_week"], sub["pass_rate"], marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Pass Rate")
    plt.title(f"Team Pass Rate Over Time – {team}")
    plt.tight_layout()
    out_path = out_dir / "example_team_pass_rate.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved:", out_path)

# -----------------------------
# Run all visualizations
# -----------------------------
if __name__ == "__main__":
    plot_fp_distribution(weekly, OUT_DIR)
    plot_fp_by_position(weekly, OUT_DIR)
    plot_player_timeseries(weekly, EXAMPLE_PLAYER, OUT_DIR)
    plot_team_pass_rate(weekly, OUT_DIR)

    print("Done. Check the 'figures_nflreadpy' folder for generated PNGs.")
