import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = Path("data/processed")
NPZ_PATH = DATA_DIR / "player_sequences_npz.npz"
META_PATH = DATA_DIR / "meta.parquet"
FEAT_META_PATH = DATA_DIR / "feature_meta.json"

def main():
    # -----------------------------
    # Load data
    # -----------------------------
    arr = np.load(NPZ_PATH, allow_pickle=True)
    meta = pd.read_parquet(META_PATH)
    feat = json.load(open(FEAT_META_PATH))

    X_num  = arr["X_num"]        # [N, T, F_num]  (standardized)
    X_mask = arr["X_mask"]       # [N, T]
    num_feats = feat["num_feats"]
    seq_len   = feat["seq_len"]

    N = X_num.shape[0]
    print(f"Total sequences: {N}, seq_len: {seq_len}, num_feats: {len(num_feats)}")

    # -----------------------------
    # Pick a random sample
    # -----------------------------
    idx = random.randint(0, N - 1)
    sample_num  = X_num[idx]        # (T, F_num)
    sample_mask = X_mask[idx]       # (T,)
    sample_meta = meta.iloc[idx]

    # Valid timesteps (ignore padded leading zeros)
    valid_len = int(sample_mask.sum())
    # The sequences are padded at the *front*, so take the last valid_len steps
    sample_num_valid = sample_num[-valid_len:, :]  # (valid_len, F_num)

    print("Random sample index:", idx)
    print("Player:", sample_meta["player_name"])
    print("Position:", sample_meta["position"])
    print("Team:", sample_meta["team"])
    print("Season/Week:", sample_meta["season"], sample_meta["week"])
    print("Valid history length:", valid_len)

    # -----------------------------
    # Build heatmap of all features
    # -----------------------------
    # Transpose to (features, time)
    data_for_plot = sample_num_valid.T  # (F_num, valid_len)

    plt.figure(figsize=(max(8, valid_len), max(10, len(num_feats) * 0.3)))
    im = plt.imshow(
        data_for_plot,
        aspect="auto",
        interpolation="nearest"
    )

    plt.colorbar(im, label="Standardized feature value")

    # y-axis: feature names
    plt.yticks(range(len(num_feats)), num_feats)
    # x-axis: relative steps, oldest → most recent
    x_labels = [f"t-{valid_len - 1 - i}" for i in range(valid_len)]
    plt.xticks(range(valid_len), x_labels, rotation=0)

    plt.xlabel("Time step in sequence (oldest → most recent)")
    plt.ylabel("Numeric features")
    plt.title(
        f"All Numeric Features in Sequence\n"
        f"{sample_meta['player_name']} – {sample_meta['position']} – "
        f"Season {sample_meta['season']} Week {sample_meta['week']}"
    )

    plt.tight_layout()
    out_path = "random_player_all_features_heatmap.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved heatmap to {out_path}")

if __name__ == "__main__":
    main()
