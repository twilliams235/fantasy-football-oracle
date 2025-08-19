import json
import numpy as np
import pandas as pd

npz = np.load("data/processed/player_sequences_npz.npz", allow_pickle=True)
meta = pd.read_parquet("data/processed/meta.parquet")
feat = json.load(open("data/processed/feature_meta.json"))

X_num = npz["X_num"]     # shape [N, T, F_num]
X_cat = npz["X_cat"]     # shape [N, T, F_cat]
X_mask = npz["X_mask"]   # shape [N, T]
y = npz["y"]
num_feats = feat["num_feats"]      # names of numeric features
cat_feats = feat["cat_feats"]      # ["team_id","opp_id","pos_id"]

print("Shapes:", X_num.shape, X_cat.shape, X_mask.shape)
print("Numeric features:", num_feats)




