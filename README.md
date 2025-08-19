# Fantasy Football Score Predictions w/ a Transformer

## Preprocessed Data

After running the preprocessing pipeline (make_player_sequences.py), the following artifacts are generated in data/processed/:

# meta.parquet

A lightweight, human-readable index of training samples.
Each row corresponds to one player-week sample (i.e., the state of a player up through Week k, with the label = fantasy points scored in Week k+1).

**Columns:**

- player_id — stable nflverse player identifier

- player_name — display name (may be abbreviated, e.g. C.McCaffrey)

- position — fantasy-relevant position (QB, RB, WR, TE)

- team — player’s team during that week (abbreviation)

- season — NFL season year

- week — NFL week number (1–22 including playoffs if available)

- y_next_ppr — label = PPR fantasy points scored by the player in the following week

**Notes:**

- meta.parquet does not contain input features like targets/carries directly.

- It’s mainly for filtering/searching samples by player/team/week and joining predictions back to human-readable IDs.

# player_sequences_npz.npz

A compressed NumPy archive containing the actual model inputs and labels.
This is what you load for training or evaluation.

**Arrays inside:**

- X_num — numeric features; shape [N, T, F_num]

    - N = number of samples

    - T = sequence length (default = 10 weeks)

    - F_num = number of numeric features (rolling/lagged stats, team context, matchup stats, etc.)

    - Values are standardized (z-scored) using global mean/std stored in feature_meta.json.

- X_cat — categorical features; shape [N, T, F_cat]

    - Integer IDs for team, opponent, position (vocabularies also stored in feature_meta.json).

- X_mask — attention mask; shape [N, T]

    - 1 = valid timestep, 0 = padding (for players with <T weeks of history).

- y — regression targets; shape [N]

    - PPR fantasy points scored in the following week.

- train_idx / val_idx — boolean masks of length N

    - Used to split samples into training vs. validation (by season).

# Companion file: feature_meta.json

Stores metadata describing the NPZ arrays:

- num_feats — ordered list of numeric feature names (e.g., receptions_lag1, receptions_ma3, carries_ma5, pass_rate, def_fp_allowed_WR, etc.).

- cat_feats — names of categorical feature slots (team_id, opp_id, pos_id).

- team_index, opp_index, pos_index — dictionaries mapping team/opponent/position strings → integer IDs.

- num_mu, num_sd — arrays of means/stds used to z-score numeric features.