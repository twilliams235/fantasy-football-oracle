# Fantasy Football Score Predictions w/ a Transformer

## Preprocessed Data

After running the preprocessing pipeline (make_player_sequences.py), the following artifacts are generated in data/processed/:

## meta.parquet

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

### player_sequences_npz.npz

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

### Companion file: feature_meta.json

Stores metadata describing the NPZ arrays:

- num_feats — ordered list of numeric feature names (e.g., receptions_lag1, receptions_ma3, carries_ma5, pass_rate, def_fp_allowed_WR, etc.).

- cat_feats — names of categorical feature slots (team_id, opp_id, pos_id).

- team_index, opp_index, pos_index — dictionaries mapping team/opponent/position strings integer IDs.

- num_mu, num_sd — arrays of means/stds used to z-score numeric features.

## Model

The core model is a custom **Transformer encoder** designed for sequence regression:

- **Inputs**
  - Numeric features (`X_num`) projected into `d_model` space
  - Categorical features (`X_cat`) embedded separately (team, opponent, position), then concatenated with numeric projection and fused
  - Optional learned positional embeddings (sequence length ≤ 512)
  - Attention mask (`X_mask`) to ignore padded timesteps

- **Architecture**
  - Projection + embedding fusion → Transformer encoder stack
  - Prepend a learned `[CLS]` token to every sequence
  - CLS-pooled output represents the entire sequence
  - Position-aware conditioning: the player’s position (RB/WR/TE/QB) at the last valid timestep is embedded and concatenated with CLS before regression
  - Regression head: LayerNorm → Linear → GELU → Dropout → Linear → scalar PPR prediction

- **Config (default)**
  - `d_model = 192`
  - `nhead = 4`
  - `num_layers = 3`
  - `ff_mult = 4` (feedforward dim = 4 × d_model)
  - `dropout = 0.1`
  - `activation = GELU`

## Training

- **Loss:** Mean Absolute Error (MAE) between predicted and true next-week PPR
- **Optimizer:** AdamW
- **Scheduler:** configurable cosine annealing
- **Batching:** mini-batches drawn from `train_idx`/`val_idx` with masks applied
- **Checkpoints:** best model weights saved to `checkpoints/fantasy_transformer_best.pt` based on validation MAE
- **Validation split:** controlled via `train_idx` / `val_idx` masks (season-based split)

**Example Log**

> Epoch 01 | Train MAE: 7.876 | Val MAE: 6.439 | LR: 5.20e-05

> saved checkpoints/fantasy_transformer_best.pt (best_val=6.439)

> Epoch 02 | Train MAE: 5.537 | Val MAE: 4.975 | LR: 1.04e-04

> saved checkpoints/fantasy_transformer_best.pt (best_val=4.975)

> Epoch 03 | Train MAE: 5.066 | Val MAE: 5.008 | LR: 1.55e-04

> Epoch 04 | Train MAE: 5.037 | Val MAE: 4.983 | LR: 2.00e-04

> Epoch 05 | Train MAE: 5.014 | Val MAE: 4.979 | LR: 1.91e-04

> Epoch 06 | Train MAE: 4.990 | Val MAE: 4.966 | LR: 1.68e-04

> saved checkpoints/fantasy_transformer_best.pt (best_val=4.966)

> Epoch 07 | Train MAE: 4.969 | Val MAE: 4.942 | LR: 1.35e-04

> saved checkpoints/fantasy_transformer_best.pt (best_val=4.942)

> Epoch 08 | Train MAE: 4.938 | Val MAE: 4.948 | LR: 9.76e-05

**Interpretation:**

- The model quickly learns to reduce validation MAE into the ~5 PPR point range.

- CLS pooling with position-aware conditioning stabilizes performance and helps generalization across RB/WR/TE/QB.