# Fantasy Football Score Predictions w/ a Transformer

This repository contains a full pipeline for predicting **next-week fantasy football PPR scores** using a **Transformer-based sequence model** trained on multi-season NFL data.

The project:

- Uses [`nflreadpy`](https://github.com/nflverse/nflreadr) to pull **player-level weekly stats, rosters, and team context** from 2017–2025.
- Builds **fixed-length sequences** of up to 10 past games per player with lagged stats, moving averages, and opponent/usage context.
- Trains a **Transformer encoder** in PyTorch to predict **next-week PPR fantasy points**.
- Evaluates performance on the **2025 season** and compares the model against **Yahoo’s fantasy projections** for selected players.


## Features

- **End-to-end data pipeline**:
  - Load multi-year NFL stats via `nflreadpy`
  - Engineer features (lags, MA3/MA5, defense-vs-position, team pass rate)
  - Build [N, T, F] sequences (& masks) for Transformer input
- **Sequence model**:
  - Transformer encoder over 10-week histories
  - Numeric + categorical feature fusion
  - Position and team context included
- **Evaluation**:
  - Train on 2017–2024, validate on 2025
  - Compute MAE vs. actual PPR
  - Compare model vs. Yahoo projections on a curated set of players
  - Generate plots (error histograms, per-player MAE, etc.)

## Usage

**1. Build Player Sequences**
This step:
- Loads weekly stats and rosters with nflreadpy

- Filters to QB/RB/WR/TE

- Computes PPR scoring (if not already present)

- Adds lag-1, MA3, MA5 features

- Builds 10-week sequences per player

- Splits into train (2017–2024) and val (2025)

- python make_player_sequences.py

```console
python make_player_sequences.py
```

This script writes:

- data/processed/player_sequences_npz.npz

- data/processed/meta.parquet

- data/processed/feature_meta.json

**2. Train the Transformer**

train.py loads the NPZ, creates train/val datasets, and trains the FantasyTransformer.

```console
python train.py
```

Key details (from train.py):

- Train: samples where season < 2025

- Val: samples where season == 2025

- Optimizer: AdamW (lr = 2e-4, weight decay 1e-4)

- Loss: Smooth L1 (Huber)

- LR schedule: warmup + cosine decay

- Gradient clipping: max_norm=1.0

- Mixed precision via torch.cuda.amp when GPU is available

The script saves the best checkpoint to: checkpoints/fantasy_transformer_best.pt based on validation MAE.

## Model Architecture (High-level)
The model is defined in models/fantasy_transformer.py:

- Projects numeric features (x_num) to d_model = 192

- Embeds categorical features (team_id, opp_id, pos_id) into d_model each

- Concatenates numeric + all categorical embeddings per timestep and fuses them via a linear layer

- Adds learned positional encodings across sequence length

- Prepends a learnable [CLS] token

- Passes sequence through a Transformer encoder:

  - num_layers = 3, nhead = 4, feed-forward dim ff_mult * d_model

- Uses [CLS] (optionally concatenated with the last position embedding) as input to a small MLP head that outputs a scalar PPR prediction