# models/fantasy_transformer.py
import torch
import torch.nn as nn
from typing import List

class FantasyTransformer(nn.Module):
    """
    Sequence regressor for next-week PPR.
    Inputs:
      x_num: (B, T, F_num)  standardized numeric features
      x_cat: (B, T, F_cat)  integer IDs (team, opp, pos)
      mask:  (B, T)         1=valid, 0=pad
    Output:
      yhat:  (B,)           predicted PPR points
    """
    def __init__(
        self,
        num_feats: int,
        cat_vocab_sizes: List[int],     # e.g., [len(team_index), len(opp_index), len(pos_index)]
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.1,
        use_pos_encoding: bool = True,
    ):
        super().__init__()
        self.num_feats = num_feats
        self.d_model = d_model
        self.use_pos = use_pos_encoding

        # Project numeric features -> d_model
        self.num_proj = nn.Linear(num_feats, d_model)

        # Sum embeddings for each categorical slot (team/opp/pos)
        self.embs = nn.ModuleList([nn.Embedding(v, d_model) for v in cat_vocab_sizes])

        # Optional learned positional encodings (length-agnostic)
        if self.use_pos:
            self.pos_enc = nn.Embedding(512, d_model)   # supports sequences up to length 512

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_mult*d_model,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Regression head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x_num: (B, T, F_num) float
        x_cat: (B, T, F_cat) long
        mask : (B, T) float/bool (1=valid, 0=pad)
        """
        B, T, _ = x_num.shape

        # Numeric projection
        h = self.num_proj(x_num)  # (B, T, d)

        # Add categorical embeddings (sum across cat slots)
        for i, emb in enumerate(self.embs):
            h = h + emb(x_cat[:, :, i])  # (B, T, d)

        # Positional encoding (0..T-1)
        if self.use_pos:
            pos_ids = torch.arange(T, device=x_num.device).unsqueeze(0).expand(B, T)
            h = h + self.pos_enc(pos_ids)

        # Build key padding mask: True where PAD (mask==0)
        key_padding = (mask == 0)

        # Encode
        z = self.encoder(h, src_key_padding_mask=key_padding)  # (B, T, d)

        # Gather the last valid timestep for each sequence
        last_idx = mask.sum(dim=1).long() - 1                       # (B,)
        last_idx = torch.clamp(last_idx, min=0)                     # safe when all-pad (shouldn't happen)
        last_vec = z[torch.arange(B, device=z.device), last_idx]    # (B, d)

        # Predict
        yhat = self.head(last_vec).squeeze(-1)                      # (B,)
        return yhat
