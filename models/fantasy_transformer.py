import torch
import torch.nn as nn
from typing import List

class FantasyTransformer(nn.Module):
    def __init__(
        self,
        num_feats: int,
        cat_vocab_sizes: List[int],
        d_model: int = 192,
        nhead: int = 4,
        num_layers: int = 3,
        ff_mult: int = 4,
        dropout: float = 0.1,
        use_pos_encoding: bool = True,
    ):
        super().__init__()
        self.num_feats = num_feats
        self.d_model = d_model
        self.use_pos = use_pos_encoding
        self.num_cat = len(cat_vocab_sizes)

        self.num_proj = nn.Linear(num_feats, d_model)

        self.embs = nn.ModuleList([nn.Embedding(v, d_model) for v in cat_vocab_sizes])

        self.fuse = nn.Linear(d_model * (1 + self.num_cat), d_model)

        if self.use_pos:
            self.pos_enc = nn.Embedding(512, d_model)

        self.cls = nn.Parameter(torch.randn(1, 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        head_in = d_model * (2 if self.num_cat >= 3 else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, mask: torch.Tensor):
        B, T, _ = x_num.shape
        device = x_num.device

        h_num = self.num_proj(x_num)

        cat_embs = []
        for i, emb in enumerate(self.embs):
            cat_embs.append(emb(x_cat[:, :, i]))
        h = torch.cat([h_num] + cat_embs, dim=-1)

        h = self.fuse(h)

        if self.use_pos:
            pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            h = h + self.pos_enc(pos_ids)

        cls_tok = self.cls.expand(B, 1, -1)
        h_in = torch.cat([cls_tok, h], dim=1)

        key_padding = (mask == 0)
        key_padding = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=device), key_padding], dim=1
        )
        z = self.encoder(h_in, src_key_padding_mask=key_padding)

        cls_vec = z[:, 0, :]

        if self.num_cat >= 3:
            last_idx = mask.sum(dim=1).long() - 1
            last_idx = torch.clamp(last_idx, min=0)

            pos_ids_last = x_cat[torch.arange(B, device=device), last_idx, 2]
            pos_emb_last = self.embs[2](pos_ids_last)

            head_in = torch.cat([cls_vec, pos_emb_last], dim=-1)
        else:
            head_in = cls_vec

        yhat = self.head(head_in).squeeze(-1)
        return yhat
