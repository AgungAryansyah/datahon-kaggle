"""TrafficGNN: Graph neural network for multimodal traffic speed prediction.

Fuses per-node speed history with event text embeddings, then propagates
through GCN layers over the road network adjacency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj):
        h = adj @ x  # (N,N) @ (B,N,C) → (B,N,C)
        return self.drop(F.relu(self.norm(self.linear(h))))


class TrafficGNN(nn.Module):
    def __init__(self, num_nodes=1260, in_steps=15, out_steps=3,
                 text_dim=384, d_model=64, num_layers=4, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers

        self.time_enc = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.text_proj = nn.Linear(text_dim, d_model)

        in_dim = d_model * 2
        self.gcn = nn.ModuleList([
            GCNLayer(in_dim if i == 0 else d_model, d_model, dropout)
            for i in range(num_layers)
        ])

        decoder_in = d_model * num_layers
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, out_steps),
        )

    def forward(self, x, text_emb, adj):
        B, T, N = x.shape

        h = x.permute(0, 2, 1).reshape(B * N, 1, T)  # (B*N, 1, T)
        h = self.time_enc(h).squeeze(-1)  # (B*N, d_model)
        h = h.reshape(B, N, -1)  # (B, N, d_model)

        t = F.relu(self.text_proj(text_emb))  # (B, d_model)
        t = t.unsqueeze(1).expand(-1, N, -1)  # (B, N, d_model)

        h = torch.cat([h, t], dim=-1)  # (B, N, 2*d_model)

        skips = []
        for layer in self.gcn:
            h = layer(h, adj)
            skips.append(h)

        h = torch.cat(skips, dim=-1)  # (B, N, d_model * num_layers)
        out = self.decoder(h)  # (B, N, out_steps)
        return out.permute(0, 2, 1)  # (B, out_steps, N)
