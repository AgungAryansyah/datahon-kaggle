"""SOTA TrafficGNN v3 — paper-backed improvements over v2.

New modules:
- TextConditionedAdj: text embedding modifies adjacency per sample
- EntityMaskedFusion: sparse cross-attention, only mentioned roads attend
- SOTATrafficGNNv3: unified model with road metadata, weighted horizons
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_v2 import GatedTCN, DiffusionGCN, AdaptiveAdj


class TextConditionedAdj(nn.Module):
    """Text-conditioned adjacency: event text modifies graph connectivity.

    From CrossTrafficLLM — the LLM text encoder output produces a
    per-sample low-rank adjacency update on top of learned + static.
    Uses shared node basis vectors modulated by text — very few params.
    """

    def __init__(self, num_nodes=1260, d_text=1024, rank=8):
        super().__init__()
        self.node_basis = nn.Parameter(torch.randn(num_nodes, rank))
        self.text_gate = nn.Sequential(
            nn.Linear(d_text, rank),
            nn.Tanh(),
        )
        nn.init.xavier_uniform_(self.node_basis)

    def forward(self, static_adj, text_pooled):
        # text_pooled: (B, d_text)
        B = text_pooled.size(0)
        gate = self.text_gate(text_pooled)  # (B, rank)
        node_mod = self.node_basis * gate.unsqueeze(1)  # (B, N, rank)
        adj_update = node_mod @ node_mod.transpose(-2, -1)  # (B, N, N)

        deg = adj_update.sum(dim=-1, keepdim=True)
        deg_inv = torch.pow(deg + 1e-8, -0.5)
        adj_update = deg_inv * adj_update * deg_inv.transpose(-2, -1)

        return static_adj.unsqueeze(0) + adj_update


class EntityMaskedFusion(nn.Module):
    """Cross-modal attention with entity-aware sparsity.

    From CadST / T3 — only roads whose names appear in the event text
    attend to text tokens. Masked roads get a learned 'no-event' embedding.
    """

    def __init__(self, d_model, d_text, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_text, d_model)
        self.W_V = nn.Linear(d_text, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model * 2, 1)
        self.drop = nn.Dropout(dropout)
        self.no_event = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, node_feat, text_feat, entity_mask=None):
        B, N, _ = node_feat.shape
        S = text_feat.size(1)

        Q = self.W_Q(node_feat).reshape(B, N, self.n_heads, self.d_head)
        Q = Q.permute(0, 2, 1, 3)
        K = self.W_K(text_feat).reshape(B, S, self.n_heads, self.d_head)
        K = K.permute(0, 2, 1, 3)
        V = self.W_V(text_feat).reshape(B, S, self.n_heads, self.d_head)
        V = V.permute(0, 2, 1, 3)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if entity_mask is not None:
            # entity_mask: (N,) boolean — True for roads mentioned in text
            mask = entity_mask.view(1, 1, N, 1).float()  # (1,1,N,1)
            attn = attn * mask + (1 - mask) * (-1e9)

        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, V)
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)
        out = self.W_O(out)

        combined = torch.cat([node_feat, out], dim=-1)
        g = torch.sigmoid(self.gate(combined))

        if entity_mask is not None:
            no_ev = self.no_event.expand(B, N, -1)
            mask_3d = entity_mask.view(1, N, 1).float()
            out = mask_3d * out + (1 - mask_3d) * no_ev
            h = g * node_feat + (1 - g) * out
            return mask_3d * h + (1 - mask_3d) * no_ev

        return g * node_feat + (1 - g) * out


class SOTATrafficGNNv3(nn.Module):
    def __init__(self, num_nodes=1260, in_steps=15, out_horizons=3,
                 d_model=64, num_tcn_layers=4, num_gcn_layers=4,
                 num_hops=2, text_dim=1024, n_heads=4, meta_dim=0,
                 dropout=0.2, use_text=True, use_text_adj=True,
                 use_entity_mask=False):
        super().__init__()
        self.use_text = use_text
        self.use_text_adj = use_text_adj
        self.use_entity_mask = use_entity_mask
        self.num_nodes = num_nodes
        self.num_gcn_layers = num_gcn_layers

        # --- Temporal ---
        if num_tcn_layers > 0:
            self.tcn = GatedTCN(
                in_steps=in_steps, residual_channels=d_model,
                skip_channels=d_model, num_layers=num_tcn_layers,
                kernel_size=2, dropout=dropout,
            )
        else:
            self.tcn = None
            self.time_enc = nn.Sequential(
                nn.Conv1d(1, d_model, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            num_tcn_layers = 1

        # --- Road metadata projection ---
        self.meta_dim = meta_dim
        if meta_dim > 0:
            self.meta_proj = nn.Linear(meta_dim, d_model)

        # --- Text ---
        if use_text:
            self.entity_fusion = EntityMaskedFusion(d_model, text_dim, n_heads, dropout)

        # --- Adjacency ---
        self.adaptive_adj = AdaptiveAdj(num_nodes, embed_dim=10)
        if use_text_adj and use_text:
            self.text_cond_adj = TextConditionedAdj(num_nodes, text_dim, rank=8)

        # --- Spatial ---
        self.gcn_layers = nn.ModuleList([
            DiffusionGCN(d_model, d_model, num_hops, dropout)
            for _ in range(num_gcn_layers)
        ])

        # --- Decoder ---
        total_skips = num_tcn_layers + num_gcn_layers
        decoder_in = d_model * total_skips
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, out_horizons),
        )

    def forward(self, x, text_feat, adj, entity_mask=None, road_meta=None):
        B, T, N = x.shape

        # --- Temporal ---
        h = x.permute(0, 2, 1).reshape(B * N, 1, T)
        if self.tcn is not None:
            h, tcn_skip_list = self.tcn(h)
            h = h.reshape(B, N, -1)
            tcn_skips = [s.reshape(B, N, -1) for s in tcn_skip_list]
        else:
            h = self.time_enc(h).squeeze(-1)
            h = h.reshape(B, N, -1)
            tcn_skips = [h]

        # --- Road metadata ---
        if self.meta_dim > 0 and road_meta is not None:
            m = self.meta_proj(road_meta)  # (B, N, d_model) or (N, d_model)
            if m.dim() == 2:
                m = m.unsqueeze(0).expand(B, -1, -1)
            h = F.relu(h + m)

        # --- Text fusion ---
        if self.use_text:
            if text_feat.dim() == 2:
                text_feat = text_feat.unsqueeze(1)  # (B, 1, d_text)
            h = self.entity_fusion(h, text_feat, entity_mask)

        # --- Adjacency ---
        static_adj = self.adaptive_adj(adj)  # (N, N)
        if self.use_text_adj and self.use_text:
            text_pooled = text_feat.mean(dim=1)  # (B, d_text)
            adj_out = self.text_cond_adj(static_adj, text_pooled)  # (B, N, N)
        else:
            adj_out = static_adj.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)

        # --- Spatial ---
        gcn_skips = []
        for i, gcn in enumerate(self.gcn_layers):
            h = gcn(h, adj_out)  # adj_out is (B, N, N), einsum handles it
            gcn_skips.append(h)

        # --- Decode ---
        all_skips = tcn_skips + gcn_skips
        h = torch.cat(all_skips, dim=-1)
        out = self.decoder(h)
        return out.permute(0, 2, 1)
