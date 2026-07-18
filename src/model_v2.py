"""SOTA TrafficGNN components with fixes.

- GatedTCN: gated dilated convolutions (fixed truncation)
- DiffusionGCN: K-hop diffusion graph convolution (fixed extra hop)
- AdaptiveAdj: learned adjacency matrix
- CrossModalFusion: multi-head cross-attention between nodes and text
- SOTATrafficGNN: unified model combining all components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedTCN(nn.Module):
    """Stack of gated dilated 1D convolutions with skip connections."""

    def __init__(self, in_steps=15, residual_channels=64, skip_channels=64,
                 num_layers=4, kernel_size=2, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.skip_convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        in_dim = residual_channels
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            self.filter_convs.append(
                nn.Conv1d(in_dim, residual_channels, kernel_size,
                          dilation=dilation, padding=padding))
            self.gate_convs.append(
                nn.Conv1d(in_dim, residual_channels, kernel_size,
                          dilation=dilation, padding=padding))
            self.skip_convs.append(nn.Conv1d(residual_channels, skip_channels, 1))

        self.start_conv = nn.Conv1d(1, residual_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.start_conv(x)
        skips = []

        for i in range(self.num_layers):
            residual = x
            f = torch.tanh(self.filter_convs[i](x))
            g = torch.sigmoid(self.gate_convs[i](x))
            x = f * g

            min_len = min(x.size(2), residual.size(2))
            x_trunc = x[:, :, :min_len]
            r_trunc = residual[:, :, :min_len]
            x = x_trunc + r_trunc

            skip = self.skip_convs[i](x)
            skips.append(skip[:, :, -1:])
            x = self.dropout(x)

        return x[:, :, -1], skips


class DiffusionGCN(nn.Module):
    """K-hop diffusion graph convolution."""

    def __init__(self, in_channels, out_channels, num_hops=2, dropout=0.1):
        super().__init__()
        self.num_hops = num_hops
        self.weights = nn.Parameter(
            torch.empty(num_hops, in_channels, out_channels))
        nn.init.xavier_uniform_(self.weights)
        self.norm = nn.LayerNorm(out_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj):
        out = 0
        h_hop = x
        for k in range(self.num_hops):
            out_k = torch.einsum("bnc,co->bno", h_hop, self.weights[k])
            out += out_k
            h_hop = adj @ h_hop
        return self.drop(F.relu(self.norm(out)))


class AdaptiveAdj(nn.Module):
    """Learned adjacency matrix, combined with static road network."""

    def __init__(self, num_nodes=1260, embed_dim=10):
        super().__init__()
        self.src_emb = nn.Embedding(num_nodes, embed_dim)
        self.dst_emb = nn.Embedding(num_nodes, embed_dim)

    def forward(self, static_adj):
        N = static_adj.size(0)
        idx = torch.arange(N, device=static_adj.device)
        s = self.src_emb(idx)
        d = self.dst_emb(idx)
        a = F.softmax(F.relu(s @ d.T), dim=-1)
        deg = a.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        a_norm = deg_inv_sqrt.diag() @ a @ deg_inv_sqrt.diag()
        return static_adj + a_norm


class CrossModalFusion(nn.Module):
    """Multi-head cross-attention: node features query text tokens."""

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

    def forward(self, node_feat, text_feat):
        B, N, _ = node_feat.shape
        S = text_feat.size(1)

        Q = self.W_Q(node_feat).reshape(B, N, self.n_heads, self.d_head)
        Q = Q.permute(0, 2, 1, 3)
        K = self.W_K(text_feat).reshape(B, S, self.n_heads, self.d_head)
        K = K.permute(0, 2, 1, 3)
        V = self.W_V(text_feat).reshape(B, S, self.n_heads, self.d_head)
        V = V.permute(0, 2, 1, 3)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, V)
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)
        out = self.W_O(out)

        combined = torch.cat([node_feat, out], dim=-1)
        g = torch.sigmoid(self.gate(combined))
        return g * node_feat + (1 - g) * out


class SOTATrafficGNN(nn.Module):
    def __init__(self, num_nodes=1260, in_steps=15, out_horizons=3,
                 d_model=64, num_tcn_layers=4, num_gcn_layers=4,
                 num_hops=2, text_dim=384, n_heads=4,
                 dropout=0.2, use_text=True, use_adaptive_adj=True,
                 fusion="cross"):
        super().__init__()
        self.use_text = use_text
        self.use_adaptive_adj = use_adaptive_adj
        self.fusion = fusion
        self.num_nodes = num_nodes
        self.num_tcn_layers = num_tcn_layers
        self.num_gcn_layers = num_gcn_layers

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

        if use_text:
            if fusion == "cross":
                self.cross_modal = CrossModalFusion(d_model, text_dim, n_heads, dropout)
            else:
                self.text_proj = nn.Sequential(
                    nn.Linear(text_dim, d_model), nn.ReLU(),
                )
                self.text_fuse = nn.Linear(d_model * 2, d_model)

        if use_adaptive_adj:
            self.adaptive_adj = AdaptiveAdj(num_nodes, embed_dim=10)

        self.gcn_layers = nn.ModuleList([
            DiffusionGCN(d_model, d_model, num_hops, dropout)
            for _ in range(num_gcn_layers)
        ])

        total_skips = num_tcn_layers + num_gcn_layers
        decoder_in = d_model * total_skips
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, out_horizons),
        )

    def forward(self, x, text_feat, adj):
        B, T, N = x.shape

        h = x.permute(0, 2, 1).reshape(B * N, 1, T)
        if self.tcn is not None:
            h, tcn_skip_list = self.tcn(h)
            h = h.reshape(B, N, -1)
            tcn_skips = [s.reshape(B, N, -1) for s in tcn_skip_list]
        else:
            h = self.time_enc(h).squeeze(-1)
            h = h.reshape(B, N, -1)
            tcn_skips = [h]

        if self.use_text:
            if self.fusion == "cross":
                if text_feat.dim() == 2:
                    text_feat = text_feat.unsqueeze(1)
                h = self.cross_modal(h, text_feat)
            else:
                if text_feat.dim() == 3:
                    text_feat = text_feat.mean(dim=1)
                t = self.text_proj(text_feat)
                t = t.unsqueeze(1).expand(-1, N, -1)
                h = F.relu(self.text_fuse(torch.cat([h, t], dim=-1)))

        if self.use_adaptive_adj:
            adj = self.adaptive_adj(adj)

        gcn_skips = []
        for gcn in self.gcn_layers:
            h = gcn(h, adj)
            gcn_skips.append(h)

        all_skips = tcn_skips + gcn_skips
        h = torch.cat(all_skips, dim=-1)
        out = self.decoder(h)
        return out.permute(0, 2, 1)
