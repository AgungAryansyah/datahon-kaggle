import torch
import torch.nn as nn
import torch.nn.functional as F


class LinkPredictor(nn.Module):
    def __init__(self, n_nodes, node_dim, hidden_dim=128):
        super().__init__()
        self.node_embed = nn.Embedding(n_nodes, node_dim)
        self.fuse = nn.Linear(node_dim * 3, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, current_ids, target_ids, candidate_ids, candidate_mask):
        current_emb = self.node_embed(current_ids)
        target_emb = self.node_embed(target_ids)
        cand_emb = self.node_embed(candidate_ids)
        fused = torch.cat([current_emb, target_emb, current_emb - target_emb], dim=-1)
        h = F.relu(self.fuse(fused))
        scores = self.score(h).squeeze(-1)
        scores = scores.masked_fill(~candidate_mask, float("-inf"))
        return scores


class DotProductLinkPredictor(nn.Module):
    def __init__(self, n_nodes, node_dim):
        super().__init__()
        self.node_embed = nn.Embedding(n_nodes, node_dim)

    def forward(self, current_ids, target_ids, candidate_ids, candidate_mask):
        current_emb = self.node_embed(current_ids)
        target_emb = self.node_embed(target_ids)
        cand_emb = self.node_embed(candidate_ids)
        scores = (current_emb * target_emb).sum(-1) + (cand_emb * target_emb).sum(-1)
        scores = scores.masked_fill(~candidate_mask, float("-inf"))
        return scores
