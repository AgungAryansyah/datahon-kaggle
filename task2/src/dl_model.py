import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, adj_norm):
        # adj_norm: sparse normalized adjacency [N, N]
        return F.relu(adj_norm @ self.w(x))


class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, out_dim)

    def forward(self, x, adj_norm):
        h = self.conv1(x, adj_norm)
        h = self.conv2(h, adj_norm)
        return h


class CandidateMLP(nn.Module):
    def __init__(self, feat_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPScorer(nn.Module):
    def __init__(self, emb_dim=384, n_cats=0, hidden_dim=128):
        super().__init__()
        feat_dim = emb_dim * 3 + n_cats * 3 + 2
        self.scorer = CandidateMLP(feat_dim, hidden_dim)

    def forward(self, curr_emb, tgt_emb, cand_emb, curr_cats, tgt_cats, cand_cats, out_deg_cand, n_links):
        x = torch.cat([
            curr_emb, tgt_emb, cand_emb,
            curr_cats, tgt_cats, cand_cats,
            out_deg_cand.unsqueeze(-1).float(),
            n_links.unsqueeze(-1).float(),
        ], dim=-1)
        return self.scorer(x)


class GCNScorer(nn.Module):
    def __init__(self, node_dim=384, hidden_dim=128, out_dim=64, n_cats=0):
        super().__init__()
        self.encoder = GCNEncoder(node_dim + n_cats, hidden_dim, out_dim)
        self.scorer = CandidateMLP(out_dim * 3 + 2, hidden_dim)

    def forward(self, curr_emb, tgt_emb, cand_emb,
                curr_cats, tgt_cats, cand_cats,
                out_deg_cand, n_links, adj_norm, node_ids,
                full_node_feats=None):
        if full_node_feats is None:
            node_feats = torch.cat([curr_emb, curr_cats], dim=-1)
            node_feats_all = node_feats
            node_embs = self.encoder(node_feats_all, adj_norm)
            cemb = node_embs[node_ids[:, 0]]
            temb = node_embs[node_ids[:, 1]]
            pemb = node_embs[node_ids[:, 2]]
        else:
            node_embs = self.encoder(full_node_feats, adj_norm)
            cemb = node_embs[node_ids[:, 0]]
            temb = node_embs[node_ids[:, 1]]
            pemb = node_embs[node_ids[:, 2]]
        x = torch.cat([
            cemb, temb, pemb,
            out_deg_cand.unsqueeze(-1).float(),
            n_links.unsqueeze(-1).float(),
        ], dim=-1)
        return self.scorer(x)


class CandidateDataset(Dataset):
    def __init__(self, states_df, adj, embeddings, cat_enc, device="cpu"):
        self.states = []
        for _, r in states_df.iterrows():
            curr, tgt, nxt = r["current_article_id"], r["target_article_id"], r["next_article_id"]
            links = adj.get(curr, [])
            if not links:
                continue
            curr_emb = torch.tensor(embeddings[curr], dtype=torch.float)
            tgt_emb = torch.tensor(embeddings[tgt], dtype=torch.float)
            curr_cats = torch.tensor(cat_enc[curr], dtype=torch.float)
            tgt_cats = torch.tensor(cat_enc[tgt], dtype=torch.float)
            for link in links:
                cand_emb = torch.tensor(embeddings[link], dtype=torch.float)
                cand_cats = torch.tensor(cat_enc[link], dtype=torch.float)
                self.states.append({
                    "curr_emb": curr_emb, "tgt_emb": tgt_emb, "cand_emb": cand_emb,
                    "curr_cats": curr_cats, "tgt_cats": tgt_cats, "cand_cats": cand_cats,
                    "out_deg_cand": float(len(adj.get(link, []))),
                    "n_links": float(len(links)),
                    "label": 1.0 if link == nxt else 0.0,
                })

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        s = self.states[i]
        return (
            s["curr_emb"], s["tgt_emb"], s["cand_emb"],
            s["curr_cats"], s["tgt_cats"], s["cand_cats"],
            torch.tensor(s["out_deg_cand"]),
            torch.tensor(s["n_links"]),
            torch.tensor(s["label"]),
        )


def build_normalized_adj(adj, n_nodes, device="cpu"):
    adj_np = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for s, ts in adj.items():
        for t in ts:
            adj_np[s, t] = 1.0
    deg = adj_np.sum(1, keepdims=True)
    deg[deg == 0] = 1
    d_inv_sqrt = np.power(deg, -0.5).flatten()
    adj_norm = adj_np * d_inv_sqrt[np.newaxis, :]
    adj_norm = adj_norm * d_inv_sqrt[:, np.newaxis]
    return torch.tensor(adj_norm, device=device)


@torch.no_grad()
def predict(model, test_df, adj, embeddings, cat_enc, adj_norm=None, device="cpu", batch_size=64):
    model.eval()
    preds = []
    for i in range(0, len(test_df), batch_size):
        batch = test_df.iloc[i:i + batch_size]
        for _, r in batch.iterrows():
            curr, tgt = r["current_article_id"], r["target_article_id"]
            links = adj.get(curr, [])
            if not links:
                preds.append(tgt)
                continue
            curr_emb = torch.tensor(embeddings[curr], dtype=torch.float, device=device).unsqueeze(0)
            tgt_emb = torch.tensor(embeddings[tgt], dtype=torch.float, device=device).unsqueeze(0)
            curr_cats = torch.tensor(cat_enc[curr], dtype=torch.float, device=device).unsqueeze(0)
            tgt_cats = torch.tensor(cat_enc[tgt], dtype=torch.float, device=device).unsqueeze(0)
            cand_embs = torch.tensor(embeddings[links], dtype=torch.float, device=device)
            cand_cats = torch.tensor(cat_enc[links], dtype=torch.float, device=device)
            out_degs = torch.tensor([len(adj.get(l, [])) for l in links], dtype=torch.float, device=device)
            n_links = torch.tensor(len(links), dtype=torch.float, device=device).repeat(len(links))

            curr_emb = curr_emb.expand(len(links), -1)
            tgt_emb = tgt_emb.expand(len(links), -1)
            curr_cats = curr_cats.expand(len(links), -1)
            tgt_cats = tgt_cats.expand(len(links), -1)

            if isinstance(model, GCNScorer):
                node_ids = torch.tensor(
                    [[curr] * len(links), [tgt] * len(links), links],
                    dtype=torch.long, device=device
                ).T
                full_feats = torch.cat([
                    torch.tensor(embeddings, dtype=torch.float, device=device),
                    torch.tensor(cat_enc, dtype=torch.float, device=device),
                ], dim=-1)
                scores = model(curr_emb, tgt_emb, cand_embs,
                               curr_cats, tgt_cats, cand_cats,
                               out_degs, n_links, adj_norm, node_ids,
                               full_node_feats=full_feats)
            else:
                scores = model(curr_emb, tgt_emb, cand_embs,
                               curr_cats, tgt_cats, cand_cats,
                               out_degs, n_links)
            preds.append(links[scores.argmax().item()])
    return np.array(preds)
