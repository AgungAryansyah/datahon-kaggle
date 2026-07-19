import numpy as np
import torch


def build_edge_index(adj: dict, device="cpu"):
    src, dst = [], []
    for s, targets in adj.items():
        for t in targets:
            src.append(s)
            dst.append(t)
    return torch.tensor([src, dst], dtype=torch.long, device=device)


def get_candidate_mask(adj: dict, article_ids, device="cpu"):
    n = max(adj.keys()) + 1
    mask = torch.zeros((len(article_ids), n), dtype=torch.bool, device=device)
    for i, aid in enumerate(article_ids):
        links = adj.get(aid, [])
        if links:
            mask[i, links] = True
    return mask


def category_overlap_matrix(cat_enc: np.ndarray):
    sim = cat_enc @ cat_enc.T
    return sim


def shortest_path_data(adj: dict, current_ids, target_ids, max_dist=10):
    results = []
    for c, t in zip(current_ids, target_ids):
        if c == t:
            results.append(0)
            continue
        visited = {c}
        queue = [(c, 0)]
        dist = max_dist
        while queue:
            node, d = queue.pop(0)
            if node == t:
                dist = d
                break
            if d >= max_dist:
                continue
            for nb in adj.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, d + 1))
        results.append(dist if dist <= max_dist else max_dist)
    return np.array(results)
