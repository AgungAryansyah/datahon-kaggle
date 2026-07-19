import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def train_one_epoch(model, optimizer, current, target, next_article,
                    candidate_ids, candidate_mask, batch_size=128):
    model.train()
    total_loss = 0
    n = len(current)
    perm = torch.randperm(n)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        cur_b = current[idx]
        tgt_b = target[idx]
        nxt_b = next_article[idx]
        cand_ids = candidate_ids[idx]
        cand_mask = candidate_mask[idx]
        logits = model(cur_b, tgt_b, cand_ids, cand_mask)
        loss = F.cross_entropy(logits, nxt_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / (n // batch_size + 1)


@torch.no_grad()
def evaluate(model, current, target, next_article,
             candidate_ids, candidate_mask, batch_size=256):
    model.eval()
    correct = total = 0
    n = len(current)
    for i in range(0, n, batch_size):
        cur_b = current[i:i + batch_size]
        tgt_b = target[i:i + batch_size]
        nxt_b = next_article[i:i + batch_size]
        cand_ids = candidate_ids[i:i + batch_size]
        cand_mask = candidate_mask[i:i + batch_size]
        logits = model(cur_b, tgt_b, cand_ids, cand_mask)
        pred = logits.argmax(dim=-1)
        correct += (pred == nxt_b).sum().item()
        total += len(cur_b)
    return correct / total
