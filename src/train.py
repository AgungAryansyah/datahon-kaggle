"""Training utilities for TrafficGNN."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .model import TrafficGNN


def build_adj_tensor(adj_matrix):
    """Convert adjacency to normalized sparse-friendly dense tensor."""
    adj = torch.tensor(adj_matrix, dtype=torch.float32)
    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
    return deg_inv_sqrt.diag() @ adj @ deg_inv_sqrt.diag()


def encode_texts(texts, model_name="all-MiniLM-L6-v2", batch_size=256):
    """Pre-compute sentence embeddings for all text strings."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        convert_to_numpy=True).astype(np.float32)


class TrafficDataset(Dataset):
    def __init__(self, X, text_embs, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.text_embs = torch.tensor(text_embs, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.text_embs[idx], self.Y[idx]


def create_dataloaders(X_train, T_emb_train, Y_train, X_val, T_emb_val, Y_val,
                       batch_size=32, num_workers=0):
    train_ds = TrafficDataset(X_train, T_emb_train, Y_train)
    val_ds = TrafficDataset(X_val, T_emb_val, Y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def train_epoch(model, loader, optimizer, criterion, adj, device):
    model.train()
    total_loss = 0
    for X, T_emb, Y in loader:
        X, T_emb, Y = X.to(device), T_emb.to(device), Y.to(device)
        optimizer.zero_grad()
        pred = model(X, T_emb, adj)
        loss = criterion(pred, Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, adj, device):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    for X, T_emb, Y in loader:
        X, T_emb, Y = X.to(device), T_emb.to(device), Y.to(device)
        pred = model(X, T_emb, adj)
        total_loss += criterion(pred, Y).item() * X.size(0)
    return total_loss / len(loader.dataset)


def train_model(model, train_loader, val_loader, adj, device,
                epochs=100, lr=1e-3, patience=15, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    patience_left = patience

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, adj, device)
        val_loss = evaluate(model, val_loader, adj, device)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"Epoch {epoch:3d} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return best_val
