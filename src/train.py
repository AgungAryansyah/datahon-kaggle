"""Training utilities for TrafficGNN v1 and v2."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def select_device(gpu_id=0):
    """Pick a specific GPU or fall back to CPU. Use gpu_id=None for CPU."""
    if gpu_id is not None and torch.cuda.is_available():
        if gpu_id < torch.cuda.device_count():
            device = torch.device(f"cuda:{gpu_id}")
            print(f"Device: {device} ({torch.cuda.get_device_name(gpu_id)})")
            return device
        print(f"GPU {gpu_id} not found ({torch.cuda.device_count()} available)")
    device = torch.device("cpu")
    print(f"Device: cpu")
    return device


def build_adj_tensor(adj_matrix):
    """Convert adjacency to normalized sparse-friendly dense tensor."""
    adj = torch.tensor(adj_matrix, dtype=torch.float32)
    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
    return deg_inv_sqrt.diag() @ adj @ deg_inv_sqrt.diag()


def encode_texts_minilm(texts, model_name="all-MiniLM-L6-v2", batch_size=256):
    """Pre-compute sentence embeddings with MiniLM."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        convert_to_numpy=True).astype(np.float32)

encode_texts = encode_texts_minilm  # backward compat (v1 notebooks)


@torch.no_grad()
def encode_texts_qwen(texts, model, tokenizer, device, batch_size=32, max_length=32):
    """Pre-compute pooled text embeddings with Qwen. Returns (N, 1024)."""
    model.eval()
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Qwen encoding", unit="batch"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding="max_length", truncation=True,
                           max_length=max_length, return_tensors="pt").to(device)
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state  # (B, L, 1024)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, 1024)
        embeddings.append(pooled.cpu().to(torch.float32).numpy())
    return np.concatenate(embeddings, axis=0)


def load_qwen(model_id="Qwen/Qwen3.5-0.8B", device=None):
    """Load Qwen with LoRA adapter for feature extraction."""
    from transformers import AutoModel, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModel.from_pretrained(
        model_id, trust_remote_code=True,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device != "cpu" else None,
        low_cpu_mem_usage=True,
    )
    base.eval()
    for p in base.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=8, lora_alpha=16,
        target_modules=["q_proj", "v_proj"], lora_dropout=0.1,
    )
    model = get_peft_model(base, lora_config)
    model.eval()
    return model, tokenizer


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


def train_epoch(model, loader, optimizer, criterion, adj, device, grad_clip=None):
    model.train()
    total_loss = 0
    for X, T_emb, Y in loader:
        X, T_emb, Y = X.to(device), T_emb.to(device), Y.to(device)
        optimizer.zero_grad()
        pred = model(X, T_emb, adj)
        loss = criterion(pred, Y)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
                epochs=100, lr=1e-3, patience=15, grad_clip=1.0, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(5, patience // 3),
    )
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    patience_left = patience

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, adj, device, grad_clip)
        val_loss = evaluate(model, val_loader, adj, device)
        scheduler.step(val_loss)

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


def train_one_config(model, train_loader, val_loader, adj, device,
                     epochs=50, lr=1e-3, patience=10, grad_clip=1.0,
                     verbose=False, horizon_weights=None):
    """Train a single config, returning best val loss, param count, and epochs used.

    Args:
        horizon_weights: optional (3,) tensor weighting h5, h10, h15 loss.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(3, patience // 3),
    )
    criterion = nn.MSELoss(reduction="none")
    best_val = float("inf")
    best_state = None
    patience_left = patience
    final_epoch = 0

    hw = torch.tensor(horizon_weights, device=device) if horizon_weights else torch.ones(3, device=device)

    for epoch in range(1, epochs + 1):
        final_epoch = epoch
        model.train()
        for Xb, Tb, Yb in train_loader:
            Xb, Tb, Yb = Xb.to(device), Tb.to(device), Yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb, Tb, adj)
            loss = ((pred - Yb) ** 2 * hw.view(1, 3, 1)).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xb, Tb, Yb in val_loader:
                Xb, Tb, Yb = Xb.to(device), Tb.to(device), Yb.to(device)
                pred = model(Xb, Tb, adj)
                val_loss += ((pred - Yb) ** 2 * hw.view(1, 3, 1)).mean().item() * Xb.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    model.load_state_dict(best_state)
    n_params = sum(p.numel() for p in model.parameters())
    return best_val, n_params, final_epoch


def build_multi_window_texts(texts, t_all, stride=5):
    """Build multi-window text strings: concatenate t, t-stride, t-2*stride.

    Args:
        texts: list of all text strings (aligned with speed timesteps)
        t_all: array of timestep indices for each window (e.g. range(N))
        stride: gap between text windows (default 5 steps = 20 min)

    Returns:
        list of combined text strings, one per window.
    """
    multi = []
    for t in t_all:
        parts = [texts[t]]
        if t >= stride:
            parts.append(texts[t - stride])
        if t >= 2 * stride:
            parts.append(texts[t - 2 * stride])
        multi.append(" | ".join(parts))
    return multi
