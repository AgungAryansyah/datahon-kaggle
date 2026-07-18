from pathlib import Path

import numpy as np


DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset-task1"

WINDOW = 15
HORIZONS = [5, 10, 15]  # steps = +20, +40, +60 min


def load_train_speeds():
    s1 = np.load(DATASET_DIR / "train" / "train_speed_m1_1_11160.npy")  # (T1, 1260)
    s2 = np.load(DATASET_DIR / "train" / "train_speed_m2_1_5039.npy")  # (T2, 1260)
    return s1, s2


def load_train_texts():
    import json

    with open(DATASET_DIR / "train" / "train_text_m1_1_11160.json") as f:
        t1 = json.load(f)
    with open(DATASET_DIR / "train" / "train_text_m2_1_5039.json") as f:
        t2 = json.load(f)

    def to_sorted_list(text_dict, prefix):
        keys = sorted(text_dict.keys(), key=lambda k: int(k.split("_")[1]))
        return [text_dict[k] for k in keys]

    return to_sorted_list(t1, "m1"), to_sorted_list(t2, "m2")


def load_test_data():
    hist = np.load(DATASET_DIR / "test" / "test_X_hist.npy")  # (540, 15, 1260)
    import json

    with open(DATASET_DIR / "test" / "test_texts.json") as f:
        texts_dict = json.load(f)
    keys = sorted(texts_dict.keys(), key=lambda k: int(k.split("_")[1]))
    texts = [texts_dict[k] for k in keys]
    return hist, texts


def load_adjacency():
    return np.load(DATASET_DIR / "static" / "matrix.npy")  # (1260, 1260) int8


def load_roads():
    import json

    with open(DATASET_DIR / "static" / "Roads1260.json") as f:
        return json.load(f)


def build_windows(speeds, texts, stride=1):
    """Generate supervised windows from a speed block and aligned text list.

    Each window: (history_15, text_str, targets_3x1260)
    Targets at indices [t+WINDOW+h] for h in HORIZONS.
    """
    T = len(speeds)
    max_horizon = max(HORIZONS)
    max_t = T - WINDOW - max_horizon

    X, T_texts, Y = [], [], []
    for t in range(0, max_t, stride):
        X.append(speeds[t : t + WINDOW])  # (15, 1260)
        T_texts.append(texts[t])
        y = np.stack([speeds[t + WINDOW + h] for h in HORIZONS], axis=0)  # (3, 1260)
        Y.append(y)

    return np.array(X, dtype=np.float32), T_texts, np.array(Y, dtype=np.float32)


def compute_norm_stats(speeds):
    """Per-road mean and std, ignoring zero speeds (masked)."""
    mask = speeds > 0
    mean = np.where(mask, speeds, 0).sum(axis=0) / mask.sum(axis=0).clip(min=1)
    diff_sq = np.where(mask, (speeds - mean) ** 2, 0)
    std = np.sqrt(diff_sq.sum(axis=0) / mask.sum(axis=0).clip(min=1))
    std = std.clip(min=1e-3)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(speeds, mean, std):
    return (speeds - mean) / std


def denormalize(speeds, mean, std):
    return speeds * std + mean


def compute_mse(y_pred, y_true):
    """MSE averaged over all elements. y_pred/y_true shape: (N, 3, 1260)."""
    return np.mean((y_pred - y_true) ** 2)


def train_val_split(X, T, Y, val_frac=0.2):
    """Temporal split — last val_frac of windows for validation."""
    n = len(X)
    split = int(n * (1 - val_frac))
    T_train = T[:split] if T is not None else None
    T_val = T[split:] if T is not None else None
    return X[:split], T_train, Y[:split], X[split:], T_val, Y[split:]


def write_submission(predictions, label="", models=None):
    """Write predictions and optionally models to a timestamped run directory.

    Args:
        predictions: (540, 3, 1260) float32
        label: optional tag appended to directory name
        models: optional dict of {name: sklearn/xgboost model} to save alongside
    Returns:
        Path to the run directory.
    """
    import csv
    import pickle
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_{label}" if label else ts
    out_dir = DATASET_DIR.parent / "submissions" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "submission.csv"
    horizon_names = ["h5", "h10", "h15"]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "speed"])
        for s in range(540):
            for hi, hn in enumerate(horizon_names):
                for r in range(1260):
                    writer.writerow([f"test_{s:05d}_{hn}_r{r}", f"{predictions[s, hi, r]:.6f}"])

    expected_rows = 540 * 3 * 1260 + 1
    with open(out_path) as f:
        actual = sum(1 for _ in f)
    assert actual == expected_rows, f"Row count mismatch: {actual} vs {expected_rows}"

    if models:
        for name, model in models.items():
            model_path = out_dir / f"model_{name}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

    print(f"Run saved to {out_dir}/ ({actual - 1} predictions" + (f", {len(models)} models" if models else "") + ")")
    return out_dir


def validate_submission(submission_path):
    """Check submission.csv matches sample_submission.csv in id set and order."""
    import csv

    sample_path = DATASET_DIR / "sample_submission.csv"
    with open(sample_path) as f:
        sample_ids = [row[0] for row in csv.reader(f)][1:]  # skip header

    with open(submission_path) as f:
        sub_ids = [row[0] for row in csv.reader(f)][1:]

    if len(sub_ids) != len(sample_ids):
        raise ValueError(f"Row count mismatch: {len(sub_ids)} vs {len(sample_ids)}")

    mismatches = [(i, s, g) for i, (s, g) in enumerate(zip(sub_ids, sample_ids)) if s != g]
    if mismatches:
        for i, s, g in mismatches[:5]:
            print(f"  Row {i+2}: got {s}, expected {g}")
        raise ValueError(f"{len(mismatches)} id mismatches")

    print(f"Validated: {len(sub_ids)} rows, all ids match sample_submission.csv")


def build_features(hist_windows, adj, roads):
    """Vectorized feature extraction from history windows.

    Args:
        hist_windows: (N, 15, 1260) float32
        adj:         (1260, 1260) int8 adjacency
        roads:       list[1260] road metadata

    Returns:
        (N * 1260, 12) float32 feature matrix
    """
    N, T, R = hist_windows.shape

    roadclass = np.array([roads[r][0].get("roadclass", 0) for r in range(R)], dtype=np.float32)
    length = np.array([roads[r][0].get("length", 0) for r in range(R)], dtype=np.float32)

    lags = np.stack(
        [
            hist_windows[:, -1, :],
            hist_windows[:, -2, :],
            hist_windows[:, -4, :],
            hist_windows[:, -8, :],
            hist_windows[:, 0, :],
        ],
        axis=-1,
    )

    mean_h = hist_windows.mean(axis=1)
    std_h = hist_windows.std(axis=1)
    trend = hist_windows[:, -1, :] - hist_windows[:, 0, :]

    degrees = adj.sum(axis=1, keepdims=True).clip(min=1)
    adj_norm = adj.astype(np.float32) / degrees.astype(np.float32)

    last_step = hist_windows[:, -1, :]
    neighbor_last = last_step @ adj_norm.T

    step_3 = hist_windows[:, -3, :]
    neighbor_3 = step_3 @ adj_norm.T

    feats = np.stack(
        [
            lags[:, :, 0],
            lags[:, :, 1],
            lags[:, :, 2],
            lags[:, :, 3],
            lags[:, :, 4],
            mean_h,
            std_h,
            trend,
            np.broadcast_to(roadclass, (N, R)),
            np.broadcast_to(length, (N, R)),
            neighbor_last,
            neighbor_3,
        ],
        axis=-1,
    )

    return feats.reshape(-1, 12).astype(np.float32)


CACHE_DIR = DATASET_DIR.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


def cache_save(name, **arrays):
    """Save numpy arrays to cache. Use cache_load(name) to retrieve."""
    import pickle

    path = CACHE_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(arrays, f)
    sizes = {k: f"{v.shape} ({v.dtype})" for k, v in arrays.items()}
    print(f"Cached {name}: {sizes}")


def cache_load(name):
    """Load cached arrays. Returns dict or None if cache missing."""
    import pickle

    path = CACHE_DIR / f"{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
