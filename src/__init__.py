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
