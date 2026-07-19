import json
from pathlib import Path
import pickle
import urllib.parse
import pandas as pd
import numpy as np

DATA_DIR = Path("dataset-task2")
CACHE_DIR = Path(".cache")


def load_articles() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "articles.csv")


def load_categories() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "categories.csv")


def load_states_train() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "states_train.csv")


def load_states_test() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "states_test.csv")


def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "sample_submission.csv")


def build_title_to_id() -> dict:
    arts = load_articles()
    return dict(zip(arts["title"].str.strip(), arts["article_id"]))


def load_or_build_adjacency() -> dict:
    path = CACHE_DIR / "wikispeedia_adj.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    title_to_id = build_title_to_id()
    links = pd.read_csv(
        "/tmp/wikispeedia_paths-and-graph/links.tsv",
        sep="\t",
        skiprows=14,
        header=None,
        names=["source", "target"],
    )
    links["source_decoded"] = (
        links["source"].apply(lambda x: urllib.parse.unquote(x).replace("_", " ").strip())
    )
    links["target_decoded"] = (
        links["target"].apply(lambda x: urllib.parse.unquote(x).replace("_", " ").strip())
    )
    links["source_id"] = links["source_decoded"].map(title_to_id)
    links["target_id"] = links["target_decoded"].map(title_to_id)
    links = links.dropna(subset=["source_id", "target_id"])
    links["source_id"] = links["source_id"].astype(int)
    links["target_id"] = links["target_id"].astype(int)
    adj = {i: [] for i in range(4604)}
    for _, row in links.iterrows():
        adj[row["source_id"]].append(row["target_id"])
    adj = {k: list(set(v)) for k, v in adj.items()}
    CACHE_DIR.mkdir(exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(adj, f)
    return adj


def make_submission(state_ids, predictions, path):
    sub = pd.DataFrame({"state_id": state_ids, "predicted_next_article_id": predictions})
    sub.to_csv(path, index=False)
    return sub


def validate_submission(path):
    expected = load_sample_submission()
    actual = pd.read_csv(path)
    assert list(actual.columns) == list(expected.columns), f"Columns: {list(actual.columns)}"
    assert len(actual) == len(expected), f"Rows: {len(actual)} != {len(expected)}"
    assert list(actual["state_id"]) == list(expected["state_id"]), "state_id mismatch"
    return True
