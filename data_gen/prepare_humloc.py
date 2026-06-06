#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler


RAW_URLS = {
    "features.csv": "https://raw.githubusercontent.com/Tianqi-py/MLGNC/main/data/HumanGo/features.csv",
    "labels.csv": "https://raw.githubusercontent.com/Tianqi-py/MLGNC/main/data/HumanGo/labels.csv",
    "edge_list.csv": "https://raw.githubusercontent.com/Tianqi-py/MLGNC/main/data/HumanGo/edge_list.csv",
    "nodes.csv": "https://raw.githubusercontent.com/Tianqi-py/MLGNC/main/data/HumanGo/nodes.csv",
}


def _download_if_missing(raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in RAW_URLS.items():
        out_path = raw_dir / filename
        if out_path.exists():
            print("[SKIP] {} already exists".format(out_path))
            continue
        print("[DOWNLOAD] {}".format(filename))
        urlretrieve(url, out_path)


def _read_numeric_csv(path):
    """
    Robust numeric CSV reader.
    Handles optional header/index columns.
    """
    try:
        df = pd.read_csv(path, header=None)
    except Exception:
        df = pd.read_csv(path)

    # Convert everything to numeric; non-numeric header cells become NaN.
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop fully non-numeric rows/cols, often caused by headers.
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    arr = df.values

    # If the first row still contains NaN, remove it as a header artifact.
    if arr.ndim == 2 and arr.shape[0] > 0 and np.isnan(arr[0]).any():
        arr = arr[1:]

    # If the first column is 0,1,2,... or 1,2,3,..., treat it as an index column.
    if arr.ndim == 2 and arr.shape[1] > 2:
        first_col = arr[:, 0]
        if np.allclose(first_col, np.arange(arr.shape[0]), equal_nan=False):
            arr = arr[:, 1:]
        elif np.allclose(first_col, np.arange(1, arr.shape[0] + 1), equal_nan=False):
            arr = arr[:, 1:]

    if np.isnan(arr).any():
        raise ValueError("NaN remains after reading {}. Please inspect file format.".format(path))

    return arr


def convert_humloc(
    raw_dir="./raw/MLGNC/HumanGo",
    out_dir="./dataset/humloc",
    download=True,
    standardize_features=True,
):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if download:
        _download_if_missing(raw_dir)

    features_path = raw_dir / "features.csv"
    labels_path = raw_dir / "labels.csv"
    edges_path = raw_dir / "edge_list.csv"

    if not features_path.exists() or not labels_path.exists() or not edges_path.exists():
        raise FileNotFoundError(
            "Missing HumanGo files. Expected features.csv, labels.csv, and edge_list.csv "
            "under {}.".format(raw_dir)
        )

    X = _read_numeric_csv(features_path).astype(np.float32)
    Y = _read_numeric_csv(labels_path).astype(np.float32)
    edges = _read_numeric_csv(edges_path)

    if edges.ndim != 2 or edges.shape[1] < 2:
        raise ValueError("Unexpected edge shape: {}".format(edges.shape))

    edges = edges[:, :2].astype(np.int64)

    N = X.shape[0]
    if Y.shape[0] != N:
        raise ValueError("X and Y node counts mismatch: X={}, Y={}".format(X.shape, Y.shape))

    # Remove self-loops.
    edges = edges[edges[:, 0] != edges[:, 1]]

    # Detect 1-based indexing. Valid 0-based max is N-1.
    if edges.max() == N:
        print("[INFO] Detected possible 1-based edge indices. Converting to 0-based.")
        edges = edges - 1

    # Drop invalid edge rows.
    valid = (
        (edges[:, 0] >= 0) & (edges[:, 0] < N) &
        (edges[:, 1] >= 0) & (edges[:, 1] < N)
    )
    num_invalid = int((~valid).sum())
    if num_invalid > 0:
        print("[WARN] Dropping {} invalid edge rows.".format(num_invalid))
        edges = edges[valid]

    if edges.size == 0:
        raise ValueError("No valid edges remain after cleaning.")

    # Canonicalize undirected edges and remove duplicates.
    u = np.minimum(edges[:, 0], edges[:, 1])
    v = np.maximum(edges[:, 0], edges[:, 1])
    undirected_edges = np.unique(np.stack([u, v], axis=1), axis=0)

    # Build symmetric adjacency.
    rows = np.concatenate([undirected_edges[:, 0], undirected_edges[:, 1]])
    cols = np.concatenate([undirected_edges[:, 1], undirected_edges[:, 0]])
    data = np.ones(rows.shape[0], dtype=np.float32)

    A = sp.coo_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32)
    A.sum_duplicates()

    if standardize_features:
        X = StandardScaler().fit_transform(X).astype(np.float32)

    sp.save_npz(out_dir / "A_coo.npz", A.tocoo())
    np.save(out_dir / "X.npy", X.astype(np.float32))
    np.save(out_dir / "Y.npy", Y.astype(np.float32))

    meta = {
        "name": "humloc",
        "source": "MLGNC / HumanGo",
        "num_nodes": int(N),
        "num_undirected_edges": int(undirected_edges.shape[0]),
        "adj_nnz_directed": int(A.nnz),
        "feature_dim": int(X.shape[1]),
        "num_labels": int(Y.shape[1]),
        "avg_labels_per_node": float(Y.sum(axis=1).mean()),
        "has_explanation_gt": False,
    }

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n[DONE] HumLoc/HumanGo converted")
    print(json.dumps(meta, indent=2))
    print("Saved to: {}".format(out_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="./raw/MLGNC/HumanGo")
    parser.add_argument("--out_dir", type=str, default="./dataset/humloc")
    parser.add_argument("--no_download", action="store_true")
    parser.add_argument("--no_standardize", action="store_true")
    args = parser.parse_args()

    convert_humloc(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        download=(not args.no_download),
        standardize_features=(not args.no_standardize),
    )


if __name__ == "__main__":
    main()