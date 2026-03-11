"""
data_process.py
==============
Utility for loading graph datasets for multi-label node classification training.

Supports the following dataset layouts:
  - BlogCatalog-style: edges.csv / edges.txt + features.npy + id_mapping.csv + groups.csv
  - Synthetic-style  : A_coo.npz + X.npy + Y.npy  (id_mapping.csv optional)

Public API
----------
    load_for_training(data_dir, device, reprocess) -> (X_t, Y_t, A_sparse, idx, meta)
    build_features_and_save(data_dir, mode)        -> writes features.npy to data_dir
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

__all__ = ["load_for_training"]


# =============================================================================
# Internal helpers – reading raw data files
# =============================================================================

def _read_id_mapping(data_dir: str):
    """
    Load the node-id mapping table from ``id_mapping.csv``.

    The file is expected to contain two columns: ``orig_id`` (original node ID
    in the raw edge list) and ``idx`` (contiguous 0-based index used internally).

    Fallback behaviour
    ------------------
    If ``id_mapping.csv`` does not exist but ``X.npy`` is present, an identity
    mapping  ``orig_id == idx``  is constructed automatically so that datasets
    that already use 0-based integer node IDs (e.g. synthetic graphs) work
    without extra preprocessing.

    Returns
    -------
    id_df   : pd.DataFrame  with columns ``orig_id``, ``idx``
    mapping : dict          orig_id -> idx
    n       : int           total number of nodes
    """
    p = Path(data_dir) / "id_mapping.csv"

    # --- Case 1: id_mapping.csv exists ---
    if p.exists():
        df = pd.read_csv(p)

        # Normalise column names; accept any two-column file.
        if not {"orig_id", "idx"}.issubset(df.columns):
            if df.shape[1] >= 2:
                df = df.rename(columns={df.columns[0]: "orig_id", df.columns[1]: "idx"})
            else:
                raise ValueError("id_mapping.csv must contain two columns (orig_id, idx)")

        df = df.astype({"orig_id": int, "idx": int})
        n = int(df["idx"].max()) + 1
        mapping = dict(zip(df["orig_id"].values, df["idx"].values))
        return df, mapping, n

    # --- Case 2: no id_mapping.csv – infer identity mapping from X.npy ---
    X_path = Path(data_dir) / "X.npy"
    if not X_path.exists():
        raise FileNotFoundError(
            f"Neither {p} nor {X_path} found; cannot infer node count."
        )

    n = np.load(X_path).shape[0]
    orig_idx = np.arange(n, dtype=int)
    df = pd.DataFrame({"orig_id": orig_idx, "idx": orig_idx})
    mapping = {i: i for i in orig_idx}

    print(f"[id_mapping] {p} not found – using identity mapping 0..{n - 1}.")
    return df, mapping, n


def _read_edges_to_coo(
    data_dir: str, mapping: dict[int, int], n: int
) -> sp.coo_matrix:
    """
    Build a symmetric, self-loop-free adjacency matrix in COO format.

    Loading priority
    ----------------
    1. ``A_coo.npz``   – precomputed sparse matrix (synthetic datasets).
    2. ``edges.csv``   – two-column CSV (u, v), no header.
    3. ``edges.txt``   – whitespace-separated edge list.

    Parameters
    ----------
    data_dir : str   path to the dataset directory
    mapping  : dict  orig_id -> 0-based idx (from ``_read_id_mapping``)
    n        : int   total number of nodes

    Returns
    -------
    sp.coo_matrix  with shape (n, n), float32 values, no self-loops
    """
    d = Path(data_dir)

    # --- Priority 1: precomputed adjacency matrix ---
    p_coo = d / "A_coo.npz"
    if p_coo.exists():
        print(f"[edges] Found precomputed A_coo.npz at {p_coo}, loading directly.")
        A = sp.load_npz(p_coo).tocoo()
        if A.shape != (n, n):
            raise ValueError(f"A_coo.npz shape {A.shape} != ({n}, {n})")
        return A

    # --- Priority 2 / 3: edge list files ---
    edge_file = next(
        (d / name for name in ("edges.csv", "edges.txt") if (d / name).exists()),
        None,
    )
    if edge_file is None:
        raise FileNotFoundError(
            f"No edge file found in {d} (tried edges.csv, edges.txt, A_coo.npz)"
        )

    # Parse edge list.
    sep = "," if edge_file.suffix == ".csv" else r"\s+"
    ed = pd.read_csv(edge_file, header=None, names=["u", "v"], sep=sep)

    def _map(x):
        """Translate an original node ID to its internal index."""
        try:
            return mapping[int(x)]
        except KeyError:
            raise KeyError(f"Edge endpoint {x} not found in id_mapping")

    u = ed["u"].map(_map).to_numpy()
    v = ed["v"].map(_map).to_numpy()

    # Remove self-loops.
    mask = u != v
    u, v = u[mask], v[mask]

    data = np.ones_like(u, dtype=np.float32)
    A = sp.coo_matrix((data, (u, v)), shape=(n, n), dtype=np.float32)

    # Make undirected (A = A + A^T) and remove any duplicate / diagonal entries.
    A = A + A.T
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A.tocoo()


def _read_Y(data_dir: str, n: int) -> np.ndarray:
    """
    Load the multi-label target matrix from ``Y.npy``.

    Parameters
    ----------
    data_dir : str  path to the dataset directory
    n        : int  expected number of nodes

    Returns
    -------
    np.ndarray  shape (n, L), dtype float32
    """
    y_path = Path(data_dir) / "Y.npy"
    if not y_path.exists():
        raise FileNotFoundError(f"Label file not found: {y_path}")

    Y = np.load(y_path, allow_pickle=False)
    if Y.shape[0] != n:
        raise ValueError(f"Y.npy node count mismatch: {Y.shape[0]} != {n}")

    return Y.astype(np.float32, copy=False)


def _build_light_features_from_A(A: sp.coo_matrix) -> np.ndarray:
    """
    Construct lightweight degree-based node features without networkx.

    Features (per node)
    -------------------
    - Degree
    - log1p(degree)
    - sqrt(degree)

    All three are z-score normalised before returning.

    Returns
    -------
    np.ndarray  shape (n, 3), dtype float32
    """
    deg = np.asarray(A.tocsr().sum(1)).reshape(-1, 1).astype(np.float32)
    X = np.hstack([deg, np.log1p(deg), np.sqrt(np.clip(deg, 0, None))])

    mu = X.mean(0, keepdims=True)
    std = X.std(0, keepdims=True) + 1e-6
    return ((X - mu) / std).astype(np.float32)


def _read_or_make_features(data_dir: str, A: sp.coo_matrix) -> np.ndarray:
    """
    Resolve node features with the following fallback chain:

    1. ``features.npy`` – hand-crafted / pre-processed features (highest priority).
    2. ``X.npy``        – raw feature matrix (e.g. from synthetic generators).
    3. Auto-generated   – lightweight degree features (see ``_build_light_features_from_A``).

    Returns
    -------
    np.ndarray  shape (n, d), dtype float32
    """
    fpath = Path(data_dir) / "features.npy"
    xpath = Path(data_dir) / "X.npy"

    for path, label in [(fpath, "features.npy"), (xpath, "X.npy")]:
        if path.exists():
            X = np.load(path, allow_pickle=False)
            if X.shape[0] != A.shape[0]:
                raise ValueError(
                    f"{label} row count {X.shape[0]} != graph size {A.shape[0]}"
                )
            if label == "X.npy":
                print(f"[features] Using X.npy from {xpath}")
            return X.astype(np.float32, copy=False)

    # Neither file exists – fall back to degree features.
    print("[features] No feature file found; using auto-generated degree features.")
    return _build_light_features_from_A(A)


def _parse_groups_or_autogen(
    data_dir: str, id_df: pd.DataFrame, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse ``groups.csv`` and return boolean train / val / test masks.

    Supported ``groups.csv`` formats
    ---------------------------------
    - Single ``group`` / ``split`` column with string labels
      ``"train"`` / ``"val"`` / ``"valid"`` / ``"validation"`` / ``"test"``.
    - Single column with integer codes  0 = train, 1 = val, 2 = test.
    - Three separate boolean columns named ``train``, ``val``, ``test``.
    - Single unnamed column interpreted as a list of *test* node IDs; the
      remaining nodes are split 80 / 20 into train / val automatically.

    Fallback
    --------
    If ``groups.csv`` is absent or the format cannot be recognised, an 80 / 10 / 10
    random split is generated, saved back to ``groups.csv``, and returned.

    Returns
    -------
    train_mask, val_mask, test_mask  –  bool np.ndarrays of length n
    """
    gpath = Path(data_dir) / "groups.csv"
    train_mask = np.zeros(n, dtype=bool)
    val_mask   = np.zeros(n, dtype=bool)
    test_mask  = np.zeros(n, dtype=bool)

    # --- No groups.csv: generate 80 / 10 / 10 split and persist ---
    if not gpath.exists():
        order = np.arange(n)
        np.random.default_rng(42).shuffle(order)
        n_tr, n_va = int(n * 0.8), int(n * 0.1)
        tr, va, te = order[:n_tr], order[n_tr:n_tr + n_va], order[n_tr + n_va:]
        train_mask[tr] = True
        val_mask[va]   = True
        test_mask[te]  = True

        out = pd.DataFrame({"node_id": id_df["orig_id"].values, "group": "train"})
        out.loc[val_mask,  "group"] = "val"
        out.loc[test_mask, "group"] = "test"
        out.to_csv(gpath, index=False)
        return train_mask, val_mask, test_mask

    # --- Parse existing groups.csv ---
    gdf = pd.read_csv(gpath)
    gcols_lower = [c.lower() for c in gdf.columns]

    # Special case: single-column file whose column name is not a recognised
    # keyword → treat the values as a list of test-set node IDs.
    _known_keys = {
        "group", "split", "set", "phase", "train", "val", "test",
        "grp", "kfold", "fold", "split_id", "node_id", "id",
    }
    if gdf.shape[1] == 1 and gdf.columns[0].lower() not in _known_keys:
        col = gdf.columns[0]
        series = gdf[col]
        test_idx = None

        # Try interpreting values as raw 0-based indices.
        try:
            as_idx = series.astype(int).values
            if np.all((as_idx >= 0) & (as_idx < n)):
                test_idx = as_idx
        except (ValueError, TypeError):
            pass

        # Fall back to treating values as original node IDs.
        if test_idx is None:
            id_str_map = dict(
                zip(id_df["orig_id"].astype(str).values, id_df["idx"].values)
            )
            mapped = (
                pd.Series(series.astype(str).values)
                .map(id_str_map)
                .dropna()
                .astype(int)
                .values
            )
            if mapped.size > 0:
                test_idx = mapped

        if test_idx is not None:
            test_mask[test_idx] = True
            rest = np.where(~test_mask)[0]
            np.random.default_rng(42).shuffle(rest)
            n_va = max(int(len(rest) * 0.2), 1)
            val_mask[rest[:n_va]]  = True
            train_mask[rest[n_va:]] = True

            # Normalise and persist.
            out = pd.DataFrame({"node_id": id_df["orig_id"].values, "group": "train"})
            out.loc[val_mask,  "group"] = "val"
            out.loc[test_mask, "group"] = "test"
            out.to_csv(gpath, index=False)
            return train_mask, val_mask, test_mask

    # --- Standard path: align rows to internal 0-based indices ---
    # Detect the node-ID column (several common names accepted).
    if "node_id" in gdf.columns:
        id_col = "node_id"
    elif "orig_id" in gdf.columns:
        id_col = "orig_id"
    elif "id" in gcols_lower:
        id_col = gdf.columns[gcols_lower.index("id")]
        gdf = gdf.rename(columns={id_col: "node_id"})
        id_col = "node_id"
    else:
        # No ID column – assume rows are already in id_df order.
        gdf["node_id"] = id_df["orig_id"].values
        id_col = "node_id"

    id_str_map = dict(
        zip(id_df["orig_id"].astype(str).values, id_df["idx"].values)
    )
    gdf["idx"] = gdf[id_col].astype(str).map(id_str_map).values

    def _apply_group_series(series: pd.Series) -> bool:
        """
        Populate the three masks from a single split-label column.
        Accepts string labels or integer codes (0/1/2).
        Returns True if the column was successfully parsed.
        """
        s = series.astype(str).str.lower()
        if set(s.unique()) & {"train", "val", "valid", "validation", "test"}:
            train_mask[s.eq("train")] = True
            val_mask[s.isin(["val", "valid", "validation"])] = True
            test_mask[s.eq("test")] = True
            return True
        # Try integer encoding: 0 = train, 1 = val, 2 = test.
        try:
            si = series.astype(int)
            train_mask[si.eq(0)] = True
            val_mask[si.eq(1)]   = True
            test_mask[si.eq(2)]  = True
            return True
        except (ValueError, TypeError):
            return False

    recognized = False

    # Attempt 1: look for a single group-label column by common names.
    for key in ["group", "split", "set", "phase", "fold", "grp", "kfold", "split_id"]:
        if key in gcols_lower:
            col = gdf.columns[gcols_lower.index(key)]
            indexed = gdf.set_index("idx")[col]
            if _apply_group_series(indexed):
                recognized = True
                break

    # Attempt 2: three separate boolean/indicator columns named train / val / test.
    if not recognized and all(x in gcols_lower for x in ["train", "val", "test"]):
        tmp = gdf.set_index("idx")
        train_mask[tmp[tmp.columns[gcols_lower.index("train")]].astype(int).astype(bool)] = True
        val_mask[tmp[tmp.columns[gcols_lower.index("val")]].astype(int).astype(bool)]     = True
        test_mask[tmp[tmp.columns[gcols_lower.index("test")]].astype(int).astype(bool)]   = True
        recognized = True

    # Fallback: unrecognised format – auto-generate and overwrite.
    if not recognized:
        print("[groups] Unrecognised groups.csv format; generating 80/10/10 split.")
        order = np.arange(n)
        np.random.default_rng(42).shuffle(order)
        n_tr, n_va = int(n * 0.8), int(n * 0.1)
        tr, va, te = order[:n_tr], order[n_tr:n_tr + n_va], order[n_tr + n_va:]
        train_mask[tr] = True
        val_mask[va]   = True
        test_mask[te]  = True

        out = pd.DataFrame({"node_id": id_df["orig_id"].values, "group": "train"})
        out.loc[val_mask,  "group"] = "val"
        out.loc[test_mask, "group"] = "test"
        out.to_csv(gpath, index=False)

    return train_mask, val_mask, test_mask


# =============================================================================
# Public API
# =============================================================================

def load_for_training(
    data_dir: str,
    device: str = "cpu",
    reprocess: bool = False,  # kept for API compatibility; unused in this version
):
    """
    Load a graph dataset and return PyTorch tensors ready for training.

    Parameters
    ----------
    data_dir  : str   Path to the dataset directory.
    device    : str   PyTorch device string, e.g. ``"cpu"`` or ``"cuda:0"``.
    reprocess : bool  Legacy parameter retained for call-site compatibility.

    Returns
    -------
    X_t      : torch.FloatTensor  shape (N, d)  – node feature matrix
    Y_t      : torch.FloatTensor  shape (N, L)  – multi-label target matrix
    A_sparse : torch.sparse_coo_tensor  shape (N, N)  – adjacency matrix
    idx      : dict  keys ``"train"``, ``"val"``, ``"test"`` →
                     torch.LongTensor of node indices
    meta     : dict  with keys ``N``, ``L``, ``data_dir``,
                     ``feature_source``, ``feature_dim``
    """
    # --- Load raw data ---
    id_df, mapping, n = _read_id_mapping(data_dir)
    A = _read_edges_to_coo(data_dir, mapping, n)
    Y = _read_Y(data_dir, n)
    X = _read_or_make_features(data_dir, A)

    # Track which feature file was actually used (for logging / meta).
    feat_file = Path(data_dir) / "features.npy"
    x_file    = Path(data_dir) / "X.npy"
    if feat_file.exists():
        feat_source = "features.npy"
    elif x_file.exists():
        feat_source = "X.npy"
    else:
        feat_source = "auto_light"

    # --- Parse train / val / test split ---
    train_mask, val_mask, test_mask = _parse_groups_or_autogen(data_dir, id_df, n)

    # --- Convert to PyTorch tensors ---
    dev = torch.device(device)
    X_t = torch.from_numpy(X).to(dev)
    Y_t = torch.from_numpy(Y).to(dev)

    # Build sparse COO adjacency tensor.
    A_coo = A.tocoo()
    edge_index = torch.from_numpy(
        np.vstack([A_coo.row, A_coo.col]).astype(np.int64)
    )
    edge_weight = torch.from_numpy(A_coo.data.astype(np.float32))
    A_sparse = (
        torch.sparse_coo_tensor(edge_index, edge_weight, size=A_coo.shape)
        .coalesce()
        .to(dev)
    )

    # Build index tensors for each split.
    idx = {
        "train": torch.from_numpy(np.where(train_mask)[0]).long().to(dev),
        "val":   torch.from_numpy(np.where(val_mask)[0]).long().to(dev),
        "test":  torch.from_numpy(np.where(test_mask)[0]).long().to(dev),
    }

    meta = {
        "N": n,
        "L": int(Y.shape[1]),
        "data_dir": data_dir,
        "feature_source": feat_source,
        "feature_dim": int(X.shape[1]),
    }

    return X_t, Y_t, A_sparse, idx, meta


def build_features_and_save(data_dir: str, mode: str = "medium") -> None:
    """
    Compute structural node features and save them as ``features.npy``.

    Feature modes
    -------------
    ``"light"``
        Degree, log1p(degree), sqrt(degree).  Fast; no networkx required at
        call time (networkx is imported inside the function, but only the light
        branch avoids heavy graph algorithms).

    ``"medium"``  *(recommended default)*
        Light features + k-core number + local clustering coefficient.
        Runs in reasonable time for graphs with up to ~100k nodes.

    ``"heavy"``
        Medium features + PageRank + eigenvector centrality + triangle count.
        Can be slow on large graphs; use with caution.

    All features are z-score normalised before saving.

    Parameters
    ----------
    data_dir : str   Path to the dataset directory.
    mode     : str   One of ``"light"``, ``"medium"``, ``"heavy"``.
    """
    import networkx as nx

    # --- Load graph ---
    id_df, mapping, n = _read_id_mapping(data_dir)
    A = _read_edges_to_coo(data_dir, mapping, n).tocsr()

    # Degree-based features (always included).
    deg = np.asarray(A.sum(1)).reshape(-1, 1).astype(np.float32)
    feature_cols = [deg, np.log1p(deg), np.sqrt(np.clip(deg, 0, None))]

    def _nx_graph(A_csr):
        """Build a NetworkX Graph from a scipy sparse matrix (API-version safe)."""
        try:
            return nx.from_scipy_sparse_array(A_csr, create_using=nx.Graph)
        except AttributeError:
            # networkx < 2.7 uses from_scipy_sparse_matrix
            return nx.from_scipy_sparse_matrix(A_csr, create_using=nx.Graph)

    # --- Medium features: k-core + clustering coefficient ---
    if mode in ("medium", "heavy"):
        G = _nx_graph(A)

        core = np.array(
            [nx.core_number(G).get(i, 0) for i in range(n)], dtype=np.float32
        ).reshape(-1, 1)

        clus = np.array(
            [nx.clustering(G).get(i, 0.0) for i in range(n)], dtype=np.float32
        ).reshape(-1, 1)

        feature_cols += [core, clus]

    # --- Heavy features: PageRank + eigenvector centrality + triangles ---
    if mode == "heavy":
        G = G if "G" in dir() else _nx_graph(A)  # reuse if already built

        try:
            pr_map = nx.pagerank(G, max_iter=50)
            pr = np.array([pr_map.get(i, 0.0) for i in range(n)], dtype=np.float32)
        except Exception:
            pr = np.zeros(n, dtype=np.float32)

        try:
            ev_map = nx.eigenvector_centrality(G, max_iter=200)
            ev = np.array([ev_map.get(i, 0.0) for i in range(n)], dtype=np.float32)
        except Exception:
            ev = np.zeros(n, dtype=np.float32)

        try:
            tri_map = nx.triangles(G)
            tri = np.array([tri_map.get(i, 0) for i in range(n)], dtype=np.float32)
        except Exception:
            tri = np.zeros(n, dtype=np.float32)

        feature_cols += [
            pr.reshape(-1, 1),
            ev.reshape(-1, 1),
            tri.reshape(-1, 1),
        ]

    # --- Normalise and save ---
    X = np.hstack(feature_cols).astype(np.float32)
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)

    out_path = Path(data_dir) / "features.npy"
    np.save(out_path, X)
    print(f"[OK] features.npy saved to {data_dir}  shape={X.shape}  mode={mode}")