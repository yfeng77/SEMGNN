#!/usr/bin/env python
"""
gen_synthetic_motif.py

Generator for the Synthetic-Motif benchmark dataset.

Graph structure (Scheme C):
  - N nodes are split into two pools:
      * support_all: L * support_per_label nodes, partitioned into L label-specific
        support pools.  Motif neighbour nodes (v1, v2, w1, w2) are drawn from
        these pools.
      * free_nodes: all remaining nodes; anchor (positive) nodes are sampled here,
        which avoids the "7/2/1 imbalance problem" caused by anchors sharing
        support pools across labels.

  - Edges are added in three passes:
      1. Global background edges (Erdős–Rényi with probability p_bg).
      2. Intra-block edges that sparsely connect each label's support pool
         (prevents the block from becoming a disconnected set of isolated nodes).
      3. Global inter-block noise edges (very sparse, probability p_inter).

Motif design (per positive anchor u, per label ell):
  - Four motif nodes (v1, v2, w1, w2) are sampled from the label's support pool.
  - The following edges are always added:
      u-v1, u-v2, v1-w1, v1-w2, v2-w1, v2-w2  (forms a "diamond" around u)
  - Optionally, a leaf edge w1-w2 is added for a stronger structural signal.

Ground-truth explanation masks:
  - Label-level GT (gt_edge_mask): shape (L, E_undirected), marks all motif edges
    for each label.
  - Node-and-label-level GT (gt_edge_mask_node): shape (M, L, E_undirected), marks
    only the specific motif edges associated with each (anchor, label) pair.

Node features:
  - Five structural features per node: degree, log-degree, clustering coefficient,
    core number, triangle count.
  - If d > 5, the remaining dimensions are filled with small Gaussian noise (std=0.05).
  - All features are standardised to zero mean and unit variance.

Output files (saved to out_dir):
  - A_coo.npz          : Symmetric adjacency matrix in COO format.
  - X.npy              : Node feature matrix [N, d].
  - Y.npy              : Multi-label target matrix [N, L].
  - expl_gt.npz        : Ground-truth explanation data (see above).
"""

from __future__ import annotations
import argparse
import numpy as np
import scipy.sparse as sp
from pathlib import Path
import networkx as nx
from sklearn.preprocessing import StandardScaler


# ============================================================
# Edge utilities
# ============================================================


def _add_edge_undirected(edge_set: set[tuple[int, int]], u: int, v: int):
    """
    Insert an undirected edge (u, v) into the canonical edge set.

    Edges are stored in canonical form with the smaller node index first
    (u < v).  Self-loops are silently ignored.

    Args:
        edge_set: Mutable set of (u, v) tuples with u < v.
        u:        First endpoint.
        v:        Second endpoint.
    """
    if u == v:
        return
    if u > v:
        u, v = v, u
    edge_set.add((u, v))


# ============================================================
# Dataset generator
# ============================================================


def gen_synthetic_motif(
    out_dir: str = "./dataset/synthetic_motif",
    seed: int = 42,
    N: int = 3000,
    L: int = 30,
    d: int = 16,
    # Global background / noise edge probability
    p_bg: float = 0.001,
    # Intra-block connectivity probability (keeps support pools connected)
    p_intra: float = 0.02,
    # Global inter-block noise edge probability
    p_inter: float = 0.0002,
    # Fraction of free_nodes used as positive anchors per label
    pos_ratio_in_block: float = 0.10,
    # If True, add the w1-w2 leaf edge for a stronger motif signal
    add_leaf_edge: bool = True,
    # Number of anchor nodes for which node-level GT is generated
    num_nodes_expl: int = 400,
):
    """
    Generate the Synthetic-Motif dataset and write all files to out_dir.

    Args:
        out_dir:           Output directory path (created if it does not exist).
        seed:              Random seed for full reproducibility.
        N:                 Total number of nodes.
        L:                 Number of labels.
        d:                 Node feature dimensionality (>= 5 for structural features).
        p_bg:              Background edge probability (Erdős–Rényi style).
        p_intra:           Intra-support-pool edge probability per label.
        p_inter:           Global inter-block noise edge probability.
        pos_ratio_in_block:Fraction of free_nodes labelled positive per label.
        add_leaf_edge:     Whether to add the w1-w2 edge inside each motif.
        num_nodes_expl:    Number of anchor nodes included in the GT explanation file.
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Node pool partitioning (Scheme C)
    # ----------------------------------------------------------------
    # Each label gets a dedicated support pool of `support_per_label` nodes.
    # Motif neighbours (v1, v2, w1, w2) are always drawn from this pool.
    # Positive anchor nodes are sampled from the shared free_nodes pool to
    # prevent the imbalance that occurs when anchors share support pools.
    support_per_label = 30  # Recommended range: 20–50; must be >= need
    need = 5 if add_leaf_edge else 4
    assert support_per_label >= need, f"support_per_label must be >= {need}"

    perm = rng.permutation(N)
    support_total = support_per_label * L
    assert support_total < N, "support_total must be < N to leave free_nodes for anchors"

    support_all = perm[:support_total]   # Reserved for motif neighbour nodes
    free_nodes  = perm[support_total:]   # Pool for anchor (positive) node sampling

    # Slice support_all into L label-specific pools
    support_pools = []
    for ell in range(L):
        s = ell * support_per_label
        e = (ell + 1) * support_per_label
        support_pools.append(support_all[s:e].astype(int))

    # ----------------------------------------------------------------
    # Edge construction
    # ----------------------------------------------------------------
    undirected_edges: set[tuple[int, int]] = set()

    # --- Pass 1: Global background edges (random sparse connectivity) ---
    num_pairs = N * (N - 1) // 2
    num_bg = int(p_bg * num_pairs)
    while len(undirected_edges) < num_bg:
        u = int(rng.integers(0, N))
        v = int(rng.integers(0, N))
        _add_edge_undirected(undirected_edges, u, v)

    # --- Pass 2: Intra-block edges (keeps each support pool connected) ---
    for ell in range(L):
        nodes = support_pools[ell]
        s = nodes.size
        num_intra = int(p_intra * (s * (s - 1) // 2))
        if num_intra <= 0:
            continue
        cnt = 0
        while cnt < num_intra:
            u = int(nodes[rng.integers(0, s)])
            v = int(nodes[rng.integers(0, s)])
            if u == v:
                continue
            before = len(undirected_edges)
            _add_edge_undirected(undirected_edges, u, v)
            if len(undirected_edges) > before:
                cnt += 1

    # --- Pass 3: Global inter-block noise edges (very sparse) ---
    num_inter = int(p_inter * num_pairs)
    cnt = 0
    while cnt < num_inter:
        u = int(rng.integers(0, N))
        v = int(rng.integers(0, N))
        if u == v:
            continue
        before = len(undirected_edges)
        _add_edge_undirected(undirected_edges, u, v)
        if len(undirected_edges) > before:
            cnt += 1

    # ----------------------------------------------------------------
    # Label assignment and motif injection
    # ----------------------------------------------------------------
    Y = np.zeros((N, L), dtype=np.float32)

    # Label-level GT: set of motif edges per label (undirected, canonical)
    gt_edges_by_label: list[set[tuple[int, int]]] = [set() for _ in range(L)]
    # Node-and-label-level GT: maps (anchor_node, label) -> set of motif edges
    motif_by_anchor: dict[tuple[int, int], set[tuple[int, int]]] = {}

    for ell in range(L):
        nodes = support_pools[ell]  # Motif neighbour candidates for this label
        s = nodes.size

        # Sample positive anchors from the shared free_nodes pool
        num_pos = max(5, int(round(pos_ratio_in_block * free_nodes.size)))
        num_pos = min(num_pos, free_nodes.size)
        anchors = rng.choice(free_nodes, size=num_pos, replace=False)

        Y[anchors, ell] = 1.0  # Mark these nodes as positive for label ell

        for u in anchors:
            # Sample 4 distinct motif nodes from the label-specific support pool
            pick = rng.choice(nodes, size=4, replace=False)
            v1, v2, w1, w2 = map(int, pick.tolist())

            # Inject the diamond motif: u -> {v1, v2} -> {w1, w2}
            edges_m: set[tuple[int, int]] = set()
            for (a, b) in [(u, v1), (u, v2), (v1, w1), (v1, w2), (v2, w1), (v2, w2)]:
                _add_edge_undirected(undirected_edges, a, b)
                aa, bb = (a, b) if a < b else (b, a)
                edges_m.add((aa, bb))

            if add_leaf_edge:
                # Optional leaf edge w1-w2 for a stronger structural signal
                _add_edge_undirected(undirected_edges, w1, w2)
                aa, bb = (w1, w2) if w1 < w2 else (w2, w1)
                edges_m.add((aa, bb))

            gt_edges_by_label[ell] |= edges_m
            motif_by_anchor[(int(u), int(ell))] = edges_m

    # ----------------------------------------------------------------
    # Finalise edge list and build adjacency matrix
    # ----------------------------------------------------------------
    # Sort for deterministic output and to enable O(1) edge-to-id lookup
    undirected_edges = sorted(list(undirected_edges))
    E = len(undirected_edges)
    edge_index_undirected = np.array(undirected_edges, dtype=np.int64)  # (E, 2), u < v
    edge2id = {e: i for i, e in enumerate(undirected_edges)}

    # Build symmetric sparse adjacency matrix (directed entries in both directions)
    rows = []
    cols = []
    for (u, v) in undirected_edges:
        rows += [u, v]
        cols += [v, u]
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.coo_matrix(
        (data, (np.array(rows), np.array(cols))),
        shape=(N, N),
        dtype=np.float32,
    )
    A.sum_duplicates()

    # ----------------------------------------------------------------
    # Node features (structural)
    # ----------------------------------------------------------------
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(undirected_edges)

    # Extract five local structural statistics as node features
    deg   = np.array([G.degree(i) for i in range(N)], dtype=np.float32).reshape(-1, 1)
    clust = np.array(list(nx.clustering(G).values()),   dtype=np.float32).reshape(-1, 1)
    core  = np.array(list(nx.core_number(G).values()),  dtype=np.float32).reshape(-1, 1)
    tri   = np.array(list(nx.triangles(G).values()),    dtype=np.float32).reshape(-1, 1)

    # Concatenate: [degree, log(1+degree), clustering, k-core, triangle count]
    X_struct = np.hstack([deg, np.log1p(deg), clust, core, tri]).astype(np.float32)  # (N, 5)

    # Pad with small Gaussian noise to reach the target dimensionality d
    if d > X_struct.shape[1]:
        X_noise = rng.normal(size=(N, d - X_struct.shape[1])).astype(np.float32) * 0.05
        X = np.hstack([X_struct, X_noise]).astype(np.float32)
    else:
        X = X_struct[:, :d].astype(np.float32)

    # Standardise all features to zero mean and unit variance
    X = StandardScaler().fit_transform(X).astype(np.float32)

    # ----------------------------------------------------------------
    # Ground-truth explanation masks
    # ----------------------------------------------------------------
    labels_expl = np.arange(L, dtype=int)

    # Label-level GT: for each label, mark every motif edge in the full graph
    gt_edge_mask = np.zeros((L, E), dtype=bool)
    for ell in range(L):
        for e in gt_edges_by_label[ell]:
            gt_edge_mask[ell, edge2id[e]] = True

    # Prioritise anchor nodes (they have the clearest GT) for the node-level GT
    anchors_all = np.where(Y.sum(1) > 0)[0]
    rng.shuffle(anchors_all)
    nodes_expl = anchors_all[:min(num_nodes_expl, anchors_all.size)].astype(int)

    M = nodes_expl.size
    # Node-and-label-level GT: for each (anchor, label) pair, mark only that
    # anchor's specific motif edges (subset of the label-level GT)
    gt_edge_mask_node = np.zeros((M, L, E), dtype=bool)
    for i, u in enumerate(nodes_expl):
        for ell in range(L):
            key = (int(u), int(ell))
            if key not in motif_by_anchor:
                continue
            for e in motif_by_anchor[key]:
                gt_edge_mask_node[i, ell, edge2id[e]] = True

    # Diagnostic: per-label positive counts and average labels per node
    cnt = Y.sum(0).astype(int)
    print(f"[Y] per-label positives: min={cnt.min()}, median={np.median(cnt)}, max={cnt.max()}")
    print(f"[Y] avg labels per node ~= {Y.sum(1).mean():.2f}")

    # ----------------------------------------------------------------
    # Save all output files
    # ----------------------------------------------------------------
    sp.save_npz(out_dir / "A_coo.npz", A.tocoo())
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "Y.npy", Y)
    np.savez_compressed(
        out_dir / "expl_gt.npz",
        labels_expl=labels_expl,
        gt_edge_mask=gt_edge_mask,                 # (L, E_undirected)
        nodes_expl=nodes_expl,                     # (M,)
        gt_edge_mask_node=gt_edge_mask_node,        # (M, L, E_undirected)
        edge_index_undirected=edge_index_undirected,  # (E, 2)
    )
    print(f"[DONE] motif saved to {out_dir}")
    print(f"  N={N}, L={L}, d={d}, undirected E={E}, A.nnz={A.nnz}")
    print(f"  nodes_expl={M}, pos_per_label(block)~{pos_ratio_in_block:.2f}")


# ============================================================
# Command-line interface
# ============================================================


def main():
    """Parse command-line arguments and generate the dataset."""
    ap = argparse.ArgumentParser(
        description="Generate the Synthetic-Motif benchmark dataset."
    )
    ap.add_argument("--out_dir",            type=str,   default="./dataset/synthetic_motif")
    ap.add_argument("--seed",               type=int,   default=42)
    ap.add_argument("--N",                  type=int,   default=3000,
                    help="Number of nodes.")
    ap.add_argument("--L",                  type=int,   default=30,
                    help="Number of labels.")
    ap.add_argument("--d",                  type=int,   default=16,
                    help="Node feature dimensionality.")
    ap.add_argument("--pos_ratio_in_block", type=float, default=0.10,
                    help="Fraction of free_nodes labelled positive per label.")
    ap.add_argument("--p_bg",               type=float, default=0.001,
                    help="Background edge probability.")
    ap.add_argument("--p_intra",            type=float, default=0.02,
                    help="Intra-support-pool edge probability.")
    ap.add_argument("--p_inter",            type=float, default=0.0002,
                    help="Global inter-block noise edge probability.")
    ap.add_argument("--num_nodes_expl",     type=int,   default=400,
                    help="Number of anchor nodes included in the GT explanation file.")
    ap.add_argument("--no_leaf_edge",       action="store_true",
                    help="If set, omit the w1-w2 leaf edge from each motif.")
    args = ap.parse_args()

    gen_synthetic_motif(
        out_dir=args.out_dir,
        seed=args.seed,
        N=args.N,
        L=args.L,
        d=args.d,
        pos_ratio_in_block=args.pos_ratio_in_block,
        p_bg=args.p_bg,
        p_intra=args.p_intra,
        p_inter=args.p_inter,
        add_leaf_edge=(not args.no_leaf_edge),
        num_nodes_expl=args.num_nodes_expl,
    )


if __name__ == "__main__":
    main()