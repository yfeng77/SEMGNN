#!/usr/bin/env python
"""
gen_synthetic_anchor.py

Generator for the Synthetic-Anchor benchmark dataset.

Graph structure:
  - N nodes with a sparse Erdős–Rényi background graph (probability p_bg).
  - For each label ell, `anchors_per_label` anchor nodes are sampled uniformly
    at random.  Each anchor is then connected to `anchor_fanout` random nodes,
    creating a local hub-and-spoke structure that distinguishes anchors from the
    background.

Label assignment (k-hop reachability):
  - A node u is labelled positive for label ell if u lies within k_gt hops of
    at least one anchor for that label (computed via BFS).
  - Among multiple anchors for the same label, the nearest-anchor distance is
    used so that every reachable node gets the shortest possible path.

Ground-truth explanation masks:
  - Label-level GT (gt_edge_mask): shape (L, E_undirected).  For each label,
    the GT marks all BFS tree edges that lie on shortest paths from any positive
    node back to its closest anchor.  This is the union over all positive nodes.
  - Node-and-label-level GT (gt_edge_mask_node): shape (M, L, E_undirected).
    For each (node, label) pair where the node is positive, the GT marks only
    the single shortest path from that node to its closest anchor.

Node features:
  - Four structural features: degree, log(1+degree), clustering coefficient,
    k-core number.
  - If d > 4, the remaining dimensions are filled with small Gaussian noise
    (std = 0.05).
  - All features are standardised to zero mean and unit variance.

Output files (saved to out_dir):
  - A_coo.npz          : Symmetric adjacency matrix in COO format.
  - X.npy              : Node feature matrix [N, d].
  - Y.npy              : Multi-label target matrix [N, L].
  - expl_gt.npz        : Ground-truth explanation data (see above).
    Keys: labels_expl, gt_edge_mask, nodes_expl, gt_edge_mask_node.
    Note: edge_index_undirected is NOT stored here (derived from A_coo if needed).
"""

from __future__ import annotations
import argparse
import numpy as np
import scipy.sparse as sp
from collections import deque
from pathlib import Path


# ============================================================
# Edge and graph utilities
# ============================================================


def _add_edge_undirected(edge_set: set[tuple[int, int]], u: int, v: int):
    """
    Insert an undirected edge (u, v) into the canonical edge set.

    Edges are stored with the smaller node index first (u < v).
    Self-loops are silently ignored.

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


def _build_adj_list(N: int, undirected_edges: list[tuple[int, int]]):
    """
    Build an adjacency list representation from a list of undirected edges.

    Args:
        N:                Number of nodes.
        undirected_edges: List of (u, v) tuples with u < v.

    Returns:
        adj: List of length N where adj[i] is a list of neighbours of node i.
    """
    adj = [[] for _ in range(N)]
    for (u, v) in undirected_edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _bfs_parent_k(adj, src: int, k: int):
    """
    BFS from `src` up to `k` hops, recording distances and parent pointers.

    The parent pointer for each node allows shortest paths from any reachable
    node back to `src` to be reconstructed in O(k) time.

    Args:
        adj: Adjacency list (list of lists).
        src: Source node index.
        k:   Maximum BFS depth (hops).

    Returns:
        dist:   Integer array of length N; dist[v] is the BFS distance from
                src to v, or -1 if v is not reachable within k hops.
        parent: Integer array of length N; parent[v] is the predecessor of v
                on the shortest path from src, or -1 if v == src or unreachable.
    """
    N = len(adj)
    dist = np.full(N, -1, dtype=int)
    parent = np.full(N, -1, dtype=int)
    q = deque([src])
    dist[src] = 0
    while q:
        u = q.popleft()
        if dist[u] == k:
            # Do not expand beyond depth k
            continue
        for v in adj[u]:
            if dist[v] != -1:
                continue
            dist[v] = dist[u] + 1
            parent[v] = u
            q.append(v)
    return dist, parent


# ============================================================
# Dataset generator
# ============================================================


def gen_synthetic_anchor(
    out_dir: str = "./dataset/synthetic_anchor",
    seed: int = 42,
    N: int = 3000,
    L: int = 30,
    d: int = 16,
    # BFS hop radius that determines label membership
    k_gt: int = 2,
    # Erdős–Rényi background edge probability
    p_bg: float = 0.002,
    # Number of anchor nodes per label
    anchors_per_label: int = 2,
    # Number of additional random edges attached to each anchor
    anchor_fanout: int = 8,
    # Number of positive nodes for which node-level GT is stored
    num_nodes_expl: int = 400,
):
    """
    Generate the Synthetic-Anchor dataset and write all files to out_dir.

    Args:
        out_dir:           Output directory (created if it does not exist).
        seed:              Random seed for reproducibility.
        N:                 Total number of nodes.
        L:                 Number of labels.
        d:                 Node feature dimensionality.
        k_gt:              BFS hop radius for label assignment and GT edges.
        p_bg:              Background Erdős–Rényi edge probability.
        anchors_per_label: Number of anchor nodes sampled per label.
        anchor_fanout:     Number of random fanout edges added per anchor.
        num_nodes_expl:    Number of nodes included in the GT explanation file.
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Step 1: Background graph (sparse Erdős–Rényi)
    # ----------------------------------------------------------------
    num_pairs = N * (N - 1) // 2
    num_bg = int(p_bg * num_pairs)
    undirected_edges: set[tuple[int, int]] = set()
    while len(undirected_edges) < num_bg:
        u = int(rng.integers(0, N))
        v = int(rng.integers(0, N))
        _add_edge_undirected(undirected_edges, u, v)

    # ----------------------------------------------------------------
    # Step 2: Anchor nodes and fanout edges
    # ----------------------------------------------------------------
    # For each label, sample `anchors_per_label` distinct anchor nodes and
    # connect each anchor to `anchor_fanout` random nodes.  This creates a
    # local hub structure that BFS can identify as a k-hop neighbourhood.
    anchors = []
    for ell in range(L):
        anc = rng.choice(N, size=anchors_per_label, replace=False).astype(int)
        anchors.append(anc)
        for a in anc:
            neigh = rng.choice(N, size=anchor_fanout, replace=False).astype(int)
            for v in neigh:
                _add_edge_undirected(undirected_edges, int(a), int(v))

    # Sort for deterministic output and O(1) edge-to-id lookup
    undirected_edges = sorted(list(undirected_edges))
    E = len(undirected_edges)
    edge2id = {e: i for i, e in enumerate(undirected_edges)}
    adj = _build_adj_list(N, undirected_edges)

    # ----------------------------------------------------------------
    # Step 3: Label assignment and label-level GT edges
    # ----------------------------------------------------------------
    Y = np.zeros((N, L), dtype=np.float32)
    # Label-level GT: union of all shortest-path tree edges within k_gt hops
    gt_edges_by_label: list[set[tuple[int, int]]] = [set() for _ in range(L)]
    # Store (dist, parent, anchor_array) per label for node-level GT computation
    dist_parents = []

    for ell in range(L):
        # Multi-anchor BFS: keep the shortest distance across all anchors
        best_dist   = np.full(N, 10**9, dtype=int)
        best_parent = np.full(N, -1, dtype=int)

        for a in anchors[ell]:
            dist, parent = _bfs_parent_k(adj, int(a), k_gt)
            # Update best_dist / best_parent wherever this anchor is closer
            mask = (dist != -1) & (dist < best_dist)
            best_dist[mask]   = dist[mask]
            best_parent[mask] = parent[mask]
            # Note: the anchor itself has dist=0 and parent=-1 by definition

        # Label positive any node within k_gt hops of at least one anchor
        pos = np.where(best_dist <= k_gt)[0]
        Y[pos, ell] = 1.0
        dist_parents.append((best_dist, best_parent, anchors[ell].copy()))

        # Build label-level GT: trace each positive node's shortest path back
        # to its anchor and collect all edges on those paths
        for u in pos:
            if best_dist[u] <= 0:
                # Skip anchors themselves (they have no incoming path edge)
                continue
            cur = int(u)
            steps = int(best_dist[u])
            for _ in range(steps):
                p = int(best_parent[cur])
                if p < 0:
                    break
                a, b = (p, cur) if p < cur else (cur, p)
                gt_edges_by_label[ell].add((a, b))
                cur = p

    # ----------------------------------------------------------------
    # Step 4: Symmetric sparse adjacency matrix
    # ----------------------------------------------------------------
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
    # Step 5: Node features (structural)
    # ----------------------------------------------------------------
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(undirected_edges)

    # Four structural features per node
    deg   = np.array([G.degree(i) for i in range(N)], dtype=np.float32).reshape(-1, 1)
    clust = np.array(list(nx.clustering(G).values()),  dtype=np.float32).reshape(-1, 1)
    core  = np.array(list(nx.core_number(G).values()), dtype=np.float32).reshape(-1, 1)

    # Concatenate: [degree, log(1+degree), clustering coefficient, k-core number]
    X_struct = np.hstack([deg, np.log1p(deg), clust, core]).astype(np.float32)  # (N, 4)

    # Pad with small Gaussian noise to reach the target dimensionality d
    if d > X_struct.shape[1]:
        X_noise = rng.normal(size=(N, d - X_struct.shape[1])).astype(np.float32) * 0.05
        X = np.hstack([X_struct, X_noise])
    else:
        X = X_struct[:, :d]

    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X).astype(np.float32)

    # ----------------------------------------------------------------
    # Step 6: Ground-truth explanation masks
    # ----------------------------------------------------------------
    labels_expl = np.arange(L, dtype=int)

    # Label-level GT: mark every collected label-level path edge
    gt_edge_mask = np.zeros((L, E), dtype=bool)
    for ell in range(L):
        for e in gt_edges_by_label[ell]:
            gt_edge_mask[ell, edge2id[e]] = True

    # Sample nodes_expl from nodes that are positive for at least one label
    pos_any = np.where(Y.sum(1) > 0)[0]
    rng.shuffle(pos_any)
    nodes_expl = pos_any[:min(num_nodes_expl, pos_any.size)].astype(int)

    M = nodes_expl.size
    # Node-and-label-level GT: for each (node, label), mark only the single
    # shortest path from that node back to its closest anchor
    gt_edge_mask_node = np.zeros((M, L, E), dtype=bool)
    for i, u in enumerate(nodes_expl):
        for ell in range(L):
            best_dist, best_parent, _anc = dist_parents[ell]
            du = int(best_dist[u])
            if du <= 0 or du > k_gt:
                # Node is either an anchor itself or not reachable within k_gt hops
                continue
            cur = int(u)
            for _ in range(du):
                p = int(best_parent[cur])
                if p < 0:
                    break
                a, b = (p, cur) if p < cur else (cur, p)
                gt_edge_mask_node[i, ell, edge2id[(a, b)]] = True
                cur = p

    # ----------------------------------------------------------------
    # Save all output files
    # ----------------------------------------------------------------
    sp.save_npz(out_dir / "A_coo.npz", A.tocoo())
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "Y.npy", Y)
    np.savez_compressed(
        out_dir / "expl_gt.npz",
        labels_expl=labels_expl,
        gt_edge_mask=gt_edge_mask,                # (L, E_undirected)
        nodes_expl=nodes_expl,                    # (M,)
        gt_edge_mask_node=gt_edge_mask_node,       # (M, L, E_undirected)
    )

    # Diagnostic summary
    print(f"[DONE] anchor saved to {out_dir}")
    print(f"  N={N}, L={L}, d={d}, k_gt={k_gt}, undirected E={E}, A.nnz={A.nnz}")
    print(f"  nodes_expl={M}, avg positives per label={Y.mean(0).mean():.4f}")
    gt_per_node = gt_edge_mask_node.reshape(M, -1).sum(axis=1)
    print(f"  GT edges per node: min={gt_per_node.min()}, mean={gt_per_node.mean():.1f}, max={gt_per_node.max()}")
    cnt = Y.sum(0).astype(int)
    print(f"  [Y] per-label positives: min={cnt.min()}, median={np.median(cnt):.0f}, max={cnt.max()}")


# ============================================================
# Command-line interface
# ============================================================


def main():
    """Parse command-line arguments and generate the dataset."""
    ap = argparse.ArgumentParser(
        description="Generate the Synthetic-Anchor benchmark dataset."
    )
    ap.add_argument("--out_dir",           type=str,   default="./dataset/synthetic_anchor")
    ap.add_argument("--seed",              type=int,   default=42)
    ap.add_argument("--N",                 type=int,   default=3000,
                    help="Number of nodes.")
    ap.add_argument("--L",                 type=int,   default=30,
                    help="Number of labels.")
    ap.add_argument("--d",                 type=int,   default=16,
                    help="Node feature dimensionality.")
    ap.add_argument("--k_gt",              type=int,   default=2,
                    help="BFS hop radius for label assignment and GT edge collection.")
    ap.add_argument("--p_bg",              type=float, default=0.002,
                    help="Background Erdős–Rényi edge probability.")
    ap.add_argument("--anchors_per_label", type=int,   default=2,
                    help="Number of anchor nodes sampled per label.")
    ap.add_argument("--anchor_fanout",     type=int,   default=8,
                    help="Number of random edges added to each anchor node.")
    ap.add_argument("--num_nodes_expl",    type=int,   default=400,
                    help="Number of positive nodes included in the GT explanation file.")
    args = ap.parse_args()

    gen_synthetic_anchor(
        out_dir=args.out_dir,
        seed=args.seed,
        N=args.N,
        L=args.L,
        d=args.d,
        k_gt=args.k_gt,
        p_bg=args.p_bg,
        anchors_per_label=args.anchors_per_label,
        anchor_fanout=args.anchor_fanout,
        num_nodes_expl=args.num_nodes_expl,
    )


if __name__ == "__main__":
    main()