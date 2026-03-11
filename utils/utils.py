# utils.py
"""
General-purpose utility functions shared across the SEMGNN codebase.
"""

from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
import torch
import random


# ============================================================
# Sparse graph utilities
# ============================================================

def sparse_to_edge_index(A_sparse: torch.Tensor) -> torch.Tensor:
    """
    Extract the edge index from a sparse COO adjacency tensor.

    Args:
        A_sparse: Sparse COO tensor of shape [N, N].

    Returns:
        Edge index tensor of shape [2, E] where E is the number of stored entries.
    """
    A_sparse = A_sparse.coalesce()
    return A_sparse.indices()

# ============================================================
# Training / data-split helpers
# ============================================================

def set_seed(seed: int = 42):
    """
    Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def multilabel_stratified_split(
    Y: np.ndarray,
    train: float = 0.7,
    val: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Greedy multi-label stratified train / val / test split.

    Algorithm: labels are processed in descending order of frequency.  For each
    label, unassigned positive nodes are greedily allocated to whichever split
    still needs the most samples for that label.  After all labels are processed,
    any remaining unassigned nodes are filled into the train and val splits to
    reach their target sizes; the rest go to test.

    This approach does not guarantee perfect stratification but produces splits
    that are approximately proportional for all labels simultaneously.

    Args:
        Y:     Binary multi-label matrix of shape [N, L].
        train: Fraction of nodes assigned to the training split.
        val:   Fraction of nodes assigned to the validation split.
        seed:  Random seed for shuffling within each label.

    Returns:
        Tuple (train_idx, val_idx, test_idx) of integer index arrays.
    """
    rng = np.random.RandomState(seed)
    n, m = Y.shape
    target_tr = int(round(n * train))
    target_va = int(round(n * val))

    label_counts = Y.sum(0).astype(int)
    # Process labels from most frequent to least frequent
    order = np.argsort(-label_counts)
    assign = -np.ones(n, dtype=int)   # -1 = unassigned; 0 = train; 1 = val; 2 = test

    # Pre-compute per-label per-split target counts
    need = np.array([
        [
            int(round(label_counts[j] * train)),
            int(round(label_counts[j] * val)),
            int(label_counts[j]
                - int(round(label_counts[j] * train))
                - int(round(label_counts[j] * val))),
        ]
        for j in range(m)
    ])

    for j in order:
        idx = np.where(Y[:, j] > 0)[0]
        rng.shuffle(idx)
        for split_id in (0, 1, 2):
            k = need[j, split_id]
            if k <= 0:
                continue
            # Only consider nodes not yet assigned
            cand = idx[assign[idx] == -1]
            if cand.size <= k:
                assign[cand] = split_id
                need[j, split_id] -= cand.size
            else:
                pick = cand[:k]
                assign[pick] = split_id
                need[j, split_id] = 0

    def fill_to(split_id: int, target: int):
        """Fill a split to its target size using remaining unassigned nodes."""
        cur = np.sum(assign == split_id)
        if cur >= target:
            return
        left = np.where(assign == -1)[0]
        rng.shuffle(left)
        take = min(left.size, target - cur)
        if take > 0:
            assign[left[:take]] = split_id

    fill_to(0, target_tr)
    fill_to(1, target_va)
    # All remaining unassigned nodes go to the test split
    assign[assign == -1] = 2

    return (
        np.where(assign == 0)[0],
        np.where(assign == 1)[0],
        np.where(assign == 2)[0],
    )
