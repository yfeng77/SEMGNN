"""
GT Registry Module

Centralized registry for all ground-truth (GT) explanation data.
This module provides a process-wide singleton that stores both
label-level and node-level GT masks, along with related metadata.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


class GTRegistry:
    """
    Singleton registry for ground-truth explanation data.

    Responsibilities:
        1. Store all GT-related data (label-level and node-level).
        2. Provide a centralized access point for GT lookup.
        3. Maintain the mapping from node IDs to node-level GT indices.

    Notes:
        - This class follows a singleton pattern, so all callers share
          the same registry instance within the current process.
        - The current implementation is not explicitly synchronized for
          multi-threaded writes.
    """

    _instance = None

    def __new__(cls):
        """Create the singleton instance if it does not already exist."""
        if cls._instance is None:
            cls._instance = super(GTRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the registry only once."""
        if self._initialized:
            return

        # Label-level GT for all labels: shape (L, E_full)
        self.gt_edge_mask: Optional[np.ndarray] = None

        # Mapping from edge tuple (u, v) to edge ID
        self.edge2id: Optional[Dict] = None

        # Node-level GT for explainable nodes: shape (M, L, E_full)
        self.gt_edge_mask_node: Optional[np.ndarray] = None

        # List/array of explainable node IDs: shape (M,)
        self.nodes_expl: Optional[np.ndarray] = None

        # Mapping: node_id -> index in nodes_expl / gt_edge_mask_node
        self.node2gtidx: Optional[Dict[int, int]] = None

        # Full edge list: shape (E_full, 2)
        self.edge_index: Optional[np.ndarray] = None

        # Registry status flag
        self.is_initialized: bool = False

        self._initialized = True

    def set_all(
        self,
        gt_edge_mask: Optional[np.ndarray] = None,
        edge2id: Optional[Dict] = None,
        gt_edge_mask_node: Optional[np.ndarray] = None,
        nodes_expl: Optional[np.ndarray] = None,
        edge_index: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> None:
        """
        Register all GT-related data at once.

        Args:
            gt_edge_mask:
                Label-level GT mask with shape (L, E_full).
            edge2id:
                Mapping from edge tuple (u, v) to edge ID.
            gt_edge_mask_node:
                Node-level GT mask with shape (M, L, E_full).
            nodes_expl:
                Array of explainable node IDs with shape (M,).
            edge_index:
                Full edge list with shape (E_full, 2).
            verbose:
                Whether to print a registration summary.
        """
        self.gt_edge_mask = gt_edge_mask
        self.edge2id = edge2id
        self.gt_edge_mask_node = gt_edge_mask_node
        self.nodes_expl = nodes_expl
        self.edge_index = edge_index

        # Build node_id -> GT index mapping automatically.
        if nodes_expl is not None:
            self.node2gtidx = {
                int(node_id): idx for idx, node_id in enumerate(nodes_expl)
            }
        else:
            self.node2gtidx = None

        self.is_initialized = True

        if verbose:
            print("\n" + "=" * 70)
            print("GT REGISTRY INITIALIZED")
            print("=" * 70)

            if gt_edge_mask is not None:
                print(f"✓ Label-level GT: {gt_edge_mask.shape}")
            if edge2id is not None:
                print(f"✓ Edge mapping: {len(edge2id)} edges")
            if gt_edge_mask_node is not None:
                print(f"✓ Node-level GT: {gt_edge_mask_node.shape}")
            if nodes_expl is not None:
                print(f"✓ Explainable nodes: {len(nodes_expl)}")
            if edge_index is not None:
                print(f"✓ Edge index: {edge_index.shape}")

            print("=" * 70 + "\n")

    def get_node_gt_mask(
        self,
        node_id: int,
        label_ids: Sequence[int],
        fallback_to_label_level: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Get the GT edge mask for a specific node by merging multiple labels.

        The lookup first tries node-level GT. If node-level GT is unavailable
        for the requested node, it can optionally fall back to label-level GT.

        Args:
            node_id:
                Target node ID.
            label_ids:
                Labels to merge into a single GT edge mask.
            fallback_to_label_level:
                If True, use label-level GT when node-level GT is unavailable.

        Returns:
            A boolean array of shape (E_full,) representing the merged GT mask,
            or None if no suitable GT is available.
        """
        # Prefer node-level GT when available for the target node.
        if self.gt_edge_mask_node is not None and self.node2gtidx is not None:
            if node_id in self.node2gtidx:
                idx = self.node2gtidx[node_id]
                gt_this_node = self.gt_edge_mask_node[idx]  # Shape: (L, E_full)

                num_edges = gt_this_node.shape[1]
                node_gt_mask = np.zeros(num_edges, dtype=bool)

                for lbl in label_ids:
                    if 0 <= lbl < gt_this_node.shape[0]:
                        node_gt_mask |= gt_this_node[lbl]

                return node_gt_mask

        # Fall back to label-level GT if requested.
        if fallback_to_label_level and self.gt_edge_mask is not None:
            num_edges = self.gt_edge_mask.shape[1]
            label_gt_mask = np.zeros(num_edges, dtype=bool)

            for lbl in label_ids:
                if 0 <= lbl < self.gt_edge_mask.shape[0]:
                    label_gt_mask |= self.gt_edge_mask[lbl]

            return label_gt_mask

        return None

    def has_node_gt(self, node_id: int) -> bool:
        """
        Check whether a node has node-level GT available.

        Args:
            node_id:
                Target node ID.

        Returns:
            True if node-level GT exists for the node, otherwise False.
        """
        if self.node2gtidx is None:
            return False
        return node_id in self.node2gtidx

    def get_gt_edge_count(self, node_id: int, label_ids: Sequence[int]) -> int:
        """
        Get the number of GT edges for a node and a set of labels.

        Args:
            node_id:
                Target node ID.
            label_ids:
                Labels to merge for GT counting.

        Returns:
            Number of active GT edges. Returns 0 if no GT is available.
        """
        mask = self.get_node_gt_mask(node_id, label_ids)
        if mask is None:
            return 0
        return int(mask.sum())

    def reset(self) -> None:
        """
        Clear all registered GT data.

        This is typically used when switching to a new dataset or experiment.
        """
        self.gt_edge_mask = None
        self.edge2id = None
        self.gt_edge_mask_node = None
        self.nodes_expl = None
        self.node2gtidx = None
        self.edge_index = None
        self.is_initialized = False

        print("[GT Registry] Reset completed")

    def get_info(self) -> dict:
        """
        Return a summary of the current registry state.

        Returns:
            A dictionary containing availability flags and basic counts.
        """
        return {
            "is_initialized": self.is_initialized,
            "has_label_level_gt": self.gt_edge_mask is not None,
            "has_node_level_gt": self.gt_edge_mask_node is not None,
            "has_edge_mapping": self.edge2id is not None,
            "num_explainable_nodes": len(self.nodes_expl) if self.nodes_expl is not None else 0,
            "num_edges": self.gt_edge_mask.shape[1] if self.gt_edge_mask is not None else 0,
            "num_labels": self.gt_edge_mask.shape[0] if self.gt_edge_mask is not None else 0,
        }

    def __repr__(self) -> str:
        """Return a compact string summary of the registry."""
        info = self.get_info()
        return (
            f"GTRegistry("
            f"initialized={info['is_initialized']}, "
            f"nodes={info['num_explainable_nodes']}, "
            f"edges={info['num_edges']}, "
            f"labels={info['num_labels']})"
        )


# ----------------------------------------------------------------------
# Global singleton instance
# ----------------------------------------------------------------------
gt_registry = GTRegistry()


# ----------------------------------------------------------------------
# Convenience functions for backward compatibility
# ----------------------------------------------------------------------
def set_gt_registry(
    gt_edge_mask,
    edge2id,
    gt_edge_mask_node=None,
    nodes_expl=None,
    edge_index=None,
    verbose=True,
) -> None:
    """
    Backward-compatible wrapper for GT registration.

    Existing code can continue using this function. Internally, it delegates
    to the global GTRegistry singleton.
    """
    gt_registry.set_all(
        gt_edge_mask=gt_edge_mask,
        edge2id=edge2id,
        gt_edge_mask_node=gt_edge_mask_node,
        nodes_expl=nodes_expl,
        edge_index=edge_index,
        verbose=verbose,
    )


def get_gt_registry() -> GTRegistry:
    """
    Return the global GTRegistry instance.

    Returns:
        The process-wide GTRegistry singleton.
    """
    return gt_registry