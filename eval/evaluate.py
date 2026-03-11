"""
evaluate.py

Evaluation utilities for the SEMGNN pipeline.

Three evaluator classes are provided:

  1. PredictionEvaluator
     Standard multi-label classification metrics (Micro/Macro F1, AUPRC,
     Hamming loss, etc.) computed over a masked node set.

  2. GTExplanationEvaluator
     Explanation quality metrics against ground-truth edge masks stored in
     gt_registry.  Uses the per-label full-graph masks produced by
     SelfExplainableGNN.compute_full_graph_edge_masks_multilabel and evaluates
     Precision, Recall, IoU, F1, and Average Precision per (node, label) pair.

  3. FidelityEvaluator
     Fidelity-based explanation quality metrics for settings where ground-truth
     masks are unavailable.  Computes Fidelity+ (keep important edges) and
     Fidelity- (remove important edges) per label, using the same explanation
     pipeline as GTExplanationEvaluator for consistency.

Standalone helpers:
  - compute_adaptive_topM: Compute an adaptive top-M edge selection count based
    on subgraph size.
  - normalize_label_masks_to_undirected: Convert directed per-label edge masks
    to undirected edge masks using gt_registry.dir2undir.
"""

from __future__ import annotations
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm

from utils.gt_registry import gt_registry
from SEMGNN import SelfExplainableGNN
from torch_geometric.utils import k_hop_subgraph
import math

from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    hamming_loss,
    average_precision_score,
)
import time
import warnings


# ============================================================
# Standalone helpers
# ============================================================


def compute_adaptive_topM(
    G: int,
    min_edges: int = 12,
    max_frac: float = 0.30,
    cap: int = 500,
) -> int:
    """
    Compute an adaptive top-M edge selection count for a subgraph of size G.

    The formula scales sub-linearly with G:
      - For small subgraphs (G <= 2000): target = max(min_edges, 0.15 * G)
      - For large subgraphs (G > 2000):  target = max(100, 5 * sqrt(G))

    The result is further clamped so that:
      - It does not exceed max_frac * G (prevents over-selecting in dense subgraphs).
      - It is at least min_edges.
      - It does not exceed cap or G itself.

    Args:
        G:         Number of (undirected) edges in the subgraph.
        min_edges: Minimum number of edges to always select.
        max_frac:  Maximum fraction of G that can be selected.
        cap:       Hard upper bound on the number of selected edges.

    Returns:
        Integer top-M value in [min_edges, min(cap, G)].
    """
    G = int(G)
    if G <= 0:
        return 0
    if G <= 2000:
        target = max(min_edges, int(G * 0.15))
    else:
        target = max(100, int(math.sqrt(G) * 5))
    topM = min(target, int(math.ceil(max_frac * G)))
    topM = max(topM, min_edges)
    topM = min(topM, cap, G)
    return int(topM)


def normalize_label_masks_to_undirected(
    label_masks: Dict[int, torch.Tensor],
    edge_index: torch.Tensor,
) -> Dict[int, torch.Tensor]:
    """
    Convert per-label directed edge masks to undirected edge masks.

    If a mask already has E_undir entries it is passed through unchanged.
    If it has E_dir entries, directed scores are averaged into their
    corresponding undirected edge using gt_registry.dir2undir.

    The function is a no-op when gt_registry.dir2undir is not available.

    Args:
        label_masks: Dict mapping label_id -> edge mask tensor.
                     Each mask can be of shape [E_dir] or [E_undir].
        edge_index:  Directed edge index [2, E_dir] (used to determine E_dir).

    Returns:
        Dict mapping label_id -> undirected edge mask tensor of shape [E_undir].

    Raises:
        ValueError: If a mask size matches neither E_dir nor E_undir.
    """
    E_dir = edge_index.size(1)
    E_undir = gt_registry.edge_index.shape[0]
    dir2undir = getattr(gt_registry, "dir2undir", None)
    if dir2undir is None:
        return label_masks

    d2u = torch.from_numpy(dir2undir).to(edge_index.device).long()

    out = {}
    for lid, m in label_masks.items():
        if m.numel() == E_undir:
            # Already undirected; pass through
            out[lid] = m
        elif m.numel() == E_dir:
            # Average directed mask scores into undirected edges
            s = torch.zeros(E_undir, device=m.device, dtype=m.dtype)
            c = torch.zeros(E_undir, device=m.device, dtype=m.dtype)
            s.scatter_add_(0, d2u, m)
            c.scatter_add_(0, d2u, torch.ones_like(m))
            out[lid] = s / c.clamp(min=1)
        else:
            raise ValueError(
                f"Label {lid} mask size {m.numel()} != E_dir {E_dir} and != E_undir {E_undir}"
            )
    return out


# ============================================================
# 1. Prediction quality evaluator
# ============================================================


class PredictionEvaluator:
    """
    Evaluate multi-label classification performance on a masked node set.

    Computes both threshold-dependent metrics (F1, precision, recall,
    Hamming loss, subset accuracy) and threshold-free metrics (AUPRC / mAP).

    Args:
        num_labels: Total number of labels L.
    """

    def __init__(self, num_labels: int):
        self.num_labels = num_labels

    @torch.no_grad()
    def evaluate(
        self,
        model: SelfExplainableGNN,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        Y: torch.Tensor,
        eval_mask: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict:
        """
        Run a full forward pass and compute prediction quality metrics.

        Args:
            model:      Trained SelfExplainableGNN instance.
            X:          Node feature matrix [N, in_dim].
            edge_index: Edge indices [2, E].
            Y:          Binary label targets [N, L].
            eval_mask:  Boolean node mask [N] selecting nodes to evaluate.
            threshold:  Decision threshold applied to sigmoid probabilities.

        Returns:
            Dict with keys:
              - micro_f1, macro_f1: Micro / macro F1 scores.
              - micro_precision, micro_recall: Micro precision and recall.
              - macro_precision, macro_recall: Macro precision and recall.
              - hamming_loss: Hamming loss (lower is better).
              - subset_accuracy: Fraction of nodes with exact label match.
              - micro_auprc, macro_auprc: AUPRC / mAP scores.
              - num_samples: Number of evaluated nodes.
              - per_label: List of per-label metric dicts.
              - worst_labels / best_labels: 5 labels with lowest / highest F1.
        """
        model.eval()

        result = model(X, edge_index, return_explanation=False)
        logits = result["logits"][eval_mask]
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= threshold).astype(int)
        targets = Y[eval_mask].cpu().numpy().astype(int)

        # --- Aggregate metrics ---
        p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
            targets, preds, average="micro", zero_division=0
        )
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            targets, preds, average="macro", zero_division=0
        )

        hl = hamming_loss(targets, preds)
        subset_acc = float(np.mean(np.all(preds == targets, axis=1)))

        # Threshold-free area under precision-recall curve
        # Suppress sklearn warnings for labels with no positive samples
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            micro_auprc = average_precision_score(targets, probs, average="micro")
            macro_auprc = average_precision_score(targets, probs, average="macro")

        # --- Per-label metrics ---
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, preds, average=None, zero_division=0
        )

        per_label = []
        for i in range(self.num_labels):
            per_label.append({
                "label": i,
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            })

        # Sort by F1 to surface the most / least reliable labels
        per_label_sorted = sorted(per_label, key=lambda x: x["f1"])

        return {
            "micro_f1": float(f1_micro),
            "macro_f1": float(f1_macro),
            "micro_precision": float(p_micro),
            "micro_recall": float(r_micro),
            "macro_precision": float(p_macro),
            "macro_recall": float(r_macro),
            "hamming_loss": float(hl),
            "subset_accuracy": float(subset_acc),
            "micro_auprc": float(micro_auprc),
            "macro_auprc": float(macro_auprc),
            "num_samples": int(eval_mask.sum()),
            "per_label": per_label,
            "worst_labels": per_label_sorted[:5],
            "best_labels": per_label_sorted[-5:][::-1],
        }


# ============================================================
# 2. GT-based explanation quality evaluator
# ============================================================


class GTExplanationEvaluator:
    """
    Evaluate explanation quality against ground-truth edge masks (from gt_registry).

    For each node the evaluator:
      1. Runs model.explain_node_with_per_label_mask to obtain per-label important
         edge sets.
      2. Maps predicted edges and GT edges onto the canonical undirected edge space.
      3. Restricts GT edges to those visible in the k-hop subgraph (fair evaluation).
      4. Computes Precision, Recall, IoU, F1, and Average Precision per (node, label).
      5. Aggregates macro-average metrics across labels, then across nodes.

    Pre-computed per-label full-graph edge masks can be passed in via the
    `label_masks` argument to avoid redundant computation across nodes.

    Args:
        min_edges:   Minimum edges selected per explanation (passed to topM).
        max_frac:    Maximum fraction of subgraph edges to select.
        cap:         Hard cap on selected edges per explanation.
        k_for_size:  Hop count used to determine subgraph size (should match k_hop).
    """

    def __init__(
        self,
        min_edges: int = 20,
        max_frac: float = 0.40,
        cap: int = 500,
        k_for_size: int = 3,
    ):
        self.min_edges = int(min_edges)
        self.max_frac = float(max_frac)
        self.cap = int(cap)
        self.k_for_size = int(k_for_size)

    @torch.no_grad()
    def evaluate_single_node(
        self,
        model: SelfExplainableGNN,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        Y: torch.Tensor,
        node_idx: int,
        label_ids: List[int],
        k_hop: int = 3,
        label_masks: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict:
        """
        Evaluate GT explanation quality for a single node across multiple labels.

        Args:
            model:       Trained SelfExplainableGNN.
            X:           Node features [N, in_dim].
            edge_index:  Edge indices [2, E].
            Y:           Binary label targets [N, L].
            node_idx:    Target node index.
            label_ids:   Labels to evaluate (typically the model's predicted labels).
            k_hop:       Number of hops for subgraph extraction.
            label_masks: Pre-computed {label_id: undirected_edge_mask [E_undir]}.
                         If provided, masks are used for AP computation and skip
                         recomputation.

        Returns:
            Dict with keys:
              - status: "ok" | "not_initialized".
              - per_label_metrics: {label_id: {precision, recall, iou, f1, ap, ...}}.
              - macro_average: Mean metrics across labels.
              - unified_metrics: Metrics treating the union of all labels' GT/pred.
              - num_labels_evaluated, num_pred_labels, coverage.
        """
        if not gt_registry.is_initialized:
            return {"status": "not_initialized"}

        node_idx = int(node_idx)

        explanation = model.explain_node_with_per_label_mask(
            X, edge_index,
            node_idx=node_idx,
            k_hop=k_hop,
            label_ids=label_ids,
            label_masks=label_masks,
            Y=Y,
            train_mask=None,   # Not needed at inference time
            pos_weight=None,
            min_edges=self.min_edges,
            max_frac=self.max_frac,
            cap=self.cap,
        )

        per_label_explanations = explanation['per_label_explanations']
        edge2id = gt_registry.edge2id

        # ---- Build the set of undirected edge IDs visible in the k-hop subgraph ----
        # This is used to restrict GT evaluation to edges the model could have seen.
        N = X.size(0)
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=k_hop,
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=N,
        )

        edge2id = gt_registry.edge2id
        sub_undir_eids: set[int] = set()
        # The sorted list is needed later for AP computation (consistent ordering)
        sub_eids_list: List[int] = []

        E_sub = sub_edge_index.size(1)
        for i in range(E_sub):
            u_sub = int(sub_edge_index[0, i])
            v_sub = int(sub_edge_index[1, i])
            u_full = int(subset[u_sub])
            v_full = int(subset[v_sub])
            a, b = (u_full, v_full) if u_full < v_full else (v_full, u_full)
            eid = edge2id.get((a, b), None)
            if eid is not None:
                sub_undir_eids.add(int(eid))

        sub_eids_list = sorted(list(sub_undir_eids))
        sub_eids_t = (
            torch.tensor(sub_eids_list, device=X.device, dtype=torch.long)
            if sub_eids_list else None
        )

        # ---- Per-label evaluation ----
        per_label_metrics = {}

        for label_id in label_ids:
            # Fetch GT mask for this specific label
            gt_mask_label = gt_registry.get_node_gt_mask(
                node_id=node_idx,
                label_ids=[label_id],
                fallback_to_label_level=True,
            )

            if gt_mask_label is None or label_id not in per_label_explanations:
                continue

            # Restrict GT to edges visible in the k-hop subgraph (fair evaluation)
            gt_edges = set(np.where(gt_mask_label)[0].tolist()) & sub_undir_eids
            if len(gt_edges) == 0:
                continue

            # Map predicted (u, v) pairs to canonical undirected edge IDs
            pred_edges_raw = per_label_explanations[label_id]['important_edges']
            pred_edges_global: set[int] = set()
            for u, v in pred_edges_raw:
                u, v = int(u), int(v)
                if u > v:
                    u, v = v, u
                eid = edge2id.get((u, v), None)
                if eid is not None:
                    pred_edges_global.add(eid)

            # Compute standard retrieval metrics
            tp        = len(pred_edges_global & gt_edges)
            pred_pos  = len(pred_edges_global)
            gt_pos    = len(gt_edges)
            union     = len(pred_edges_global | gt_edges)

            precision = tp / pred_pos if pred_pos > 0 else 0.0
            recall    = tp / gt_pos   if gt_pos   > 0 else 0.0
            iou       = tp / union    if union     > 0 else 0.0
            f1        = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0.0
            )

            # Average Precision over subgraph edges (requires continuous mask scores)
            ap = float("nan")
            if (
                label_masks is not None
                and sub_eids_t is not None
                and label_id in label_masks
            ):
                # Extract continuous mask scores for the subgraph's undirected edges
                scores = label_masks[label_id][sub_eids_t].detach().cpu().numpy()
                y_true = gt_mask_label[sub_eids_list].astype(int)
                if len(np.unique(y_true)) >= 2:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ap = float(average_precision_score(y_true, scores))

            per_label_metrics[label_id] = {
                'precision': float(precision),
                'recall':    float(recall),
                'iou':       float(iou),
                'f1':        float(f1),
                'ap':        float(ap),
                'tp':        int(tp),
                'pred_edges': int(pred_pos),
                'gt_edges':   int(gt_pos),
            }

        # ---- Macro-average across labels ----
        if per_label_metrics:
            ap_list = [
                m['ap'] for m in per_label_metrics.values()
                if np.isfinite(m['ap'])
            ]
            macro_average = {
                'precision': float(np.mean([m['precision'] for m in per_label_metrics.values()])),
                'recall':    float(np.mean([m['recall']    for m in per_label_metrics.values()])),
                'iou':       float(np.mean([m['iou']       for m in per_label_metrics.values()])),
                'f1':        float(np.mean([m['f1']        for m in per_label_metrics.values()])),
                'ap':        float(np.mean(ap_list)) if ap_list else float("nan"),
            }
        else:
            macro_average = {}

        # ---- Unified (label-union) metrics ----
        # All GT edges across every label, vs. the union of all predicted edges
        all_gt_edges: set[int] = set()
        for label_id in label_ids:
            gt_mask = gt_registry.get_node_gt_mask(node_idx, [label_id], True)
            if gt_mask is not None:
                all_gt_edges.update(
                    set(np.where(gt_mask)[0].tolist()) & sub_undir_eids
                )

        all_pred_edges: set[int] = set()
        for label_exp in per_label_explanations.values():
            for u, v in label_exp['important_edges']:
                u, v = int(u), int(v)
                if u > v:
                    u, v = v, u
                eid = edge2id.get((u, v), None)
                if eid is not None:
                    all_pred_edges.add(eid)

        tp_unified = len(all_pred_edges & all_gt_edges)
        unified_metrics = {
            'precision': tp_unified / len(all_pred_edges) if all_pred_edges else 0.0,
            'recall':    tp_unified / len(all_gt_edges)   if all_gt_edges   else 0.0,
            'iou':       tp_unified / len(all_pred_edges | all_gt_edges)
                         if (all_pred_edges | all_gt_edges) else 0.0,
        }

        num_pred  = len(label_ids)
        num_eval  = len(per_label_metrics)
        coverage  = num_eval / max(num_pred, 1)

        return {
            'status': 'ok',
            'node_idx': node_idx,
            'per_label_metrics': per_label_metrics,
            'macro_average': macro_average,
            'unified_metrics': unified_metrics,
            'num_labels_evaluated': int(num_eval),
            'num_pred_labels': int(num_pred),
            'coverage': float(coverage),
        }

    @torch.no_grad()
    def evaluate_batch(
        self,
        model: SelfExplainableGNN,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        Y: torch.Tensor,
        node_indices: List[int],
        k_hop: int = 3,
        verbose: bool = True,
        return_per_node: bool = False,
        train_mask: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None,
        label_masks: Optional[Dict[int, torch.Tensor]] = None,
        threshold: float = 0.5,
    ) -> Dict:
        """
        Evaluate GT explanation quality over a batch of nodes.

        Pre-computes full-graph per-label masks if not already provided, then
        evaluates each node and aggregates macro-average statistics.

        Args:
            model:           Trained SelfExplainableGNN.
            X:               Node features [N, in_dim].
            edge_index:      Edge indices [2, E].
            Y:               Binary label targets [N, L].
            node_indices:    List of node indices to evaluate.
            k_hop:           Number of hops for subgraph extraction.
            verbose:         Show a tqdm progress bar.
            return_per_node: Include per-node result dicts in the output.
            train_mask:      Boolean node mask used for gradient guidance.
            pos_weight:      Per-label positive class weights [L].
            label_masks:     Pre-computed {label_id: undirected_mask}.
                             Masks for missing labels are computed automatically.
            threshold:       Sigmoid threshold for determining predicted labels.

        Returns:
            Dict with aggregated metrics (mean ± std for precision, recall, IoU,
            F1, AP, coverage, and runtime), plus optional per-node results.
        """
        if not gt_registry.is_initialized:
            return {"status": "not_initialized"}

        model.eval()
        results = []
        times = []
        coverages = []
        aps = []

        print("Preparing per-label full graph masks for GT evaluation...")

        result_all = model(X, edge_index, return_explanation=False)
        probs_all = torch.sigmoid(result_all["logits"]).detach()

        # Collect the union of all labels predicted for any eval node
        active_labels_needed: set[int] = set()
        for node_idx in node_indices:
            node_idx = int(node_idx)
            probs = probs_all[node_idx].cpu().numpy()
            pred_labels = np.where(probs >= threshold)[0].tolist()
            if not pred_labels:
                pred_labels = [int(np.argmax(probs))]
            active_labels_needed.update(pred_labels)

        active_labels_needed_sorted = sorted(list(active_labels_needed))
        print(f"Active labels needed: {active_labels_needed_sorted}")

        if label_masks is None:
            label_masks = {}

        # Normalise any provided masks to undirected representation
        label_masks = normalize_label_masks_to_undirected(label_masks, edge_index)

        # Compute masks for labels not already covered
        missing = [l for l in active_labels_needed_sorted if l not in label_masks]
        if missing:
            print(f"Computing missing masks for {len(missing)} labels ...")
            computed = model.compute_full_graph_edge_masks_multilabel(
                X, edge_index, missing,
                Y=Y,
                train_mask=train_mask,
                pos_weight=pos_weight,
            )
            computed = normalize_label_masks_to_undirected(computed, edge_index)
            label_masks.update(computed)

        print(f"Using per-label masks: {len(label_masks)}")
        active_labels = active_labels_needed_sorted

        iterator = (
            tqdm(node_indices, desc="Evaluating GT (Per-Label)")
            if verbose else node_indices
        )

        for node_idx in iterator:
            node_idx = int(node_idx)

            probs = probs_all[node_idx].cpu().numpy()
            pred_labels = np.where(probs >= threshold)[0].tolist()
            if not pred_labels:
                pred_labels = [int(np.argmax(probs))]

            t0 = time.perf_counter()
            m = self.evaluate_single_node(
                model=model,
                X=X,
                edge_index=edge_index,
                Y=Y,
                node_idx=node_idx,
                label_ids=pred_labels,
                k_hop=k_hop,
                label_masks=label_masks,
            )
            t1 = time.perf_counter()

            if m.get("status") == "ok":
                times.append(t1 - t0)
                coverages.append(m.get("coverage", float("nan")))
                if (
                    m.get("macro_average")
                    and "ap" in m["macro_average"]
                    and np.isfinite(m["macro_average"]["ap"])
                ):
                    aps.append(m["macro_average"]["ap"])
                results.append(m)

        if not results:
            return {"status": "no_valid_nodes"}

        all_precision = []
        all_recall = []
        all_iou = []
        all_f1 = []

        for r in results:
            if r.get('macro_average'):
                all_precision.append(r['macro_average']['precision'])
                all_recall.append(r['macro_average']['recall'])
                all_iou.append(r['macro_average']['iou'])
                all_f1.append(r['macro_average']['f1'])

        summary = {
            "status": "ok",
            "num_evaluated": int(len(results)),
            "macro_precision_mean": float(np.mean(all_precision)) if all_precision else 0.0,
            "macro_precision_std":  float(np.std(all_precision))  if all_precision else 0.0,
            "macro_recall_mean":    float(np.mean(all_recall))    if all_recall    else 0.0,
            "macro_recall_std":     float(np.std(all_recall))     if all_recall    else 0.0,
            "macro_iou_mean":       float(np.mean(all_iou))       if all_iou       else 0.0,
            "macro_iou_std":        float(np.std(all_iou))        if all_iou       else 0.0,
            "macro_f1_mean":        float(np.mean(all_f1))        if all_f1        else 0.0,
            "macro_f1_std":         float(np.std(all_f1))         if all_f1        else 0.0,
            "num_labels": len(active_labels),
            "macro_ap_mean": float(np.mean(aps)) if aps else float("nan"),
            "macro_ap_std":  float(np.std(aps))  if aps else float("nan"),
            "coverage_mean": float(np.nanmean(coverages)) if coverages else float("nan"),
            "coverage_std":  float(np.nanstd(coverages))  if coverages else float("nan"),
            "runtime_sec_per_node_mean": float(np.mean(times)) if times else float("nan"),
            "runtime_sec_per_node_std":  float(np.std(times))  if times else float("nan"),
        }

        if return_per_node:
            summary["per_node"] = results

        return summary


# ============================================================
# 3. Fidelity-based explanation evaluator (no GT required)
# ============================================================


class FidelityEvaluator:
    """
    Evaluate explanation quality via Fidelity metrics (no ground-truth required).

    Core idea (per label):
      - **Fidelity+**: Re-encode the k-hop subgraph using only the selected
        (important) edges as active weights (others zeroed).  The predicted
        probability for the target label should remain close to the full-graph
        baseline.  Fidelity+ = 1 - MSE(p_keep, p_full).
      - **Fidelity-**: Re-encode the subgraph using only the *complement* of the
        selected edges (important edges zeroed).  The predicted probability should
        drop, indicating that the selected edges were truly necessary.
        Fidelity- = p_full - p_drop.

    Edge selection uses the same pipeline as GTExplanationEvaluator
    (explain_node_with_per_label_mask) to ensure comparability between the two
    evaluation modes.

    Args:
        min_edges: Minimum edges selected per label per explanation.
        max_frac:  Maximum fraction of subgraph edges to select.
        cap:       Hard cap on selected edges per explanation.
    """

    def __init__(self, min_edges: int = 12, max_frac: float = 0.30, cap: int = 200):
        self.min_edges = min_edges
        self.max_frac = max_frac
        self.cap = cap

    @torch.no_grad()
    def evaluate_single_node(
        self,
        model: SelfExplainableGNN,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        node_idx: int,
        k_hop: int = 3,
        threshold: float = 0.5,
        label_masks: Optional[Dict[int, torch.Tensor]] = None,
        Y: Optional[torch.Tensor] = None,
        train_mask: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None,
        use_node_local_necessity: bool = True,
        alpha: float = 0.6,
    ) -> Dict:
        """
        Compute Fidelity+ and Fidelity- for a single node across its predicted labels.

        The method shares the same explanation source as GTExplanationEvaluator
        (explain_node_with_per_label_mask) so that the two evaluators are
        directly comparable.

        Args:
            model:                   Trained SelfExplainableGNN.
            X:                       Node features [N, in_dim].
            edge_index:              Edge indices [2, E].
            node_idx:                Target node index.
            k_hop:                   Number of hops for subgraph extraction.
            threshold:               Sigmoid threshold for predicted labels.
            label_masks:             Pre-computed {label_id: undirected_mask}.
            Y:                       Binary targets [N, L] (for gradient guidance).
            train_mask:              Boolean node mask (for gradient guidance).
            pos_weight:              Per-label positive class weights [L].
            use_node_local_necessity:Whether to blend local necessity into the mask.
            alpha:                   Global vs. local necessity blend weight.

        Returns:
            Dict with keys:
              - status: "ok" | "no_edges" | "no_explanation" | "no_valid_labels".
              - macro_fidelity_pos, macro_fidelity_neg: Label-averaged fidelity.
              - sparsity: Mean fraction of edges NOT selected.
              - edge_mask_mean: Mean mask score over subgraph edges.
              - per_label: Per-label fidelity details.
              - G_sub_undirected: Number of undirected edges in the subgraph.
        """
        model.eval()
        device = X.device
        N = X.size(0)
        node_idx = int(node_idx)

        # --- Determine labels to explain (consistent with GT evaluator) ---
        res_fullgraph = model(X, edge_index, return_explanation=False)
        probs_node = torch.sigmoid(res_fullgraph["logits"][node_idx]).detach()
        pred_labels = (probs_node >= threshold).nonzero(as_tuple=True)[0].tolist()
        if not pred_labels:
            pred_labels = [int(torch.argmax(probs_node))]

        # --- Extract k-hop subgraph for keep/drop forward passes ---
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=k_hop,
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=N,
        )
        node_l = int(mapping.item())
        x_sub = X[subset]
        E_sub = sub_edge_index.size(1)
        if E_sub == 0:
            return {"status": "no_edges"}

        # Baseline: full subgraph prediction (all edges active)
        logits_sub_full = model.predict(model.encode(x_sub, sub_edge_index))
        probs_sub_full = torch.sigmoid(logits_sub_full[node_l]).detach()

        # --- Obtain per-label edge explanations ---
        expl = model.explain_node_with_per_label_mask(
            X=X,
            edge_index=edge_index,
            node_idx=node_idx,
            k_hop=k_hop,
            label_ids=pred_labels,
            label_masks=label_masks,
            Y=Y,
            train_mask=train_mask,
            pos_weight=pos_weight,
            min_edges=self.min_edges,
            max_frac=self.max_frac,
            cap=self.cap,
            use_node_local_necessity=use_node_local_necessity,
            alpha=alpha,
        )
        per_label_exps = expl.get("per_label_explanations", {})
        if not per_label_exps:
            return {"status": "no_explanation"}

        # --- Group directed subgraph edges into canonical undirected (u<v) keys ---
        groups: Dict[Tuple[int, int], List[int]] = {}
        for i in range(E_sub):
            u_sub = int(sub_edge_index[0, i])
            v_sub = int(sub_edge_index[1, i])
            u_full = int(subset[u_sub])
            v_full = int(subset[v_sub])
            key = (min(u_full, v_full), max(u_full, v_full))
            groups.setdefault(key, []).append(i)

        G_sub = len(groups)
        if G_sub == 0:
            return {"status": "no_edges"}

        # Build a tensor of subgraph undirected edge IDs for mask diagnostics
        has_gt_edge2id = (
            gt_registry.is_initialized and (gt_registry.edge2id is not None)
        )
        sub_eids_t = None
        if has_gt_edge2id:
            sub_eids = [
                int(gt_registry.edge2id[key])
                for key in groups.keys()
                if key in gt_registry.edge2id
            ]
            if sub_eids:
                sub_eids_t = torch.tensor(sub_eids, device=device, dtype=torch.long)

        # --- Per-label Fidelity computation ---
        per_label_fidelity: Dict[int, Dict] = {}

        for lid in pred_labels:
            if lid not in per_label_exps:
                continue

            sel_undir = per_label_exps[lid]["important_edges"]  # List of [u, v]
            num_sel_undir = len(sel_undir)
            sparsity_l = 1.0 - (num_sel_undir / float(max(G_sub, 1)))

            # Diagnostic: mean mask score over subgraph edges for this label
            edge_mask_mean_l = float("nan")
            edge_mask_std_l  = float("nan")
            if (
                has_gt_edge2id
                and label_masks is not None
                and sub_eids_t is not None
                and lid in label_masks
            ):
                vals = label_masks[lid][sub_eids_t]
                edge_mask_mean_l = vals.mean().item()
                edge_mask_std_l  = vals.std(unbiased=False).item()

            # Map selected undirected edges back to directed subgraph indices
            selected_directed = []
            for u, v in sel_undir:
                key = (min(int(u), int(v)), max(int(u), int(v)))
                if key in groups:
                    selected_directed.extend(groups[key])

            if len(selected_directed) == 0:
                continue
            selected_directed = torch.tensor(
                selected_directed, device=device, dtype=torch.long
            )

            # Binary edge weight vectors for keep and drop passes
            keep_w = torch.zeros(E_sub, device=device, dtype=x_sub.dtype)
            keep_w[selected_directed] = 1.0
            drop_w = torch.ones(E_sub, device=device, dtype=x_sub.dtype)
            drop_w[selected_directed] = 0.0

            # Forward passes with keep / drop masks
            logits_keep = model.predict(
                model.encode(x_sub, sub_edge_index, edge_weight=keep_w)
            )
            logits_drop = model.predict(
                model.encode(x_sub, sub_edge_index, edge_weight=drop_w)
            )

            p_keep = torch.sigmoid(logits_keep[node_l, lid])
            p_drop = torch.sigmoid(logits_drop[node_l, lid])
            p_full = probs_sub_full[lid]

            # Fidelity+: how well the explanation alone recovers the full prediction
            fid_pos = 1.0 - F.mse_loss(p_keep, p_full).item()
            # Fidelity-: how much the prediction drops when the explanation is removed
            fid_neg = (p_full - p_drop).item()

            per_label_fidelity[lid] = {
                "fidelity_pos":       float(fid_pos),
                "fidelity_neg":       float(fid_neg),
                "num_edges_selected": int(num_sel_undir),
                "sparsity":           float(sparsity_l),
                "edge_mask_mean":     float(edge_mask_mean_l),
                "edge_mask_std":      float(edge_mask_std_l),
            }

        if not per_label_fidelity:
            return {"status": "no_valid_labels"}

        fid_pos_list = [v["fidelity_pos"] for v in per_label_fidelity.values()]
        fid_neg_list = [v["fidelity_neg"] for v in per_label_fidelity.values()]
        spars_list   = [v["sparsity"]     for v in per_label_fidelity.values()]
        emm_list = [
            v["edge_mask_mean"] for v in per_label_fidelity.values()
            if np.isfinite(v["edge_mask_mean"])
        ]

        return {
            "status":             "ok",
            "node_idx":           node_idx,
            "pred_labels":        pred_labels,
            "macro_fidelity_pos": float(np.mean(fid_pos_list)),
            "macro_fidelity_neg": float(np.mean(fid_neg_list)),
            "sparsity":           float(np.mean(spars_list)),
            "edge_mask_mean":     float(np.mean(emm_list)) if emm_list else float("nan"),
            "per_label":          per_label_fidelity,
            "G_sub_undirected":   int(G_sub),
        }

    @torch.no_grad()
    def evaluate_batch(
        self,
        model: SelfExplainableGNN,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        node_indices: List[int],
        k_hop: int = 3,
        threshold: float = 0.5,
        verbose: bool = True,
        label_masks: Optional[Dict[int, torch.Tensor]] = None,
        Y: Optional[torch.Tensor] = None,
        train_mask: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None,
        use_node_local_necessity: bool = True,
        alpha: float = 0.6,
    ) -> Dict:
        """
        Evaluate Fidelity metrics over a batch of nodes.

        Args:
            model:                   Trained SelfExplainableGNN.
            X:                       Node features [N, in_dim].
            edge_index:              Edge indices [2, E].
            node_indices:            List of node indices to evaluate.
            k_hop:                   Number of hops for subgraph extraction.
            threshold:               Sigmoid threshold for predicted labels.
            verbose:                 Show a tqdm progress bar.
            label_masks:             Pre-computed per-label undirected masks.
            Y:                       Binary targets (for gradient guidance).
            train_mask:              Boolean node mask (for gradient guidance).
            pos_weight:              Per-label positive class weights.
            use_node_local_necessity:Whether to blend local necessity scores.
            alpha:                   Global vs. local necessity blend weight.

        Returns:
            Dict with aggregated Fidelity+/-, sparsity, edge_mask_mean, and
            runtime statistics (mean ± std).
        """
        model.eval()
        results = []
        times = []
        iterator = (
            tqdm(node_indices, desc="Evaluating Fidelity (Aligned)")
            if verbose else node_indices
        )

        for node_idx in iterator:
            t0 = time.perf_counter()
            m = self.evaluate_single_node(
                model=model, X=X, edge_index=edge_index,
                node_idx=int(node_idx),
                k_hop=k_hop, threshold=threshold,
                label_masks=label_masks, Y=Y,
                train_mask=train_mask, pos_weight=pos_weight,
                use_node_local_necessity=use_node_local_necessity,
                alpha=alpha,
            )
            t1 = time.perf_counter()
            if m.get("status") == "ok":
                times.append(t1 - t0)
                results.append(m)

        if not results:
            return {"status": "no_valid_nodes"}

        return {
            "status":        "ok",
            "num_evaluated": len(results),
            "fidelity_pos_mean": float(np.mean([r["macro_fidelity_pos"] for r in results])),
            "fidelity_pos_std":  float(np.std ([r["macro_fidelity_pos"] for r in results])),
            "fidelity_neg_mean": float(np.mean([r["macro_fidelity_neg"] for r in results])),
            "fidelity_neg_std":  float(np.std ([r["macro_fidelity_neg"] for r in results])),
            "sparsity_mean": float(np.mean([r["sparsity"] for r in results])),
            "sparsity_std":  float(np.std ([r["sparsity"] for r in results])),
            "edge_mask_mean": (
                float(np.mean([
                    r["edge_mask_mean"] for r in results
                    if np.isfinite(r["edge_mask_mean"])
                ]))
                if any(np.isfinite(r["edge_mask_mean"]) for r in results)
                else float("nan")
            ),
            "per_node": results,
            "runtime_sec_per_node_mean": float(np.mean(times)) if times else float("nan"),
            "runtime_sec_per_node_std":  float(np.std(times))  if times else float("nan"),
        }