"""
Optimal Threshold Selection for Multi-label Classification
===========================================================
Automatically search for the best decision threshold on a validation set.

Design principles:
  1. Threshold is a hyper-parameter: optimise on val, evaluate on test only.
  2. Different models may converge to different optimal thresholds.
  3. Two strategies are supported:
       - Global threshold:    a single scalar applied to all labels.
       - Per-label threshold: an independent threshold optimised per label.

Supported optimisation metrics:
  micro_f1, macro_f1, micro_precision, macro_precision,
  micro_recall, macro_recall.
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
from sklearn.metrics import f1_score, precision_score, recall_score
from dataclasses import dataclass


@dataclass
class ThresholdSearchResult:
    """
    Container for the result of a threshold grid search.

    Attributes:
        best_threshold:  Optimal threshold found during search.
                         Scalar float for global strategy, or array of shape [L]
                         for per-label strategy.
        best_metric:     Metric value achieved at best_threshold.
        metric_name:     Name of the metric that was optimised.
        threshold_curve: Dict mapping each searched threshold to its metric value.
                         Empty for per-label strategy (too many curves to store).
    """
    best_threshold: float | np.ndarray  # scalar (global) or (L,) (per-label)
    best_metric: float
    metric_name: str
    threshold_curve: Dict[float, float]  # threshold -> metric value

    def __repr__(self):
        if isinstance(self.best_threshold, np.ndarray):
            return (f"ThresholdSearchResult(\n"
                    f"  strategy: per-label,\n"
                    f"  {self.metric_name}: {self.best_metric:.4f},\n"
                    f"  threshold range: [{self.best_threshold.min():.3f}, "
                    f"{self.best_threshold.max():.3f}]\n)")
        else:
            return (f"ThresholdSearchResult(\n"
                    f"  strategy: global,\n"
                    f"  best_threshold: {self.best_threshold:.3f},\n"
                    f"  {self.metric_name}: {self.best_metric:.4f}\n)")


class OptimalThresholdFinder:
    """
    Grid-search the best decision threshold for multi-label classification.

    The search is performed on a validation set; the resulting threshold should
    be applied to the test set via :meth:`apply_threshold`.

    Example usage::

        finder = OptimalThresholdFinder()

        # After training, find the optimal threshold on the validation set
        result = finder.find_optimal_threshold(
            probs=val_probs,      # (N, L) predicted probabilities
            targets=val_targets,  # (N, L) ground-truth binary labels
            strategy="global",    # or "per_label"
            metric="micro_f1",    # or "macro_f1"
        )

        # Apply the optimal threshold when evaluating on the test set
        test_preds = finder.apply_threshold(test_probs, result.best_threshold)

    Args:
        search_range: (min, max) bounds for the threshold grid.
        search_step:  Grid spacing (smaller = finer but slower).
        verbose:      Print progress and best result to stdout.
    """

    def __init__(
        self,
        search_range: Tuple[float, float] = (0.1, 0.9),
        search_step: float = 0.01,
        verbose: bool = True,
    ):
        self.search_range = search_range
        self.search_step = search_step
        self.verbose = verbose

        # Pre-build the candidate threshold list once at construction time
        self.thresholds = np.arange(
            search_range[0],
            search_range[1] + search_step,
            search_step,
        )

    def find_optimal_threshold(
        self,
        probs: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
        strategy: str = "global",
        metric: str = "micro_f1",
    ) -> ThresholdSearchResult:
        """
        Search for the optimal threshold on the provided validation data.

        Args:
            probs:    Predicted sigmoid probabilities, shape [N, L].
            targets:  Binary ground-truth labels, shape [N, L].
            strategy: "global" (shared threshold) or "per_label" (one per label).
            metric:   Metric to maximise: "micro_f1", "macro_f1",
                      "micro_precision", "macro_precision",
                      "micro_recall", "macro_recall".

        Returns:
            ThresholdSearchResult with the best threshold and its metric value.

        Raises:
            ValueError: For unknown strategy or metric names.
        """
        # Convert to NumPy for sklearn compatibility
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        if strategy == "global":
            return self._find_global_threshold(probs, targets, metric)
        elif strategy == "per_label":
            return self._find_per_label_threshold(probs, targets, metric)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _find_global_threshold(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        metric: str,
    ) -> ThresholdSearchResult:
        """
        Grid search for a single global threshold shared across all labels.

        Args:
            probs:   [N, L] probability matrix.
            targets: [N, L] binary target matrix.
            metric:  Metric name to maximise.

        Returns:
            ThresholdSearchResult for the best global threshold.
        """
        best_threshold = 0.5
        best_metric_value = 0.0
        threshold_curve: Dict[float, float] = {}

        if self.verbose:
            print(f"Searching global threshold for {metric}...")
            print(f"  Range: {self.search_range}, Step: {self.search_step}")

        for t in self.thresholds:
            preds = (probs >= t).astype(int)
            metric_value = self._compute_metric(preds, targets, metric)
            threshold_curve[float(t)] = float(metric_value)

            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_threshold = t

        if self.verbose:
            print(f"  Best threshold: {best_threshold:.3f}")
            print(f"  Best {metric}: {best_metric_value:.4f}")

        return ThresholdSearchResult(
            best_threshold=float(best_threshold),
            best_metric=float(best_metric_value),
            metric_name=metric,
            threshold_curve=threshold_curve,
        )

    def _find_per_label_threshold(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        metric: str,
    ) -> ThresholdSearchResult:
        """
        Independently optimise a threshold for each of the L labels.

        Labels with no positive examples in `targets` default to threshold 0.5.
        The reported `best_metric` is the aggregate metric obtained by applying
        all per-label thresholds simultaneously.

        Args:
            probs:   [N, L] probability matrix.
            targets: [N, L] binary target matrix.
            metric:  Metric name to optimise per label.

        Returns:
            ThresholdSearchResult with best_threshold as an array of shape [L].
        """
        N, L = probs.shape
        best_thresholds = np.full(L, 0.5)

        if self.verbose:
            print(f"Searching per-label thresholds for {metric}...")

        # Optimise each label independently
        label_metrics = []
        for label_idx in range(L):
            label_probs   = probs[:, label_idx]
            label_targets = targets[:, label_idx]

            # Skip labels with no positive examples (threshold defaults to 0.5)
            if label_targets.sum() == 0:
                best_thresholds[label_idx] = 0.5
                label_metrics.append(0.0)
                continue

            best_t = 0.5
            best_m = 0.0

            for t in self.thresholds:
                label_preds = (label_probs >= t).astype(int)

                # Binary TP / FP / FN for this single label
                tp = ((label_preds == 1) & (label_targets == 1)).sum()
                fp = ((label_preds == 1) & (label_targets == 0)).sum()
                fn = ((label_preds == 0) & (label_targets == 1)).sum()

                if metric in ["micro_f1", "macro_f1", "f1"]:
                    precision = tp / (tp + fp + 1e-8)
                    recall    = tp / (tp + fn + 1e-8)
                    m = 2 * precision * recall / (precision + recall + 1e-8)
                elif metric == "precision":
                    m = tp / (tp + fp + 1e-8)
                elif metric == "recall":
                    m = tp / (tp + fn + 1e-8)
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                if m > best_m:
                    best_m = m
                    best_t = t

            best_thresholds[label_idx] = best_t
            label_metrics.append(best_m)

        # Evaluate the aggregate metric using all per-label thresholds together
        preds = (probs >= best_thresholds[None, :]).astype(int)
        overall_metric = self._compute_metric(preds, targets, metric)

        if self.verbose:
            print(f"  Threshold range: [{best_thresholds.min():.3f}, "
                  f"{best_thresholds.max():.3f}]")
            print(f"  Overall {metric}: {overall_metric:.4f}")

        return ThresholdSearchResult(
            best_threshold=best_thresholds,
            best_metric=float(overall_metric),
            metric_name=metric,
            threshold_curve={},  # Omitted for per-label (L curves is impractical)
        )

    def _compute_metric(
        self,
        preds: np.ndarray,
        targets: np.ndarray,
        metric: str,
    ) -> float:
        """
        Compute a scalar classification metric from binary predictions.

        Args:
            preds:   [N, L] binary prediction matrix.
            targets: [N, L] binary target matrix.
            metric:  One of: "micro_f1", "macro_f1", "micro_precision",
                     "macro_precision", "micro_recall", "macro_recall".

        Returns:
            Scalar metric value.

        Raises:
            ValueError: If the metric name is not recognised.
        """
        if metric == "micro_f1":
            return f1_score(targets.ravel(), preds.ravel(), zero_division=0)
        elif metric == "macro_f1":
            return f1_score(targets, preds, average='macro', zero_division=0)
        elif metric == "micro_precision":
            return precision_score(targets.ravel(), preds.ravel(), zero_division=0)
        elif metric == "macro_precision":
            return precision_score(targets, preds, average='macro', zero_division=0)
        elif metric == "micro_recall":
            return recall_score(targets.ravel(), preds.ravel(), zero_division=0)
        elif metric == "macro_recall":
            return recall_score(targets, preds, average='macro', zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    @staticmethod
    def apply_threshold(
        probs: torch.Tensor | np.ndarray,
        threshold: float | np.ndarray,
    ) -> np.ndarray:
        """
        Binarise predicted probabilities using a threshold.

        Args:
            probs:     Predicted probabilities, shape [N, L].
            threshold: Scalar (global) or array of shape [L] (per-label).

        Returns:
            Binary prediction matrix of shape [N, L] with dtype int.
        """
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()

        if isinstance(threshold, (int, float)):
            # Global threshold: broadcast over all labels
            return (probs >= threshold).astype(int)
        else:
            # Per-label threshold: broadcast over all nodes
            threshold = np.array(threshold)
            return (probs >= threshold[None, :]).astype(int)

