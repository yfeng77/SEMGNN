"""
auto_tune.py
===============
Automated hyperparameter search script for SEMGNN.

Usage:
    python auto_tune.py \\
        --datasets synthetic_motif synthetic_anchor blogcatalog youtube \\
        --ckpt_dir ckpts \\
        --n_trials 40 \\
        --device cuda

Each dataset is tuned independently. Results are saved under
tune_logs/<dataset>/. After the search, the best parameters are used
to retrain and save the final model.

Requirements:
    pip install optuna
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    auc,
    f1_score,
    hamming_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    accuracy_score,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("Please install optuna first: pip install optuna")
    sys.exit(1)

from data_process.data_process import load_for_training
from SEMGNN import SelfExplainableGNN, GradientGuidedLoss
from utils.utils import set_seed, sparse_to_edge_index, multilabel_stratified_split
from eval.evaluate import FidelityEvaluator, GTExplanationEvaluator
from utils.gt_registry import gt_registry
from utils.optimal_threshold import OptimalThresholdFinder
from train_SEMGNN import (
    build_static_label_adj,
    compute_loss,
)


# ================================================================================
# Global fixed switches (not part of the hyperparameter search), adjust as needed
# ================================================================================
FIXED_USE_FOCAL            = True
FIXED_USE_LABEL_GCN_PRED   = True
FIXED_USE_LIP              = False
FIXED_USE_GRADIENT         = False
FIXED_USE_K_LABEL_SAMPLING = False
FIXED_USE_LABEL_AWARE      = False
FIXED_K_HOP                = 2

# =============================================================================
# Composite score weights
# =============================================================================
SPARSITY_PENALTY = 1.5   # Penalty coefficient for low explanation sparsity
GT_WEIGHT        = 1.0   # Bonus weight for ground-truth explanation F1


# =============================================================================
# Per-dataset configuration
# =============================================================================
# Each entry specifies:
#   has_gt            : whether a ground-truth explanation file exists
#   real_fast         : halve fast_epochs by default (for large real-world graphs)
#   pred_f1_floor     : minimum macro-F1 required in full-training evaluation
#   fast_pred_f1_floor: looser floor used during fast (Optuna trial) evaluation
#   ckpt_suffix       : suffix appended to the saved checkpoint filename
#   init_params       : warm-start parameters enqueued as the first Optuna trial
#   search_space      : per-parameter bounds / choices for TPE sampling
#   fast_epochs/patience : epoch budget for Optuna trials
#   full_epochs/patience : epoch budget for the final retrain
DATASET_CONFIG = {
    "blogcatalog": {
        "has_gt":             False,
        "real_fast":          True,
        "pred_f1_floor":      0.282,
        "fast_pred_f1_floor": 0.18,
        "ckpt_suffix":        "_explainable_lagcn",
        "init_params": {
            "hidden_dim": 256, "num_layers": 2, "dropout": 0.3,
            "lr": 1e-3, "weight_decay": 1e-4,
            "lambda_mask_reg": 0.2, "lambda_teacher": 0.5,
            "lambda_delete": 0.08, "lambda_masked_target": 0.8,
            "teacher_conf": 0.55,
            "base_target_ratio": 0.10,
            "label_pred_emb_dim": 64, "label_pred_gcn_layers": 1,
            "label_adj_topk": 0,
            "max_frac": 0.50, "min_edges": 2000, "cap": 20000,
            "alpha": 0.6,
        },
        "search_space": {
            "hidden_dim":            [256, 512],
            "num_layers":            (2, 3),
            "dropout":               (0.1, 0.5),
            "lr":                    (1e-4, 5e-3),
            "weight_decay":          (1e-5, 1e-3),
            "lambda_mask_reg":       (0.1, 1.0),
            "lambda_teacher":        (0.2, 1.5),
            "lambda_delete":         (0.05, 0.5),
            "lambda_masked_target":  (0.3, 1.2),
            "teacher_conf":          (0.45, 0.70),
            "base_target_ratio":     (0.03, 0.20),
            "label_pred_emb_dim":    [32, 64, 128],
            "label_pred_gcn_layers": (1, 2),
            "label_adj_topk":        [0, 10, 20],    # 0 = no truncation
            "max_frac":              (0.30, 0.70),
            "min_edges":             (500, 5000),
            "cap":                   [5000, 10000, 20000],
            "alpha":                 (0.4, 0.8),
        },
        "fast_epochs": 200, "fast_patience": 30,
        "full_epochs": 600, "full_patience": 60,
    },

    "synthetic_motif": {
        "has_gt":             True,
        "real_fast":          False,
        "pred_f1_floor":      0.682,
        "fast_pred_f1_floor": 0.55,
        "ckpt_suffix":        "_explainable_lagcn",
        "init_params": {
            "hidden_dim": 256, "num_layers": 3, "dropout": 0.2,
            "lr": 1e-3, "weight_decay": 1e-4,
            "lambda_mask_reg": 1.5, "lambda_teacher": 0.1,
            "lambda_delete": 0.7, "lambda_masked_target": 1.0,
            "teacher_conf": 0.5,
            "base_target_ratio": 0.12,
            "label_pred_emb_dim": 64, "label_pred_gcn_layers": 1,
            "label_adj_topk": 0,
            "max_frac": 0.20, "min_edges": 3, "cap": 500,
            "alpha": 0.6,
        },
        "search_space": {
            "hidden_dim":            [128, 256],
            "num_layers":            (2, 3),
            "dropout":               (0.1, 0.5),
            "lr":                    (1e-4, 5e-3),
            "weight_decay":          (1e-5, 1e-3),
            "lambda_mask_reg":       (0.1, 2.0),
            "lambda_teacher":        (0.1, 1.5),
            "lambda_delete":         (0.02, 1.0),
            "lambda_masked_target":  (0.2, 1.0),
            "teacher_conf":          (0.45, 0.70),
            "base_target_ratio":     (0.03, 0.20),
            "label_pred_emb_dim":    [32, 64, 128],
            "label_pred_gcn_layers": (1, 2),
            "label_adj_topk":        [0, 5, 10],
            "max_frac":              (0.15, 0.50),
            "min_edges":             (3, 20),
            "cap":                   [200, 500, 800],
            "alpha":                 (0.4, 0.8),
        },
        "fast_epochs": 500, "fast_patience": 40,
        "full_epochs": 800, "full_patience": 50,
    },

    "synthetic_anchor": {
        "has_gt":             True,
        "real_fast":          False,
        "pred_f1_floor":      0.796,
        "fast_pred_f1_floor": 0.65,
        "ckpt_suffix":        "_explainable_lagcn",
        "init_params": {
            "hidden_dim": 256, "num_layers": 3, "dropout": 0.3,
            "lr": 1e-3, "weight_decay": 1e-4,
            "lambda_mask_reg": 0.5, "lambda_teacher": 0.5,
            "lambda_delete": 0.05, "lambda_masked_target": 0.5,
            "teacher_conf": 0.55,
            "base_target_ratio": 0.10,
            "label_pred_emb_dim": 64, "label_pred_gcn_layers": 1,
            "label_adj_topk": 0,
            "max_frac": 0.30, "min_edges": 5, "cap": 800,
            "alpha": 0.6,
        },
        "search_space": {
            "hidden_dim":            [128, 256],
            "num_layers":            (2, 3),
            "dropout":               (0.1, 0.5),
            "lr":                    (1e-4, 5e-3),
            "weight_decay":          (1e-5, 1e-3),
            "lambda_mask_reg":       (0.1, 2.0),
            "lambda_teacher":        (0.1, 1.5),
            "lambda_delete":         (0.02, 1.0),
            "lambda_masked_target":  (0.2, 1.0),
            "teacher_conf":          (0.45, 0.70),
            "base_target_ratio":     (0.03, 0.20),
            "label_pred_emb_dim":    [32, 64, 128],
            "label_pred_gcn_layers": (1, 2),
            "label_adj_topk":        [0, 5, 10],
            "max_frac":              (0.15, 0.50),
            "min_edges":             (3, 20),
            "cap":                   [200, 500, 800],
            "alpha":                 (0.4, 0.8),
        },
        "fast_epochs": 500,  "fast_patience": 40,
        "full_epochs": 1000, "full_patience": 50,
    },

    "youtube": {
        "has_gt":             False,
        "real_fast":          True,
        "pred_f1_floor":      0.224,
        "fast_pred_f1_floor": 0.17,
        "ckpt_suffix":        "_explainable_lagcn",
        "init_params": {
            "hidden_dim": 256, "num_layers": 3, "dropout": 0.3,
            "lr": 1e-3, "weight_decay": 1e-4,
            "lambda_mask_reg": 0.2, "lambda_teacher": 0.5,
            "lambda_delete": 0.08, "lambda_masked_target": 0.8,
            "teacher_conf": 0.55,
            "base_target_ratio": 0.10,
            "label_pred_emb_dim": 64, "label_pred_gcn_layers": 1,
            "label_adj_topk": 0,
            "max_frac": 0.20, "min_edges": 10, "cap": 5000,
            "alpha": 0.6,
        },
        "search_space": {
            "hidden_dim":            [256, 512],
            "num_layers":            (2, 3),
            "dropout":               (0.1, 0.5),
            "lr":                    (1e-4, 5e-3),
            "weight_decay":          (1e-5, 1e-3),
            "lambda_mask_reg":       (0.1, 1.0),
            "lambda_teacher":        (0.2, 1.5),
            "lambda_delete":         (0.05, 0.5),
            "lambda_masked_target":  (0.3, 1.2),
            "teacher_conf":          (0.45, 0.70),
            "base_target_ratio":     (0.03, 0.20),
            "label_pred_emb_dim":    [32, 64, 128],
            "label_pred_gcn_layers": (1, 2),
            "label_adj_topk":        [0, 10, 20],
            "max_frac":              (0.10, 0.40),
            "min_edges":             (5, 50),
            "cap":                   [1000, 3000, 5000],
            "alpha":                 (0.4, 0.8),
        },
        "fast_epochs": 200, "fast_patience": 30,
        "full_epochs": 600, "full_patience": 60,
    },
}


# =============================================================================
# Composite scoring
# =============================================================================

def compute_score(
    fid_pos:  float,
    fid_neg:  float,
    sparsity: float,
    gt_f1:    Optional[float] = None,
    has_gt:   bool = False,
) -> float:
    """
    Compute a single scalar score that balances fidelity, sparsity, and
    (optionally) ground-truth explanation quality.

    score = fid_pos + fid_neg - SPARSITY_PENALTY * (1 - sparsity)
            [+ GT_WEIGHT * gt_f1  if ground-truth labels are available]

    Parameters
    ----------
    fid_pos  : Fidelity+ (higher is better).
    fid_neg  : Fidelity- (higher is better).
    sparsity : Explanation sparsity in [0, 1] (higher means sparser / more concise).
    gt_f1    : Ground-truth explanation F1 (optional).
    has_gt   : Whether a ground-truth annotation is available for this dataset.

    Returns
    -------
    float  Composite score (higher is better).
    """
    score = fid_pos + fid_neg - SPARSITY_PENALTY * (1.0 - sparsity)
    if has_gt and gt_f1 is not None:
        score += GT_WEIGHT * gt_f1
    return score


# =============================================================================
# Dataset loading (called once per dataset before the Optuna search)
# =============================================================================

def load_dataset(
    dataset: str,
    data_dir: Optional[str],
    device: torch.device,
    seed: int,
) -> Dict:
    """
    Load a graph dataset and prepare all tensors needed for training.

    Steps
    -----
    1. Load raw graph data (features, labels, adjacency) via ``load_for_training``.
    2. Register ground-truth explanation masks if ``expl_gt.npz`` is present.
    3. Perform a stratified train / val / test split (70 / 20 / 10).
    4. Compute per-label positive-class weights for the loss function.

    Parameters
    ----------
    dataset  : Dataset name (key in ``DATASET_CONFIG``).
    data_dir : Override path to the dataset folder; defaults to ``./dataset/<dataset>``.
    device   : Target torch device.
    seed     : Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        X, Y, edge_index,
        train_mask, val_mask, test_mask,
        pos_weight, N, F, L
    """
    ddir = data_dir or f"./dataset/{dataset}"
    X_t, Y_t, A_sparse, idx, meta = load_for_training(ddir, device=device)
    N, F = X_t.shape
    L = Y_t.size(1)
    edge_index = sparse_to_edge_index(A_sparse)

    # Register GT explanation masks if available (synthetic datasets only).
    gt_npz_path = os.path.join(ddir, "expl_gt.npz")
    if os.path.exists(gt_npz_path) and dataset.startswith("synthetic"):
        _register_gt(edge_index, gt_npz_path)

    # Stratified multi-label split.
    Y_np = Y_t.cpu().numpy()
    tr, va, te = multilabel_stratified_split(Y_np, train=0.7, val=0.2, seed=seed)
    train_mask = torch.zeros(N, dtype=torch.bool, device=device); train_mask[tr] = True
    val_mask   = torch.zeros(N, dtype=torch.bool, device=device); val_mask[va]   = True
    test_mask  = torch.zeros(N, dtype=torch.bool, device=device); test_mask[te]  = True

    # Per-label pos_weight = (#negative) / (#positive), capped at 50 to avoid explosion.
    Y_train = Y_t[train_mask].cpu().numpy()
    pos = Y_train.sum(axis=0) + 1e-6
    neg = train_mask.sum().item() - pos + 1e-6
    pos_weight = torch.clamp(
        torch.tensor(neg / pos, dtype=torch.float32, device=device), max=50.0
    )

    print(f"  Dataset loaded: N={N}, F={F}, L={L}, edges={edge_index.size(1)}")
    print(f"  Train={train_mask.sum()}, Val={val_mask.sum()}, Test={test_mask.sum()}")

    return {
        "X": X_t, "Y": Y_t, "edge_index": edge_index,
        "train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask,
        "pos_weight": pos_weight,
        "N": N, "F": F, "L": L,
    }


def _register_gt(edge_index: torch.Tensor, gt_npz_path: str) -> None:
    """
    Load and register ground-truth edge explanation masks into the global
    ``gt_registry``.

    The function builds a canonical undirected edge list (u < v) from either
    the pre-saved ``edge_index_undirected`` array in the .npz file or from the
    directed ``edge_index`` tensor, then maps directed edge indices to their
    undirected counterparts and stores everything in ``gt_registry``.

    Parameters
    ----------
    edge_index   : Directed edge index tensor of shape (2, E).
    gt_npz_path  : Path to the ground-truth annotation file (``expl_gt.npz``).
    """
    gt_pack = np.load(gt_npz_path, allow_pickle=True)

    # Build canonical undirected edge index: each edge stored as (min, max).
    if "edge_index_undirected" in gt_pack:
        ei_undir = gt_pack["edge_index_undirected"].astype(np.int64)
        u = np.minimum(ei_undir[:, 0], ei_undir[:, 1])
        v = np.maximum(ei_undir[:, 0], ei_undir[:, 1])
        edge_index_undir_np = np.stack([u, v], axis=1)
    else:
        # Derive undirected edges from the directed edge_index tensor.
        ei = edge_index.cpu().numpy()
        edge_set = set()
        for i in range(ei.shape[1]):
            a, b = int(ei[0, i]), int(ei[1, i])
            if a != b:
                edge_set.add((min(a, b), max(a, b)))
        edge_index_undir_np = np.array(sorted(edge_set), dtype=np.int64)

    # Map (u, v) pairs -> undirected edge ID.
    edge2id = {(int(a), int(b)): i for i, (a, b) in enumerate(edge_index_undir_np)}

    # Build a directed -> undirected index mapping array of shape (E_directed,).
    ei = edge_index.cpu().numpy()
    dir2undir = np.array(
        [
            edge2id[(min(int(ei[0, i]), int(ei[1, i])), max(int(ei[0, i]), int(ei[1, i])))]
            for i in range(ei.shape[1])
        ],
        dtype=np.int64,
    )

    gt_registry.set_all(
        gt_edge_mask=gt_pack["gt_edge_mask"].astype(bool),
        edge2id=edge2id,
        gt_edge_mask_node=gt_pack.get("gt_edge_mask_node", None),
        nodes_expl=gt_pack.get("nodes_expl", None),
        edge_index=edge_index_undir_np,
        verbose=False,
    )
    gt_registry.dir2undir = dir2undir
    gt_registry.num_directed_edges = edge_index.size(1)


# =============================================================================
# Label adjacency construction
# =============================================================================

def _build_label_adj(
    data: Dict, label_adj_topk: int, device: torch.device
) -> torch.Tensor:
    """
    Build a static Jaccard-based label co-occurrence adjacency matrix for the
    LabelGCN prediction head.

    Parameters
    ----------
    data          : Dataset dictionary (must contain ``Y`` and ``train_mask``).
    label_adj_topk: Keep only the top-k neighbours per label row (0 = keep all).
    device        : Target torch device.

    Returns
    -------
    torch.Tensor  Normalised label adjacency matrix of shape (L, L).
    """
    return build_static_label_adj(
        data["Y"], data["train_mask"],
        mode="jaccard",
        min_cooc=1,
        topk=label_adj_topk,
    ).to(device)


# =============================================================================
# Single training + evaluation run (core of the Optuna objective)
# =============================================================================

def train_and_eval(
    trial_params: Dict,
    data: Dict,
    dataset: str,
    device: torch.device,
    seed: int,
    num_eval_nodes: int,
    has_gt: bool,
    pred_f1_floor: float,
    fast_mode: bool = True,
    fast_epochs: int = 300,
    fast_patience: int = 30,
) -> Dict:
    """
    Train a SEMGNN model with the given hyperparameters and evaluate it.

    In ``fast_mode`` the model is trained for fewer epochs to keep each Optuna
    trial short.  The final retraining pass uses ``fast_mode=False`` and the
    full epoch budget from ``DATASET_CONFIG``.

    Parameters
    ----------
    trial_params    : Hyperparameter dict sampled by Optuna (or the best params
                      for the final retrain).
    data            : Dataset dictionary returned by ``load_dataset``.
    dataset         : Dataset name (key in ``DATASET_CONFIG``).
    device          : Torch device.
    seed            : Random seed.
    num_eval_nodes  : Number of test nodes used for fidelity / GT evaluation.
    has_gt          : Whether ground-truth explanations are available.
    pred_f1_floor   : Minimum acceptable macro-F1; trials below this are pruned.
    fast_mode       : If True, use ``fast_epochs`` / ``fast_patience``; otherwise
                      use the dataset's full schedule.
    fast_epochs     : Epoch budget when ``fast_mode=True``.
    fast_patience   : Early-stopping patience when ``fast_mode=True``.

    Returns
    -------
    dict containing:
        status        : "ok" | "pred_f1_too_low" | "fid_failed"
        score         : composite score (higher is better; -999 on failure)
        pred_macro_f1, pred_micro_f1,
        fid_pos, fid_neg, sparsity, gt_f1,
        threshold, model_state  (only when status == "ok")
    """
    set_seed(seed)

    X          = data["X"]
    Y          = data["Y"]
    edge_index = data["edge_index"]
    train_mask = data["train_mask"]
    val_mask   = data["val_mask"]
    test_mask  = data["test_mask"]
    pos_weight = data["pos_weight"]
    L          = data["L"]
    F_dim      = data["F"]

    # Unpack hyperparameters.
    hidden_dim            = trial_params["hidden_dim"]
    num_layers            = trial_params["num_layers"]
    dropout               = trial_params["dropout"]
    lr                    = trial_params["lr"]
    weight_decay          = trial_params["weight_decay"]
    lambda_mask_reg       = trial_params["lambda_mask_reg"]
    lambda_teacher        = trial_params["lambda_teacher"]
    lambda_delete         = trial_params["lambda_delete"]
    lambda_masked_target  = trial_params["lambda_masked_target"]
    teacher_conf          = trial_params.get("teacher_conf", 0.55)
    base_target_ratio     = trial_params["base_target_ratio"]
    label_pred_emb_dim    = trial_params["label_pred_emb_dim"]
    label_pred_gcn_layers = trial_params["label_pred_gcn_layers"]
    label_adj_topk        = trial_params["label_adj_topk"]
    max_frac              = trial_params["max_frac"]
    min_edges             = trial_params["min_edges"]
    cap                   = trial_params["cap"]
    alpha                 = trial_params["alpha"]

    cfg      = DATASET_CONFIG[dataset]
    epochs   = fast_epochs   if fast_mode else cfg["full_epochs"]
    patience = fast_patience if fast_mode else cfg["full_patience"]

    # --- Build model ---
    model = SelfExplainableGNN(
        in_dim=F_dim,
        hidden_dim=hidden_dim,
        out_dim=L,
        num_layers=num_layers,
        dropout=dropout,
        explain_hidden=64,
        temperature=1.0,
        use_gradient_guidance=FIXED_USE_GRADIENT,      # False
        grad_weight=0.4,
        use_label_aware=FIXED_USE_LABEL_AWARE,         # False
        use_label_gcn_pred=FIXED_USE_LABEL_GCN_PRED,   # True
        label_pred_emb_dim=label_pred_emb_dim,
        label_pred_gcn_layers=label_pred_gcn_layers,
    ).to(device)

    # Inject label adjacency required by the LabelGCN prediction head.
    label_adj = _build_label_adj(data, label_adj_topk, device)
    model.set_label_adj(label_adj, normalize=True, add_self_loops=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=15, verbose=False
    )

    criterion = GradientGuidedLoss(
        lambda_mask_reg=lambda_mask_reg,
        lambda_teacher=lambda_teacher,
        lambda_delete=lambda_delete,
        lambda_masked_target=lambda_masked_target,
        lambda_masked_target_floor=0.0,
        teacher_conf=teacher_conf,
        pos_weight=pos_weight,
        use_focal=FIXED_USE_FOCAL,      # True
        focal_gamma=1.0,
        sparsity_mode="adaptive_degree",
        base_target_ratio=base_target_ratio,
        min_target_ratio=0.02,
        max_target_ratio=0.30,
    )

    # --- Training loop ---
    best_val_score = -1.0
    best_state     = None
    wait           = 0

    for epoch in range(1, epochs + 1):
        # SEMGNN uses an automatic 3-stage loss schedule.
        criterion.step_auto3stage()

        model.train()
        optimizer.zero_grad(set_to_none=True)

        # k_label_sampling is disabled for this variant.
        model._use_k_label_sampling = False

        need_removed = criterion.lambda_delete > 1e-12
        result = model(
            X, edge_index,
            return_explanation=True,
            compute_gradient=FIXED_USE_GRADIENT,    # False
            compute_removed=need_removed,
            targets=Y,
            train_mask=train_mask,
            pos_weight=pos_weight,
        )

        loss, _ = compute_loss(
            result,
            Y,
            train_mask,
            criterion,
            edge_index,
            pos_weight,
            FIXED_USE_FOCAL,
            1.0,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Validate every 5 epochs using area under the precision-recall curve.
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_out   = model(X, edge_index, return_explanation=False,
                                  compute_gradient=False)
                val_probs = torch.sigmoid(val_out["logits"][val_mask]).cpu().numpy()
                val_tgts  = Y[val_mask].cpu().numpy()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prec, rec, _ = precision_recall_curve(val_tgts.ravel(), val_probs.ravel())
                val_score = auc(rec, prec)

            scheduler.step(val_score)

            if val_score > best_val_score + 1e-4:
                best_val_score = val_score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    model.use_gradient_guidance = False  # Not needed at inference time.

    # --- Find optimal classification threshold on the validation set ---
    with torch.no_grad():
        all_out   = model(X, edge_index, return_explanation=False, compute_gradient=False)
        val_probs = torch.sigmoid(all_out["logits"][val_mask])
        val_tgts  = Y[val_mask]

    tf = OptimalThresholdFinder(
        search_range=(0.1, 0.9), search_step=0.05, verbose=False
    )
    tf_result = tf.find_optimal_threshold(
        val_probs, val_tgts, strategy="global", metric="micro_f1"
    )
    threshold = tf_result.best_threshold

    # --- Prediction quality gate ---
    with torch.no_grad():
        test_probs = torch.sigmoid(all_out["logits"][test_mask]).cpu().numpy()
        test_tgts  = Y[test_mask].cpu().numpy()

    test_preds    = (test_probs >= threshold).astype(int)
    pred_macro_f1 = float(f1_score(test_tgts, test_preds, average="macro", zero_division=0))
    pred_micro_f1 = float(f1_score(test_tgts.ravel(), test_preds.ravel(), zero_division=0))

    # In fast mode we use a looser micro-F1 floor to avoid over-pruning.
    effective_floor = (
        cfg.get("fast_pred_f1_floor", pred_f1_floor) if fast_mode else pred_f1_floor
    )
    check_f1 = pred_micro_f1 if fast_mode else pred_macro_f1
    if check_f1 < effective_floor:
        print(
            f"    x {'micro' if fast_mode else 'macro'}_f1={check_f1:.4f} "
            f"< floor={effective_floor:.4f} ({'fast' if fast_mode else 'final'}), pruned"
        )
        return {
            "status":        "pred_f1_too_low",
            "pred_macro_f1": pred_macro_f1,
            "pred_micro_f1": pred_micro_f1,
            "score":         -999.0,
        }

    # --- Prepare evaluation node set ---
    test_nodes = test_mask.nonzero(as_tuple=True)[0].cpu().numpy().astype(int).tolist()
    rng = np.random.RandomState(seed)
    rng.shuffle(test_nodes)

    # Prioritise GT-annotated nodes when available.
    eval_nodes: List[int] = []
    if has_gt and gt_registry.is_initialized and gt_registry.nodes_expl is not None:
        gt_nodes   = {int(n) for n in gt_registry.nodes_expl.tolist()}
        eval_nodes = [n for n in test_nodes if n in gt_nodes][:num_eval_nodes]
    if len(eval_nodes) < num_eval_nodes:
        used = set(eval_nodes)
        eval_nodes.extend(
            [n for n in test_nodes if n not in used][: num_eval_nodes - len(eval_nodes)]
        )

    # --- Pre-compute per-label edge masks for all active labels ---
    with torch.no_grad():
        probs_all = torch.sigmoid(all_out["logits"]).detach()

    active_labels: set = set()
    for nidx in eval_nodes:
        preds_n = np.where(probs_all[nidx].cpu().numpy() >= threshold)[0].tolist()
        if not preds_n:
            preds_n = [int(torch.argmax(probs_all[nidx]))]
        active_labels.update(preds_n)
    active_labels_sorted = sorted(active_labels)

    label_masks = model.compute_full_graph_edge_masks_multilabel(
        X, edge_index, active_labels_sorted,
        Y=Y, train_mask=train_mask, pos_weight=pos_weight,
    )

    # --- Fidelity evaluation ---
    fid_eval = FidelityEvaluator(min_edges=min_edges, max_frac=max_frac, cap=cap)
    fid_res  = fid_eval.evaluate_batch(
        model, X, edge_index, node_indices=eval_nodes,
        k_hop=FIXED_K_HOP,
        threshold=threshold, verbose=False,
        label_masks=label_masks, Y=Y,
        train_mask=train_mask, pos_weight=pos_weight,
        use_node_local_necessity=True, alpha=alpha,
    )

    if fid_res.get("status") != "ok":
        return {"status": "fid_failed", "score": -999.0}

    fid_pos  = fid_res["fidelity_pos_mean"]
    fid_neg  = fid_res["fidelity_neg_mean"]
    sparsity = fid_res["sparsity_mean"]

    # --- Ground-truth explanation evaluation (synthetic datasets only) ---
    gt_f1 = None
    if has_gt and gt_registry.is_initialized and gt_registry.nodes_expl is not None:
        gt_eval = GTExplanationEvaluator(min_edges=min_edges, max_frac=max_frac, cap=cap)
        gt_res  = gt_eval.evaluate_batch(
            model, X, edge_index, Y, node_indices=eval_nodes,
            k_hop=FIXED_K_HOP,
            verbose=False,
            train_mask=train_mask, pos_weight=pos_weight,
            label_masks=label_masks, threshold=threshold,
        )
        if gt_res.get("status") == "ok":
            gt_f1 = gt_res.get("macro_f1_mean", 0.0)

    score = compute_score(fid_pos, fid_neg, sparsity, gt_f1, has_gt)

    return {
        "status":        "ok",
        "score":         score,
        "pred_macro_f1": pred_macro_f1,
        "pred_micro_f1": pred_micro_f1,
        "fid_pos":       fid_pos,
        "fid_neg":       fid_neg,
        "sparsity":      sparsity,
        "gt_f1":         gt_f1,
        "threshold":     float(threshold),
        "model_state":   {k: v.cpu() for k, v in model.state_dict().items()},
    }


# =============================================================================
# Optuna objective factory
# =============================================================================

def make_objective(
    data, dataset, device, seed, num_eval_nodes, has_gt, pred_f1_floor,
    fast_epochs_override=None, fast_patience_override=None,
):
    """
    Create and return an Optuna objective function for the given dataset.

    The returned callable samples hyperparameters from the dataset's
    ``search_space``, runs ``train_and_eval`` in fast mode, and returns the
    composite score.  OOM errors are caught and the trial is assigned -999.

    Parameters
    ----------
    data, dataset, device, seed, num_eval_nodes, has_gt, pred_f1_floor :
        Passed through to ``train_and_eval``.
    fast_epochs_override  : Override ``fast_epochs`` from ``DATASET_CONFIG``.
    fast_patience_override: Override ``fast_patience`` from ``DATASET_CONFIG``.

    Returns
    -------
    Callable[[optuna.Trial], float]
    """
    cfg = DATASET_CONFIG[dataset]
    ss  = cfg["search_space"]
    fe  = fast_epochs_override   if fast_epochs_override  is not None else cfg["fast_epochs"]
    fp  = fast_patience_override if fast_patience_override is not None else cfg["fast_patience"]

    def objective(trial: optuna.Trial) -> float:
        # Sample all hyperparameters from the defined search space.
        params = {
            "hidden_dim":            trial.suggest_categorical("hidden_dim",            ss["hidden_dim"]),
            "num_layers":            trial.suggest_int(        "num_layers",            *ss["num_layers"]),
            "dropout":               trial.suggest_float(      "dropout",               *ss["dropout"],               step=0.1),
            "lr":                    trial.suggest_float(      "lr",                    *ss["lr"],                    log=True),
            "weight_decay":          trial.suggest_float(      "weight_decay",          *ss["weight_decay"],          log=True),
            "lambda_mask_reg":       trial.suggest_float(      "lambda_mask_reg",       *ss["lambda_mask_reg"]),
            "lambda_teacher":        trial.suggest_float(      "lambda_teacher",        *ss["lambda_teacher"]),
            "lambda_delete":         trial.suggest_float(      "lambda_delete",         *ss["lambda_delete"]),
            "lambda_masked_target":  trial.suggest_float(      "lambda_masked_target",  *ss["lambda_masked_target"]),
            "teacher_conf":          trial.suggest_float(      "teacher_conf",          *ss["teacher_conf"],          step=0.05),
            "base_target_ratio":     trial.suggest_float(      "base_target_ratio",     *ss["base_target_ratio"]),
            "label_pred_emb_dim":    trial.suggest_categorical("label_pred_emb_dim",    ss["label_pred_emb_dim"]),
            "label_pred_gcn_layers": trial.suggest_int(        "label_pred_gcn_layers", *ss["label_pred_gcn_layers"]),
            "label_adj_topk":        trial.suggest_categorical("label_adj_topk",        ss["label_adj_topk"]),
            "max_frac":              trial.suggest_float(      "max_frac",              *ss["max_frac"],              step=0.05),
            "min_edges":             trial.suggest_int(        "min_edges",             *ss["min_edges"]),
            "cap":                   trial.suggest_categorical("cap",                   ss["cap"]),
            "alpha":                 trial.suggest_float(      "alpha",                 *ss["alpha"],                 step=0.1),
        }

        try:
            result = train_and_eval(
                trial_params=params, data=data, dataset=dataset,
                device=device, seed=seed,
                num_eval_nodes=num_eval_nodes,
                has_gt=has_gt, pred_f1_floor=pred_f1_floor,
                fast_mode=True,
                fast_epochs=fe, fast_patience=fp,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  Trial {trial.number:3d} | OOM - skipping this parameter set")
            return -999.0

        score  = result.get("score", -999.0)
        status = result.get("status", "unknown")

        print(
            f"  Trial {trial.number:3d} | {status:15s} | "
            f"score={score:7.4f} | "
            f"Fid+={result.get('fid_pos', float('nan')):.4f} "
            f"Fid-={result.get('fid_neg', float('nan')):.4f} "
            f"Sparsity={result.get('sparsity', float('nan')):.4f} "
            + (f"GT_F1={result.get('gt_f1', 0.0):.4f} " if has_gt else "")
            + f"| PredMF1={result.get('pred_macro_f1', float('nan')):.4f}"
        )

        # Store auxiliary metrics as Optuna user attributes for later analysis.
        for k in ["fid_pos", "fid_neg", "sparsity", "gt_f1", "pred_macro_f1", "pred_micro_f1"]:
            v = result.get(k)
            if v is not None:
                trial.set_user_attr(k, float(v))

        return score

    return objective


# =============================================================================
# Final retraining with the best hyperparameters
# =============================================================================

def final_train(
    best_params: Dict,
    data: Dict,
    dataset: str,
    device: torch.device,
    seed: int,
    num_eval_nodes: int,
    has_gt: bool,
    pred_f1_floor: float,
    save_dir: str,
) -> Dict:
    """
    Retrain the model using the best parameters found by Optuna with the full
    epoch budget, evaluate, and save the checkpoint.

    The saved ``.pt`` file contains:
        - ``model_state``  : state dict of the trained model
        - ``best_params``  : hyperparameters used for this run
        - ``fixed_config`` : the global fixed switches
        - ``eval_results`` : fidelity / GT / prediction metrics

    Parameters
    ----------
    best_params    : Best hyperparameter dict from ``study.best_trial.params``.
    data           : Dataset dictionary from ``load_dataset``.
    dataset        : Dataset name.
    device         : Torch device.
    seed           : Random seed.
    num_eval_nodes : Number of nodes used for explanation evaluation.
    has_gt         : Whether GT annotations exist.
    pred_f1_floor  : Minimum acceptable macro-F1 for the final model.
    save_dir       : Directory where the checkpoint will be written.

    Returns
    -------
    dict  Result dict from ``train_and_eval`` (status, score, metrics, …).
    """
    print(f"\n{'=' * 60}")
    print("Final training (full epochs) with best parameters")
    print(f"{'=' * 60}")
    print(json.dumps(best_params, indent=2))

    result = train_and_eval(
        trial_params=best_params, data=data, dataset=dataset,
        device=device, seed=seed,
        num_eval_nodes=num_eval_nodes,
        has_gt=has_gt, pred_f1_floor=pred_f1_floor,
        fast_mode=False,
        fast_epochs=0, fast_patience=0,
    )

    if result["status"] != "ok":
        print(f"Warning: final training failed with status '{result['status']}'")
        return result

    print("\nFinal results:")
    print(f"  Composite score : {result['score']:.4f}")
    print(f"  Fidelity+       : {result['fid_pos']:.4f}")
    print(f"  Fidelity-       : {result['fid_neg']:.4f}")
    print(f"  Sparsity        : {result['sparsity']:.4f}")
    if result.get("gt_f1") is not None:
        print(f"  GT F1           : {result['gt_f1']:.4f}")
    print(f"  Pred Macro-F1   : {result['pred_macro_f1']:.4f}")
    print(f"  Threshold       : {result['threshold']:.3f}")

    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"{dataset}_explainable_lagcn.pt")
    torch.save(
        {
            "model_state":  result["model_state"],
            "best_params":  best_params,
            "fixed_config": {
                "use_focal":            FIXED_USE_FOCAL,
                "use_label_gcn_pred":   FIXED_USE_LABEL_GCN_PRED,
                "use_gradient":         FIXED_USE_GRADIENT,
                "use_k_label_sampling": FIXED_USE_K_LABEL_SAMPLING,
                "use_label_aware":      FIXED_USE_LABEL_AWARE,
                "k_hop":                FIXED_K_HOP,
            },
            "eval_results": {
                k: result[k]
                for k in ["score", "fid_pos", "fid_neg", "sparsity",
                           "gt_f1", "pred_macro_f1", "threshold"]
                if k in result
            },
        },
        ckpt_path,
    )
    print(f"\nFinal model saved: {ckpt_path}")
    return result


# =============================================================================
# Full pipeline for a single dataset
# =============================================================================

def tune_dataset(dataset: str, args: argparse.Namespace, device: torch.device):
    """
    Run the complete hyperparameter search + final training pipeline for one
    dataset.

    Steps
    -----
    1. Validate dataset name against ``DATASET_CONFIG``.
    2. Load dataset once (shared across all trials).
    3. Optionally halve ``fast_epochs`` for large real-world graphs.
    4. Create (or resume) an Optuna study backed by SQLite for crash recovery.
    5. Enqueue the warm-start ``init_params`` as the first trial.
    6. Run ``n_trials`` trials; after each trial save an incremental JSON summary.
    7. Retrain the best parameter set with the full epoch budget.
    8. Persist the final result as ``final_result.json``.

    Parameters
    ----------
    dataset : Dataset name (key in ``DATASET_CONFIG``).
    args    : Parsed CLI arguments.
    device  : Torch device.

    Returns
    -------
    dict | None  Final result dict, or None if no trial completed successfully.
    """
    cfg = DATASET_CONFIG.get(dataset)
    if cfg is None:
        print(f"Warning: unknown dataset '{dataset}', skipping.")
        return

    has_gt        = cfg["has_gt"]
    pred_f1_floor = cfg["pred_f1_floor"]

    log_dir = os.path.join(args.log_dir, dataset)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'#' * 60}")
    print(f"  Dataset: {dataset.upper()}")
    print(f"  has_gt={has_gt} | pred_f1_floor={pred_f1_floor}")
    print(f"  n_trials={args.n_trials} | eval_nodes={args.num_eval_nodes}")
    print(f"  Fixed config: focal=True  label_gcn_pred=True  lip=False")
    print(f"                gradient=False  k_label_sampling=False  label_aware=False  k_hop=2")
    print(f"{'#' * 60}")

    data = load_dataset(dataset, args.data_dir, device, args.seed)

    # Optionally halve fast_epochs for large real-world graphs to speed up search.
    use_real_fast = cfg.get("real_fast", False) or args.real_fast
    if use_real_fast:
        cfg = dict(cfg)  # Shallow copy to avoid mutating the global config.
        cfg["fast_epochs"]   = max(80,  cfg["fast_epochs"]  // 2)
        cfg["fast_patience"] = max(15,  cfg["fast_patience"] // 2)
        src = "(dataset default)" if cfg.get("real_fast", False) else "(--real_fast flag)"
        print(
            f"  [real_fast {src}] fast_epochs={cfg['fast_epochs']}  "
            f"fast_patience={cfg['fast_patience']}"
        )

    # --- SQLite-backed Optuna study (supports crash recovery via --resume) ---
    db_path    = os.path.join(log_dir, f"optuna_{dataset}.db")
    storage    = f"sqlite:///{db_path}"
    study_name = f"semgnn_la_{dataset}"

    already_exists = os.path.exists(db_path)
    if already_exists and args.resume:
        print(f"  [resume] Existing study found, continuing from checkpoint: {db_path}")
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
            sampler=optuna.samplers.TPESampler(seed=args.seed),
        )
        done = len([t for t in study.trials if t.value is not None])
        print(f"  {done} trials already completed; searching up to {args.n_trials} total.")
    else:
        if already_exists and not args.resume:
            # Non-resume mode with existing DB: delete and restart to avoid contamination.
            os.remove(db_path)
            print("  [new] Removed old DB; starting fresh search.")
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage,
            sampler=optuna.samplers.TPESampler(seed=args.seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0),
            load_if_exists=False,
        )
        # Warm-start: enqueue domain-knowledge initialisation parameters as trial 0.
        if cfg["init_params"]:
            study.enqueue_trial(cfg["init_params"])
            print("  ★ Trial 0 will use init_params as a warm start.")

    fixed_config = {
        "use_focal":            FIXED_USE_FOCAL,
        "use_label_gcn_pred":   FIXED_USE_LABEL_GCN_PRED,
        "use_gradient":         FIXED_USE_GRADIENT,
        "use_k_label_sampling": FIXED_USE_K_LABEL_SAMPLING,
        "use_label_aware":      FIXED_USE_LABEL_AWARE,
        "k_hop":                FIXED_K_HOP,
    }
    summary_path = os.path.join(log_dir, "search_summary.json")

    # Wrap start time in a list so the closure can reference it without
    # needing the `nonlocal` keyword (compatible with Python 3.7+).
    trial_start_time = [time.time()]

    def save_after_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """
        Callback: atomically write an incremental JSON summary after every trial.

        Uses a write-then-rename pattern so the file is never left in a
        partially-written state if the process is interrupted mid-write.
        """
        completed = [t for t in study.trials if t.value is not None]
        if not completed:
            return

        best        = study.best_trial
        elapsed_now = time.time() - trial_start_time[0]

        summary = {
            "dataset":           dataset,
            "n_trials_done":     len(completed),
            "n_trials_target":   args.n_trials,
            "elapsed_min":       elapsed_now / 60,
            "best_score":        best.value,
            "best_trial_number": best.number,
            "best_params":       best.params,
            "best_metrics":      best.user_attrs,
            "fixed_config":      fixed_config,
            "all_trials": [
                {"number": t.number, "score": t.value,
                 "params": t.params, "attrs": t.user_attrs}
                for t in completed
            ],
        }

        tmp_path = summary_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(summary, f, indent=2)
        os.replace(tmp_path, summary_path)  # Atomic replace.

        print(
            f"  [log] Trial #{trial.number} done → "
            f"current best #{best.number} score={best.value:.4f}  "
            f"({len(completed)}/{args.n_trials})  saved: {summary_path}"
        )

    objective = make_objective(
        data=data, dataset=dataset, device=device,
        seed=args.seed, num_eval_nodes=args.num_eval_nodes,
        has_gt=has_gt, pred_f1_floor=pred_f1_floor,
        fast_epochs_override=cfg["fast_epochs"],
        fast_patience_override=cfg["fast_patience"],
    )

    done_so_far = len([t for t in study.trials if t.value is not None])
    remaining   = max(0, args.n_trials - done_so_far)
    print(f"  Running {remaining} trials (target total: {args.n_trials})")

    t0 = time.time()
    trial_start_time[0] = t0
    study.optimize(
        objective,
        n_trials=remaining,
        callbacks=[save_after_trial],
        show_progress_bar=False,
    )
    elapsed = time.time() - t0

    print(f"\nSearch complete. Elapsed: {elapsed / 60:.1f} min")
    completed_trials = [t for t in study.trials if t.value is not None]
    if not completed_trials:
        print("Warning: no trials completed successfully; skipping final training.")
        return None

    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best score: {study.best_value:.4f}")

    best_params = study.best_trial.params
    best_attrs  = study.best_trial.user_attrs

    print("\nBest parameters:")
    print(json.dumps(best_params, indent=2))
    print("\nBest metrics:")
    print(json.dumps(best_attrs, indent=2))

    # Retrain with the best parameters at full epoch budget.
    final_result = final_train(
        best_params=best_params, data=data, dataset=dataset,
        device=device, seed=args.seed,
        num_eval_nodes=args.num_eval_nodes,
        has_gt=has_gt, pred_f1_floor=pred_f1_floor,
        save_dir=os.path.join(log_dir, "ckpts"),
    )

    final_summary = {
        "dataset":      dataset,
        "best_params":  best_params,
        "final_result": {k: v for k, v in final_result.items() if k != "model_state"},
    }
    final_path = os.path.join(log_dir, "final_result.json")
    with open(final_path, "w") as f:
        json.dump(final_summary, f, indent=2)
    print(f"Final result saved: {final_path}")

    return final_result


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    """
    Parse CLI arguments, run the hyperparameter search for each requested
    dataset sequentially, and print / save a cross-dataset summary table.
    """
    parser = argparse.ArgumentParser(description="SEMGNN automated hyperparameter search")

    parser.add_argument(
        "--datasets", nargs="+",
        default=["blogcatalog", "youtube", "synthetic_motif", "synthetic_anchor"],
        choices=["blogcatalog", "synthetic_motif", "synthetic_anchor", "youtube"],
        help="List of datasets to tune.",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Dataset root directory (default: ./dataset/<dataset>).",
    )
    parser.add_argument(
        "--log_dir", type=str, default="tune_logs",
        help="Directory where logs and checkpoints are saved.",
    )
    parser.add_argument(
        "--n_trials", type=int, default=40,
        help="Number of Optuna trials per dataset (recommended: 40-60).",
    )
    parser.add_argument(
        "--num_eval_nodes", type=int, default=30,
        help="Number of test nodes used for explanation evaluation per trial (recommended: 20-50).",
    )
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--resume", action="store_true", default=False,
        help="Resume an interrupted search from the SQLite checkpoint.",
    )
    parser.add_argument(
        "--real_fast", action="store_true", default=False,
        help="Speed-up mode for large graphs: halve fast_epochs (slightly lower accuracy).",
    )

    args   = parser.parse_args()
    device = torch.device(args.device)

    print(f"\n{'=' * 60}")
    print("SEMGNN Automated Hyperparameter Search")
    print(f"{'=' * 60}")
    print(f"Datasets     : {args.datasets}")
    print(f"Trials/dataset: {args.n_trials}")
    print(f"Eval nodes   : {args.num_eval_nodes}")
    print(f"Device       : {args.device}")
    print(f"Log dir      : {args.log_dir}")
    print(f"Resume       : {args.resume}")
    print(f"Real fast    : {args.real_fast}")
    print(f"{'=' * 60}")
    print("Fixed config:")
    print("  use_focal=True | use_label_gcn_pred=True")
    print("  use_gradient=False")
    print("  use_k_label_sampling=False | use_label_aware=False")
    print("  k_hop=2")
    print(f"{'=' * 60}\n")

    all_results: Dict[str, Optional[Dict]] = {}
    total_start = time.time()

    for dataset in args.datasets:
        all_results[dataset] = tune_dataset(dataset=dataset, args=args, device=device)

    total_elapsed = time.time() - total_start

    # Print cross-dataset summary table.
    print(f"\n{'=' * 60}")
    print(f"All done! Total elapsed: {total_elapsed / 60:.1f} min")
    print(f"{'=' * 60}")
    print(
        f"\n{'Dataset':<25} {'Score':>8} {'Fid+':>8} {'Fid-':>8} "
        f"{'Sparsity':>10} {'GT_F1':>8} {'PredMF1':>10}"
    )
    print("-" * 80)
    for ds, res in all_results.items():
        if res and res.get("status") == "ok":
            print(
                f"{ds:<25} "
                f"{res.get('score', 0):.4f}   "
                f"{res.get('fid_pos', 0):.4f}   "
                f"{res.get('fid_neg', 0):.4f}   "
                f"{res.get('sparsity', 0):.4f}     "
                f"{res.get('gt_f1') or 0:.4f}   "
                f"{res.get('pred_macro_f1', 0):.4f}"
            )
        else:
            print(f"{ds:<25} FAILED")

    # Save cross-dataset summary to JSON.
    os.makedirs(args.log_dir, exist_ok=True)
    summary_all_path = os.path.join(args.log_dir, "all_datasets_summary.json")
    with open(summary_all_path, "w") as f:
        json.dump(
            {
                ds: {k: v for k, v in (res or {}).items() if k != "model_state"}
                for ds, res in all_results.items()
            },
            f, indent=2,
        )
    print(f"\nCross-dataset summary saved: {summary_all_path}")


if __name__ == "__main__":
    main()