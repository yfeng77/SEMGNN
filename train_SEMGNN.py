"""
train_SEMGNN.py

Training script for the Self-Explainable Multi-label GNN (SEMGNN).

Pipeline overview:
  1. Load dataset and (optionally) ground-truth explanation masks.
  2. Stratified multi-label train / val / test split.
  3. Build model, optimizer, loss criterion, and label adjacency.
  4. Training loop with automatic 3-stage loss scheduling and early stopping.
  5. Optimal threshold search on the validation set.
  6. Full evaluation: prediction metrics, fidelity, and GT explanation quality.
  7. Save model checkpoint and unified evaluation JSON.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    accuracy_score,
    precision_recall_curve,
    auc,
)

# Ensure the project root is on sys.path so local modules are importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from data_process.data_process import load_for_training
from SEMGNN import SelfExplainableGNN, GradientGuidedLoss
from utils.utils import (
    set_seed,
    sparse_to_edge_index,
    multilabel_stratified_split,
)
from eval.evaluate import (
    PredictionEvaluator,
    FidelityEvaluator,
    GTExplanationEvaluator,
)
from utils.gt_registry import gt_registry
from utils.optimal_threshold import OptimalThresholdFinder


# ============================================================
# Utility / helper functions
# ============================================================


def log_la_grad_stats(model, step: int, every: int = 100):
    """
    Print gradient magnitude statistics for the LabelAwareEdgeScorer parameters.

    This is a lightweight diagnostic to verify that the label-aware scorer is
    actually receiving gradient signal during training.  The function is a no-op
    when (a) step is not a multiple of `every`, or (b) the model has no
    label_edge_scorer attribute.

    Args:
        model: The SelfExplainableGNN instance.
        step:  Current training step (epoch number).
        every: Print interval — only logs when step % every == 0.
    """
    if step % every != 0:
        return
    scorer = getattr(model, "label_edge_scorer", None)
    if scorer is None:
        return

    def _gmean(p):
        """Return mean absolute gradient for parameter p, or NaN if unavailable."""
        if (p is None) or (p.grad is None):
            return float("nan")
        return float(p.grad.detach().abs().mean().item())

    msg = (
        f"[LA-grad step={step}] "
        f"label_emb={_gmean(scorer.label_embedding.weight):.3e} | "
        f"edge_to_rank={_gmean(scorer.edge_to_rank.weight):.3e} | "
        f"label_to_rank={_gmean(scorer.label_to_rank[0].weight):.3e}"
    )
    # Append per-layer GCN gradient stats if label correlation GCN is present
    if hasattr(scorer, "label_gcn"):
        gcn_vals = []
        for i, layer in enumerate(scorer.label_gcn):
            gcn_vals.append(f"gcn{i}={_gmean(layer.weight):.3e}")
        msg += " | " + " ".join(gcn_vals)

    print(msg)


def build_static_label_adj(
    Y: torch.Tensor,
    train_mask: torch.Tensor,
    mode: str = "jaccard",  # Supported modes: "jaccard" or "npmi"
    eps: float = 1e-8,
    min_cooc: int = 1,
    topk: int = 0,          # 0 = keep all; >0 = keep top-k entries per row
) -> torch.Tensor:
    """
    Build a static label co-occurrence adjacency matrix from training labels.

    Two similarity measures are supported:

    - **Jaccard**: A[i, j] = |i ∩ j| / |i ∪ j|.  Values are in [0, 1].
    - **NPMI**:    Normalised Pointwise Mutual Information.  Negative values are
                  clamped to 0 so the result can be used directly as an adjacency.

    The resulting matrix is symmetrised and the diagonal is zeroed out (no
    self-loops).  An optional top-k truncation can be applied per row to keep the
    graph sparse when L is large.

    Args:
        Y:          Full label matrix [N, L] (int or float, 0/1 entries).
        train_mask: Boolean mask [N] selecting training nodes.
        mode:       Similarity measure — "jaccard" or "npmi".
        eps:        Small constant for numerical stability.
        min_cooc:   Minimum co-occurrence count; pairs below this are zeroed out.
        topk:       If > 0, keep only the top-k strongest connections per label.

    Returns:
        Symmetric adjacency matrix [L, L] with zero diagonal.
    """
    Ytr = Y[train_mask].float()         # (n_train, L)
    n = float(Ytr.size(0))
    counts = (Ytr.t() @ Ytr)            # (L, L) co-occurrence count matrix
    counts = counts.clamp(min=0)

    # Zero out noisy low-frequency co-occurrences
    if min_cooc > 1:
        counts = torch.where(counts >= min_cooc, counts, torch.zeros_like(counts))

    freq = torch.diag(counts)           # Per-label occurrence counts (L,)

    if mode == "jaccard":
        union = freq.view(-1, 1) + freq.view(1, -1) - counts
        adj = counts / (union + eps)
    elif mode == "npmi":
        p_i = freq / (n + eps)
        p_ij = counts / (n + eps)
        num = torch.log((p_ij + eps) / (p_i.view(-1, 1) * p_i.view(1, -1) + eps))
        den = -torch.log(p_ij + eps)
        adj = num / (den + eps)
        adj = torch.clamp(adj, min=0.0)  # Retain only positive correlations
    else:
        raise ValueError(f"Unknown label_adj mode: {mode}")

    adj.fill_diagonal_(0.0)
    adj = 0.5 * (adj + adj.t())         # Ensure exact symmetry

    # Optional row-wise top-k sparsification to avoid O(L^2) density
    if topk and topk > 0:
        L = adj.size(0)
        k = min(topk, L - 1)
        new_adj = torch.zeros_like(adj)
        vals, idx = torch.topk(adj, k=k, dim=1)
        row = torch.arange(L, device=adj.device).view(-1, 1).expand_as(idx)
        new_adj[row, idx] = vals
        # Re-symmetrise after top-k sparsification
        adj = 0.5 * (new_adj + new_adj.t())
        adj.fill_diagonal_(0.0)

    return adj


def compute_loss(
    result: Dict,
    Y: torch.Tensor,
    train_mask: torch.Tensor,
    criterion: GradientGuidedLoss,
    edge_index: torch.Tensor,
    pos_weight: torch.Tensor,
    use_focal: bool,
    focal_gamma: float,
) -> tuple:
    """
    Compute the combined training loss from model outputs and the criterion.

    The total loss is composed of:
      - ``pred_full_loss``:  Standard BCE (or focal) on full-graph predictions vs GT.
      - ``loss_teacher``:    Teacher distillation — masked predictions track full logits.
      - ``loss_delete``:     Delete loss — predictions from the complement mask should drop.
      - ``loss_masked_target``: Warm-up GT anchor on masked predictions (stage 0 only).
      - ``loss_mask_reg``:   Adaptive sparsity regularisation on the edge mask.

    Args:
        result:      Output dict from model.forward() with return_explanation=True.
        Y:           Binary label targets [N, L].
        train_mask:  Boolean node mask [N].
        criterion:   GradientGuidedLoss instance (holds current lambda values).
        edge_index:  Edge indices [2, E].
        pos_weight:  Per-label positive class weights [L].
        use_focal:   Whether to use focal loss for the full-graph prediction term.
        focal_gamma: Focal loss gamma parameter.

    Returns:
        Tuple (total_loss, loss_dict) where total_loss is the scalar to
        backpropagate and loss_dict is a flat dict of detached scalars for logging.
    """
    # --- 1. Compute all explanation-related loss terms via the criterion ---
    losses = criterion(
        logits_full=result["logits"],
        logits_masked=result["logits_masked"],
        logits_removed=result.get("logits_removed", None),
        edge_mask=result["edge_mask"],
        targets=Y,
        train_mask=train_mask,
        edge_index=edge_index,
        grad_scores=result.get("grad_scores"),
        edge_mask_soft=result.get("edge_mask_soft"),
    )

    # --- 2. Full-graph prediction loss (standard BCE or focal) ---
    if use_focal:
        bce = F.binary_cross_entropy_with_logits(
            result["logits"][train_mask],
            Y[train_mask].float(),
            reduction='none',
            pos_weight=pos_weight,
        )
        probs = torch.sigmoid(result["logits"][train_mask])
        pt = torch.where(Y[train_mask] == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** focal_gamma
        pred_full_loss = (focal_weight * bce).mean()
    else:
        pred_full_loss = F.binary_cross_entropy_with_logits(
            result["logits"][train_mask],
            Y[train_mask].float(),
            pos_weight=pos_weight,
        )

    # --- 3. Combine all terms into the total loss ---
    total_loss = (
        pred_full_loss
        + criterion.lambda_teacher       * losses["loss_teacher"]
        + criterion.lambda_delete        * losses["loss_delete"]
        + criterion.lambda_masked_target * losses["loss_masked_target"]
        + criterion.lambda_mask_reg      * losses["loss_mask_reg"]
    )

    # --- Build a flat loss_dict for logging (all values detached) ---
    loss_dict = {}

    # Copy everything from the criterion output
    for k, v in losses.items():
        loss_dict[k] = v

    # Add prediction loss and total loss with convenient aliases
    loss_dict.update({
        "loss_total":            total_loss.detach(),
        "loss_pred":             pred_full_loss.detach(),
        "total":                 total_loss.detach(),
        "pred_full_weighted":    pred_full_loss.detach(),
        "teacher":               losses["loss_teacher"].detach(),
        "delete":                losses["loss_delete"].detach(),
        "masked_target":         losses["loss_masked_target"].detach(),
        "mask_reg":              losses["loss_mask_reg"].detach(),
    })

    return total_loss, loss_dict


# ============================================================
# Main training function
# ============================================================


def train_explainable(args):
    """
    Full training and evaluation pipeline for SEMGNN.

    Executes the following steps sequentially:
      1. Data loading and (optional) GT explanation mask loading.
      2. Train / val / test split.
      3. Model and optimizer initialisation.
      4. Label adjacency construction (when label-aware components are enabled).
      5. Training loop with 3-stage scheduling and early stopping.
      6. Optimal threshold search on the validation set.
      7. Test set evaluation (prediction + fidelity + GT quality).
      8. Checkpoint and JSON result saving.

    Args:
        args: Parsed argparse.Namespace object (see main() for full argument list).

    Returns:
        The trained SelfExplainableGNN model (in eval mode with best weights loaded).
    """
    gt_registry.reset()
    set_seed(args.seed)
    device = torch.device(args.device)

    # ============================================================
    # 1. Load dataset
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Loading dataset: {args.dataset}")
    print(f"{'='*60}")

    data_dir = args.data_dir or f"./dataset/{args.dataset}"
    X_t, Y_t, A_sparse, idx, meta = load_for_training(data_dir, device=device)

    N, F = X_t.shape
    L = Y_t.size(1)
    edge_index = sparse_to_edge_index(A_sparse)

    print(f"Nodes: {N}, Features: {F}, Labels: {L}")
    print(f"Edges: {edge_index.size(1)}")

    # --- Load ground-truth explanation masks (synthetic datasets only) ---
    gt_npz_path = os.path.join(data_dir, "expl_gt.npz")
    if os.path.exists(gt_npz_path) and args.dataset.startswith("synthetic"):
        print(f"\nLoading GT from {gt_npz_path}...")
        gt_pack = np.load(gt_npz_path, allow_pickle=True)

        def build_edge_mappings(edge_index_directed: torch.Tensor, gt_pack):
            """
            Build directed-to-undirected edge index mapping.

            If the GT pack contains a pre-built undirected edge index, it is used
            directly.  Otherwise, the undirected edges are derived from the directed
            graph by canonicalising each edge as (min(u,v), max(u,v)).

            Returns:
                Tuple of:
                  - edge_index_undir_np: [E_undir, 2] undirected edge array.
                  - edge2id:             Dict mapping (u, v) -> undirected edge index.
                  - dir2undir:           [E_dir] array mapping directed edges to
                                         undirected edge indices.
            """
            if "edge_index_undirected" in gt_pack:
                ei_undir = gt_pack["edge_index_undirected"].astype(np.int64)
                u = np.minimum(ei_undir[:, 0], ei_undir[:, 1])
                v = np.maximum(ei_undir[:, 0], ei_undir[:, 1])
                edge_index_undir_np = np.stack([u, v], axis=1)
            else:
                # Derive undirected edges from directed edge_index on the fly
                ei = edge_index_directed.detach().cpu().numpy()
                edge_set = set()
                for i in range(ei.shape[1]):
                    a = int(ei[0, i]); b = int(ei[1, i])
                    if a == b:
                        continue
                    if a > b:
                        a, b = b, a
                    edge_set.add((a, b))
                edge_index_undir_np = np.array(sorted(edge_set), dtype=np.int64)

            edge2id = {(int(a), int(b)): i for i, (a, b) in enumerate(edge_index_undir_np)}

            # Map each directed edge to its undirected counterpart
            ei = edge_index_directed.detach().cpu().numpy()
            dir2undir = np.empty(ei.shape[1], dtype=np.int64)
            for i in range(ei.shape[1]):
                a = int(ei[0, i]); b = int(ei[1, i])
                if a > b:
                    a, b = b, a
                dir2undir[i] = edge2id[(a, b)]

            print(f"[Edge Mapping] {len(edge2id)} undirected edges from {edge_index_directed.size(1)} directed")
            return edge_index_undir_np, edge2id, dir2undir

        edge_index_gt, edge2id, dir2undir = build_edge_mappings(edge_index, gt_pack)

        gt_edge_mask = gt_pack["gt_edge_mask"].astype(bool)
        nodes_expl = gt_pack.get("nodes_expl", None)
        gt_edge_mask_node = gt_pack.get("gt_edge_mask_node", None)

        if gt_edge_mask.shape[1] != len(edge2id):
            raise ValueError(
                f"GT mask size mismatch: {gt_edge_mask.shape[1]} != {len(edge2id)}."
            )

        # Register GT data in the global registry for access during explanation
        gt_registry.set_all(
            gt_edge_mask=gt_edge_mask,
            edge2id=edge2id,
            gt_edge_mask_node=gt_edge_mask_node,
            nodes_expl=nodes_expl,
            edge_index=edge_index_gt,
            verbose=True,
        )
        gt_registry.dir2undir = dir2undir
        gt_registry.num_directed_edges = edge_index.size(1)

        # Sanity checks
        assert gt_registry.gt_edge_mask.shape[1] == gt_registry.edge_index.shape[0]
        assert hasattr(gt_registry, "dir2undir")
        assert gt_registry.dir2undir.shape[0] == edge_index.size(1)

    # ============================================================
    # 2. Train / val / test split
    # ============================================================
    if args.split_path is not None:
        # Load a pre-saved split to ensure reproducibility across runs
        split = torch.load(args.split_path)
        tr, va, te = split["train"], split["val"], split["test"]
        train_mask = torch.zeros(N, dtype=torch.bool, device=device); train_mask[tr] = True
        val_mask   = torch.zeros(N, dtype=torch.bool, device=device); val_mask[va] = True
        test_mask  = torch.zeros(N, dtype=torch.bool, device=device); test_mask[te] = True
    else:
        # Stratified split that respects the multi-label distribution
        Y_np = Y_t.cpu().numpy()
        tr, va, te = multilabel_stratified_split(Y_np, train=0.7, val=0.2, seed=args.seed)
        train_mask = torch.zeros(N, dtype=torch.bool, device=device); train_mask[tr] = True
        val_mask   = torch.zeros(N, dtype=torch.bool, device=device); val_mask[va] = True
        test_mask  = torch.zeros(N, dtype=torch.bool, device=device); test_mask[te] = True

    print(f"Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")

    # ============================================================
    # 3. Model initialisation
    # ============================================================
    print(f"\n{'='*60}")
    print("Initializing")
    print(f"{'='*60}")

    model = SelfExplainableGNN(
        in_dim=F,
        hidden_dim=args.hidden_dim,
        out_dim=L,
        num_layers=args.num_layers,
        dropout=args.dropout,
        explain_hidden=args.edge_scorer_hidden,
        temperature=args.temperature,
        use_gradient_guidance=args.use_gradient,
        grad_weight=args.grad_weight,
        use_label_aware=args.use_label_aware,
        label_edge_hidden=64,
        label_emb_dim=32,
        label_rank=16,
        label_use_correlation=args.label_use_correlation,
        label_gcn_layers=1,
        label_dropout=0.0,
        use_label_gcn_pred=args.use_label_gcn_pred,
        label_pred_emb_dim=args.label_pred_emb_dim,
        label_pred_gcn_layers=args.label_pred_gcn_layers,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Gradient guidance for training: {'ENABLED' if args.use_gradient else 'DISABLED'}")

    # ============================================================
    # 4. Training setup
    # ============================================================
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # Reduce LR on validation plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=20, verbose=True
    )

    # Compute per-label positive class weights to address class imbalance
    Y_train = Y_t[train_mask].cpu().numpy()
    pos = Y_train.sum(axis=0) + 1e-6
    neg = train_mask.sum().item() - pos + 1e-6
    pos_weight = torch.tensor(neg / pos, device=device).float()
    pos_weight = torch.clamp(pos_weight, max=args.pos_weight_clip)

    print(f"\nPos weight range: [{pos_weight.min():.2f}, {pos_weight.max():.2f}]")

    criterion = GradientGuidedLoss(
        lambda_mask_reg=args.lambda_mask_reg,
        lambda_teacher=args.lambda_teacher,
        lambda_delete=args.lambda_delete,
        lambda_masked_target=args.lambda_masked_target,
        lambda_masked_target_floor=0.0,
        teacher_conf=args.teacher_conf,
        pos_weight=pos_weight,
        use_focal=args.use_focal,
        focal_gamma=args.focal_gamma,
        sparsity_mode=args.sparsity_mode,
        base_target_ratio=args.base_target_ratio,
        min_target_ratio=args.min_target_ratio,
        max_target_ratio=args.max_target_ratio,
    )

    # ============================================================
    # 5. Label adjacency (only built when required by model config)
    # ============================================================
    needs_label_adj = (
        (args.use_label_aware and args.label_use_correlation)
        or args.use_label_gcn_pred
    )
    if needs_label_adj:
        print(f"\n{'='*60}")
        print("Building Label Adjacency Matrix")
        print(f"{'='*60}")
        label_adj = build_static_label_adj(
            Y_t, train_mask,
            mode=args.label_adj_mode,
            min_cooc=args.label_adj_min_cooc,
            topk=args.label_adj_topk,
        ).to(device)
        # Distribute the adjacency to all sub-modules that use label graph information
        model.set_label_adj(label_adj, normalize=True, add_self_loops=True)
        print(f"Label adj built and injected (mode={args.label_adj_mode})")

    # ============================================================
    # 6. Training loop
    # ============================================================
    best_val_score = -1.0
    best_state = None
    wait = 0

    print(f"\n{'='*60}")
    print("Training")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        # Advance the 3-stage scheduler and set la_stage_scale accordingly
        criterion.step_auto3stage()
        if criterion._stage == 0:
            # Stage 0: disable label-aware scorer contribution entirely
            model.la_stage_scale.fill_(0.0)
            model._la_detach = True
        elif criterion._stage == 1:
            # Stage 1: ramp in label-aware scorer at half strength; detach to stabilise
            model.la_stage_scale.fill_(0.5)
            model._la_detach = True
        else:
            # Stage 2: full training — small la_scale, gradient flows through scorer
            model.la_stage_scale.fill_(0.10)
            model._la_detach = False

        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Configure label sampling strategy for the k-label aggregated mask
        if args.use_k_label_sampling:
            model._use_k_label_sampling = True
            model._lip_weights_cache = None
            model._num_label_samples = args.num_label_samples
            model._label_sampling_strategy = args.label_sampling_strategy
        else:
            model._use_k_label_sampling = False

        # Only compute the 'removed' encoding when the delete loss is active
        need_removed = (criterion.lambda_delete > 1e-12)
        result = model(
            X_t,
            edge_index,
            return_explanation=True,
            compute_gradient=args.use_gradient,
            compute_removed=need_removed,
            targets=Y_t,
            train_mask=train_mask,
            pos_weight=pos_weight,
        )

        loss, loss_dict = compute_loss(
            result, Y_t, train_mask,
            criterion, edge_index,
            pos_weight, args.use_focal, args.focal_gamma,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        log_la_grad_stats(model, step=epoch, every=25)
        optimizer.step()

        # ---- Validation ----
        if epoch % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(X_t, edge_index, return_explanation=False, compute_gradient=False)
                val_probs = torch.sigmoid(val_out["logits"][val_mask]).cpu().numpy()
                val_targets = Y_t[val_mask].cpu().numpy()

                # Micro AUPRC (treats all labels equally at the instance level)
                prec_micro, rec_micro, _ = precision_recall_curve(
                    val_targets.ravel(), val_probs.ravel()
                )
                micro_auprc = auc(rec_micro, prec_micro)

                # Macro AUPRC (average over labels with at least one positive)
                L = val_targets.shape[1]
                macro_auprc_list = []
                for i in range(L):
                    if val_targets[:, i].sum() > 0:
                        prec, rec, _ = precision_recall_curve(
                            val_targets[:, i], val_probs[:, i]
                        )
                        macro_auprc_list.append(auc(rec, prec))
                macro_auprc = np.mean(macro_auprc_list) if macro_auprc_list else 0.0

                # Combined validation score used for early stopping and LR scheduling
                val_score = micro_auprc + macro_auprc

                # Quick F1 at fixed 0.5 threshold for monitoring
                monitor_preds = (val_probs >= 0.5).astype(int)
                monitor_f1 = f1_score(val_targets, monitor_preds, average='micro', zero_division=0)

            scheduler.step(val_score)

            # Track best model state for early stopping
            if val_score > best_val_score + 1e-3:
                best_val_score = val_score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
                improved = "↑"
            else:
                wait += 1
                improved = " "

            # Verbose logging every 5 eval steps or at the final epoch
            if epoch % (args.eval_every * 5) == 0 or epoch == args.epochs:

                def _to_float(x):
                    """Safely cast a tensor or scalar to Python float."""
                    if x is None:
                        return None
                    try:
                        return float(x.item())
                    except Exception:
                        try:
                            return float(x)
                        except Exception:
                            return None

                def _getf(*keys, default=None):
                    """Return the first available float value from loss_dict by key priority."""
                    for k in keys:
                        if k in loss_dict:
                            v = _to_float(loss_dict.get(k))
                            if v is not None:
                                return v
                    return default

                total   = _getf("loss_total", "total", default=0.0)
                pred    = _getf("loss_pred", "pred_full_weighted", default=0.0)
                teacher = _getf("loss_teacher", "teacher", default=0.0)
                mt      = _getf("loss_masked_target", "masked_target", default=0.0)
                delete  = _getf("loss_delete", "delete", default=0.0)
                reg     = _getf("loss_mask_reg", "mask_reg", default=0.0)

                stage = loss_dict.get("stage", None)
                if stage is not None:
                    try:
                        stage = int(stage.item())
                    except Exception:
                        try:
                            stage = int(stage)
                        except Exception:
                            stage = None

                lmt = _getf("lambda_masked_target_current", "lambda_masked_target", default=None)
                lt  = _getf("lambda_teacher_current",       "lambda_teacher",       default=None)
                ld  = _getf("lambda_delete_current",        "lambda_delete",        default=None)
                lr  = _getf("lambda_mask_reg_current",      "lambda_mask_reg",      default=None)

                m_mean = _getf("mask_mean", default=None)
                m_max  = _getf("mask_max",  default=None)
                m_std  = _getf("mask_std",  default=None)
                m_min  = _getf("mask_min",  default=None)
                m_frac = _getf("mask_sparsity", "mask_frac_gt_05", default=None)

                ema_conf    = _getf("ema_conf",    default=None)
                ema_teacher = _getf("ema_teacher", default=None)
                conf_gate   = _getf("conf_gate",   default=None)
                t_improve   = _getf("teacher_improve", default=None)

                head = f"[{epoch:04d}]{improved} "

                loss_part = f"loss={total:.4f} (pred={pred:.3f} teacher={teacher:.3f}"
                if mt is not None:
                    loss_part += f" mt={mt:.3f}"
                    if lmt is not None:
                        loss_part += f"[λ={lmt:.3f}]"
                if delete is not None:
                    loss_part += f" del={delete:.3f}"
                    if ld is not None:
                        loss_part += f"[λ={ld:.2f}]"
                loss_part += f" reg={reg:.3f})"

                metric_part = (
                    f" val μAUPRC={micro_auprc:.4f} MAUPRC={macro_auprc:.4f} | "
                    f"(monitor μF1@0.5={monitor_f1:.4f}) |"
                )

                sched_part = ""
                if stage is not None:
                    sched_part += f" S{stage} "
                lambdas = []
                if lt is not None: lambdas.append(f"λt={lt:.2f}")
                if ld is not None: lambdas.append(f"λd={ld:.2f}")
                if lr is not None: lambdas.append(f"λr={lr:.2f}")
                if lambdas:
                    sched_part += (" " + " ".join(lambdas) + " |")

                gate_items = []
                if ema_conf is not None:    gate_items.append(f"cE={ema_conf:.3f}")
                if ema_teacher is not None: gate_items.append(f"tE={ema_teacher:.3f}")
                if conf_gate is not None:   gate_items.append(f"g={conf_gate:.2f}")
                if t_improve is not None:   gate_items.append(f"Δt={t_improve:.2f}")
                gate_part = (" " + " ".join(gate_items) + " |") if gate_items else ""

                mask_part = ""
                if m_mean is not None:
                    mask_part += f" mask={m_mean:.3f}"
                    extra = []
                    if m_std is not None:  extra.append(f"σ={m_std:.3f}")
                    if m_min is not None:  extra.append(f"min={m_min:.2f}")
                    if m_max is not None:  extra.append(f"max={m_max:.2f}")
                    if m_frac is not None: extra.append(f">0.5={m_frac:.2f}")
                    if extra:
                        mask_part += " (" + " ".join(extra) + ")"

                log_str = head + loss_part + metric_part + sched_part + gate_part + mask_part
                print(log_str)

            if wait >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # ============================================================
    # 7. Load best checkpoint + threshold optimisation + full evaluation
    # ============================================================
    if best_state is not None:
        model.load_state_dict(best_state)

    # Enable gradient guidance for inference explanations
    model.use_gradient_guidance = True
    model.grad_weight = 0.4

    print(f"\n{'='*60}")
    print("Finding Optimal Threshold on Validation Set")
    print(f"{'='*60}")

    model.eval()
    with torch.no_grad():
        result_all = model(X_t, edge_index, return_explanation=False, compute_gradient=False)

        val_probs = torch.sigmoid(result_all["logits"][val_mask])
        val_targets = Y_t[val_mask]

        test_probs = torch.sigmoid(result_all["logits"][test_mask])
        test_targets = Y_t[test_mask]

    # Grid search for the global threshold that maximises validation Micro-F1
    threshold_finder = OptimalThresholdFinder(
        search_range=(0.1, 0.9),
        search_step=0.01,
        verbose=True,
    )

    threshold_result = threshold_finder.find_optimal_threshold(
        probs=val_probs,
        targets=val_targets,
        strategy="global",
        metric="micro_f1",
    )

    optimal_threshold = threshold_result.best_threshold
    print(f"\n✓ Optimal threshold found: {optimal_threshold:.3f}")
    print(f"  Validation Micro-F1: {threshold_result.best_metric:.4f}")

    print(f"\n{'='*60}")
    print(f"Test Set Evaluation (threshold={optimal_threshold:.3f})")
    print(f"{'='*60}")

    # Apply optimal threshold and compute all prediction metrics
    test_preds_np = threshold_finder.apply_threshold(test_probs, optimal_threshold)
    test_targets_np = test_targets.cpu().numpy()

    pred_results = {
        "micro_f1": f1_score(test_targets_np.ravel(), test_preds_np.ravel(), zero_division=0),
        "macro_f1": f1_score(test_targets_np, test_preds_np, average='macro', zero_division=0),
        "micro_precision": precision_score(test_targets_np.ravel(), test_preds_np.ravel(), zero_division=0),
        "macro_precision": precision_score(test_targets_np, test_preds_np, average='macro', zero_division=0),
        "micro_recall": recall_score(test_targets_np.ravel(), test_preds_np.ravel(), zero_division=0),
        "macro_recall": recall_score(test_targets_np, test_preds_np, average='macro', zero_division=0),
        "hamming_loss": hamming_loss(test_targets_np, test_preds_np),
        "subset_accuracy": accuracy_score(test_targets_np, test_preds_np),
        "optimal_threshold": float(optimal_threshold),
    }

    # Compute AUPRC metrics (threshold-independent)
    test_probs_np = test_probs.cpu().numpy()
    prec_micro, rec_micro, _ = precision_recall_curve(
        test_targets_np.ravel(), test_probs_np.ravel()
    )
    pred_results["micro_auprc"] = auc(rec_micro, prec_micro)

    macro_auprc = []
    for i in range(L):
        if test_targets_np[:, i].sum() > 0:
            prec, rec, _ = precision_recall_curve(
                test_targets_np[:, i], test_probs_np[:, i]
            )
            macro_auprc.append(auc(rec, prec))
    pred_results["macro_auprc"] = np.mean(macro_auprc) if macro_auprc else 0.0

    print(f"\n[Prediction Performance with Optimal Threshold]")
    print(f"  Optimal Threshold: {pred_results['optimal_threshold']:.3f}")
    print(f"  Micro-F1:    {pred_results['micro_f1']:.4f}")
    print(f"  Macro-F1:    {pred_results['macro_f1']:.4f}")
    print(f"  Micro-Prec:  {pred_results['micro_precision']:.4f}")
    print(f"  Macro-Prec:  {pred_results['macro_precision']:.4f}")
    print(f"  Micro-Rec:   {pred_results['micro_recall']:.4f}")
    print(f"  Macro-Rec:   {pred_results['macro_recall']:.4f}")
    print(f"  Micro-AUPRC: {pred_results['micro_auprc']:.4f}")
    print(f"  Macro-AUPRC: {pred_results['macro_auprc']:.4f}")
    print(f"  Hamming:     {pred_results['hamming_loss']:.4f}")
    print(f"  Subset-Acc:  {pred_results['subset_accuracy']:.4f}")

    # ---- Select evaluation nodes (GT nodes prioritised) ----
    test_nodes = test_mask.nonzero(as_tuple=True)[0].cpu().numpy().astype(int).tolist()
    rng = np.random.RandomState(args.seed)
    rng.shuffle(test_nodes)

    if hasattr(args, "eval_nodes_path") and args.eval_nodes_path is not None:
        eval_nodes = [int(n) for n in np.load(args.eval_nodes_path).tolist()]
        print(f"\nEval nodes: {len(eval_nodes)} (loaded from {args.eval_nodes_path})")
    else:
        eval_nodes = []
        # First, add test nodes that have GT explanations available
        if gt_registry.is_initialized and gt_registry.nodes_expl is not None:
            gt_nodes = set(int(n) for n in gt_registry.nodes_expl.tolist())
            eval_nodes.extend([n for n in test_nodes if n in gt_nodes][:args.num_eval_nodes])

        # Fill remaining budget with random test nodes
        if len(eval_nodes) < args.num_eval_nodes:
            used = set(eval_nodes)
            fill = [n for n in test_nodes if n not in used]
            eval_nodes.extend(fill[:args.num_eval_nodes - len(eval_nodes)])

        print(f"\nEval nodes: {len(eval_nodes)} (GT prioritized: {gt_registry.is_initialized})")

    # ---- Pre-compute per-label full-graph edge masks ----
    print(f"\n{'='*60}")
    print("Pre-computing Per-Label Full Graph Masks")
    print(f"{'='*60}")

    result_all = model(X_t, edge_index, return_explanation=False, compute_gradient=False)
    probs_all = torch.sigmoid(result_all["logits"]).detach()

    # Collect all labels predicted for any eval node to avoid redundant computation
    active_labels_in_eval = set()
    for node_idx in eval_nodes:
        probs = probs_all[node_idx].cpu().numpy()
        pred_labels = np.where(probs >= optimal_threshold)[0].tolist()
        if not pred_labels:
            pred_labels = [int(np.argmax(probs))]
        active_labels_in_eval.update(pred_labels)

    active_labels_in_eval = sorted(list(active_labels_in_eval))
    print(f"Active labels in eval set: {len(active_labels_in_eval)}")

    label_masks_for_eval = model.compute_full_graph_edge_masks_multilabel(
        X_t, edge_index, active_labels_in_eval,
        Y=Y_t, train_mask=train_mask, pos_weight=pos_weight,
        use_corr_refine=True,
        corr_refine_beta=0.2,
        corr_refine_topk=5,
    )
    print(f"Computed per-label masks: {len(label_masks_for_eval)}")

    # ---- Fidelity evaluation ----
    print(f"\n{'='*60}")
    print("Fidelity Evaluation")
    print(f"{'='*60}")

    fid_eval = FidelityEvaluator(
        min_edges=args.topk_min_edges,
        max_frac=args.topk_max_frac,
        cap=args.topk_cap,
    )
    fid_results = fid_eval.evaluate_batch(
        model, X_t, edge_index,
        node_indices=eval_nodes,
        k_hop=args.k_hop,
        threshold=optimal_threshold,
        verbose=True,
        label_masks=label_masks_for_eval,
        Y=Y_t,
        train_mask=train_mask,
        pos_weight=pos_weight,
        use_node_local_necessity=True,
        alpha=0.6,
    )

    print(f"\n[Fidelity Results]")
    print(f"  Fidelity+: {fid_results['fidelity_pos_mean']:.4f} ± {fid_results['fidelity_pos_std']:.4f}")
    print(f"  Fidelity-: {fid_results['fidelity_neg_mean']:.4f} ± {fid_results['fidelity_neg_std']:.4f}")
    print(f"  Sparsity:  {fid_results['sparsity_mean']:.4f} ± {fid_results['sparsity_std']:.4f}")

    # ---- GT-based explanation quality evaluation ----
    gt_results = None
    if gt_registry.is_initialized and gt_registry.nodes_expl is not None:
        print(f"\n{'='*60}")
        print("GT-based Evaluation")
        print(f"{'='*60}")

        gt_eval = GTExplanationEvaluator(
            min_edges=args.topk_min_edges,
            max_frac=args.topk_max_frac,
            cap=args.topk_cap,
        )
        gt_results = gt_eval.evaluate_batch(
            model, X_t, edge_index, Y_t,
            node_indices=eval_nodes,
            k_hop=args.k_hop,
            verbose=True,
            train_mask=train_mask,
            pos_weight=pos_weight,
            label_masks=label_masks_for_eval,
            threshold=optimal_threshold,
        )

        if gt_results.get("status") == "ok":
            print(f"\n[GT Multi-Label Macro-Average]")
            print(f"  Nodes: {gt_results.get('num_evaluated', 0)}, Labels: {gt_results.get('num_labels', 0)}")
            print(f"  Precision: {gt_results.get('macro_precision_mean', 0.0):.4f} ± {gt_results.get('macro_precision_std', 0.0):.4f}")
            print(f"  Recall:    {gt_results.get('macro_recall_mean', 0.0):.4f} ± {gt_results.get('macro_recall_std', 0.0):.4f}")
            print(f"  IoU:       {gt_results.get('macro_iou_mean', 0.0):.4f} ± {gt_results.get('macro_iou_std', 0.0):.4f}")
            print(f"  F1:        {gt_results.get('macro_f1_mean', 0.0):.4f} ± {gt_results.get('macro_f1_std', 0.0):.4f}")
        else:
            print(f"\n⚠️  GT Evaluation failed: {gt_results.get('status', 'unknown')}")

    # ============================================================
    # 8. Save checkpoint and evaluation results
    # ============================================================
    os.makedirs("ckpts", exist_ok=True)
    threshold_config = {
        "optimal_threshold": float(optimal_threshold),
        "val_metric": float(threshold_result.best_metric),
        "strategy": "global",
        "search_range": [0.1, 0.9],
        "search_step": 0.01,
    }

    # Build filename suffix to distinguish model variants
    suffix = ""
    if args.use_label_gcn_pred:
        suffix += "_lagcn"
    if args.use_label_aware:
        suffix += "_la"
        if args.label_use_correlation:
            suffix += "_corr"

    save_path = f"ckpts/{args.dataset}_explainable{suffix}.pt"

    save_dict = {
        'model_state': model.state_dict(),
        'args': vars(args),
        'test_results': pred_results,
        'threshold_config': threshold_config,
    }
    torch.save(save_dict, save_path)
    print(f"\nModel saved to {save_path}")

    # ---- Assemble and save unified evaluation JSON ----
    unified_eval = {
        "model_type": "explainable",
        "dataset": args.dataset,
        "threshold_config": threshold_config,
        "use_label_aware": args.use_label_aware,
        "label_use_correlation": args.label_use_correlation if args.use_label_aware else False,
        "prediction_test": {
            "micro_precision": pred_results["micro_precision"],
            "micro_recall": pred_results["micro_recall"],
            "micro_f1": pred_results["micro_f1"],
            "macro_precision": pred_results["macro_precision"],
            "macro_recall": pred_results["macro_recall"],
            "macro_f1": pred_results["macro_f1"],
            "hamming_loss": pred_results["hamming_loss"],
            "subset_accuracy": pred_results["subset_accuracy"],
            "micro_auprc": pred_results["micro_auprc"],
            "macro_auprc": pred_results["macro_auprc"],
        },
        "fidelity": {
            "fidelity_pos_mean": fid_results.get("fidelity_pos_mean"),
            "fidelity_pos_std": fid_results.get("fidelity_pos_std"),
            "fidelity_neg_mean": fid_results.get("fidelity_neg_mean"),
            "fidelity_neg_std": fid_results.get("fidelity_neg_std"),
            "sparsity_mean": fid_results.get("sparsity_mean"),
            "sparsity_std": fid_results.get("sparsity_std"),
            "edge_mask_mean": fid_results.get("edge_mask_mean"),
            "runtime_sec_per_node_mean": fid_results.get("runtime_sec_per_node_mean"),
            "runtime_sec_per_node_std": fid_results.get("runtime_sec_per_node_std"),
        },
        "gt": None,
    }

    if gt_results and gt_results.get("status") == "ok":
        unified_eval["gt"] = {
            "macro_precision_mean": gt_results.get("macro_precision_mean", 0.0),
            "macro_precision_std": gt_results.get("macro_precision_std", 0.0),
            "macro_recall_mean": gt_results.get("macro_recall_mean", 0.0),
            "macro_recall_std": gt_results.get("macro_recall_std", 0.0),
            "macro_f1_mean": gt_results.get("macro_f1_mean", 0.0),
            "macro_f1_std": gt_results.get("macro_f1_std", 0.0),
            "macro_iou_mean": gt_results.get("macro_iou_mean", 0.0),
            "macro_iou_std": gt_results.get("macro_iou_std", 0.0),
            "coverage_mean": gt_results.get("coverage_mean"),
            "coverage_std": gt_results.get("coverage_std"),
            "runtime_sec_per_node_mean": gt_results.get("runtime_sec_per_node_mean"),
            "runtime_sec_per_node_std": gt_results.get("runtime_sec_per_node_std"),
            "macro_ap_mean": gt_results.get("macro_ap_mean"),
            "macro_ap_std": gt_results.get("macro_ap_std"),
        }

    os.makedirs("eval_logs", exist_ok=True)
    out_path = f"eval_logs/{args.dataset}_explainable{suffix}_unified_eval.json"
    with open(out_path, "w") as f:
        json.dump(unified_eval, f, indent=2)
    print(f"Unified evaluation saved to {out_path}")

    return model


# ============================================================
# Command-line interface
# ============================================================


def main():
    """Parse command-line arguments and launch the training pipeline."""
    parser = argparse.ArgumentParser(description="Train SEMGNN")

    # ---- Dataset ----
    parser.add_argument("--dataset", type=str, default="synthetic_anchor",
                        choices=["synthetic_motif", "synthetic_anchor", "blogcatalog", "youtube"])
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to dataset directory. Defaults to ./dataset/<dataset>.")
    parser.add_argument("--split_path", type=str, default=None,
                        help="Path to a pre-saved train/val/test split .pt file.")

    # ---- Model architecture ----
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--edge_scorer_hidden", type=int, default=64,
                        help="Hidden size of the base EdgeScorer MLP.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sigmoid temperature for soft edge masks.")

    # ---- Training hyperparameters ----
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--pos_weight_clip", type=float, default=50.0,
                        help="Maximum value for per-label positive class weights.")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience (epochs without improvement).")
    parser.add_argument("--eval_every", type=int, default=5,
                        help="Run validation every N epochs.")
    parser.add_argument("--use_focal", action="store_true", default=False,
                        help="Use focal loss instead of standard BCE for the prediction term.")
    parser.add_argument("--focal_gamma", type=float, default=1.0)

    # ---- Loss weights ----
    parser.add_argument("--lambda_mask_reg", type=float, default=1.5)
    parser.add_argument("--lambda_teacher", type=float, default=0.5)
    parser.add_argument("--lambda_delete", type=float, default=0.2)
    parser.add_argument("--lambda_masked_target", type=float, default=0.5)
    parser.add_argument("--teacher_conf", type=float, default=0.6,
                        help="Confidence threshold used in the delete loss term.")

    # ---- Gradient guidance ----
    parser.add_argument("--use_gradient", action="store_true", default=False,
                        help="Enable gradient-based necessity scores during training. Inference-time explanation uses gradient guidance by design.",
    )
    parser.add_argument("--grad_weight", type=float, default=0.4,
                        help="Blend weight for gradient necessity in the combined mask.")

    # ---- Sparsity regularisation ----
    parser.add_argument("--sparsity_mode", type=str, default="adaptive_degree",
                        choices=["fixed", "adaptive_degree", "adaptive_local"])
    parser.add_argument("--base_target_ratio", type=float, default=0.10)
    parser.add_argument("--min_target_ratio", type=float, default=0.02)
    parser.add_argument("--max_target_ratio", type=float, default=0.30)

    # ---- Evaluation settings ----
    parser.add_argument("--k_hop", type=int, default=2,
                        help="Number of hops for k-hop subgraph extraction during explanation.")
    parser.add_argument("--num_eval_nodes", type=int, default=100,
                        help="Number of nodes to evaluate explanations on.")
    parser.add_argument("--topk_max_frac", type=float, default=0.25,
                        help="Maximum fraction of subgraph edges to select per explanation.")
    parser.add_argument("--topk_min_edges", type=int, default=3,
                        help="Minimum number of edges to always select per explanation.")
    parser.add_argument("--topk_cap", type=int, default=800,
                        help="Hard cap on the number of selected edges per explanation.")
    parser.add_argument("--eval_nodes_path", type=str, default=None,
                        help="Path to a .npy file of specific node indices to evaluate.")

    # ---- Label-aware edge scorer ----
    parser.add_argument( "--use_label_aware", action=argparse.BooleanOptionalAction, default=True, 
                        help="Enable the LabelAwareEdgeScorer (default: enabled). Use --no-use_label_aware to disable.",)
    parser.add_argument('--label_use_correlation', action=argparse.BooleanOptionalAction, default=True,
                        help="Enable label correlation GCN inside LabelAwareEdgeScorer (default: enabled). Use --no-label_use_correlation to disable.")
    parser.add_argument('--label_adj_mode', type=str, default='jaccard',
                        choices=['jaccard', 'npmi'],
                        help="Similarity measure for building the label adjacency matrix.")
    parser.add_argument('--label_adj_min_cooc', type=int, default=3,
                        help="Minimum co-occurrence count for a label pair to appear in the adjacency.")
    parser.add_argument('--label_adj_topk', type=int, default=5,
                        help="Top-k neighbours to keep per label (0 = keep all).")

    # ---- K-label sampling (reduces memory during training) ----
    parser.add_argument('--num_label_samples', type=int, default=4,
                        help="Number of labels to sample per step when use_k_label_sampling is on.")
    parser.add_argument('--label_sampling_strategy', type=str, default='uniform',
                        help="Label sampling strategy (currently only 'uniform' is supported).")
    parser.add_argument('--use_k_label_sampling', action=argparse.BooleanOptionalAction, default=True,
                        help="Enable k-label aggregated mask computation during training (default: enabled). Use --no-use_k_label_sampling to disable.")

    # ---- Miscellaneous ----
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    # ---- Label-GCN prediction head ----
    parser.add_argument("--use_label_gcn_pred", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable the LabelGCNPredictor prediction head (default: enabled). Use --no-use_label_gcn_pred to disable.")
    parser.add_argument("--label_pred_emb_dim", type=int, default=64,
                        help="Label embedding dimensionality for LabelGCNPredictor.")
    parser.add_argument("--label_pred_gcn_layers", type=int, default=1,
                        help="Number of GCN layers in LabelGCNPredictor.")

    args = parser.parse_args()
    train_explainable(args)


if __name__ == "__main__":
    main()