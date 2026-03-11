"""
SEMGNN.py

Self-Explainable Multi-label Graph Neural Network (SEMGNN).

Architecture overview:
  - BalancedSAGEEncoder: graph encoder to produce node embeddings
  - EdgeScorer / LabelAwareEdgeScorer: edge importance scorer (sufficiency)
  - LabelGCNPredictor: label-correlation-aware prediction head
  - SelfExplainableGNN: main model combining encoder, predictor, and explainer
  - GradientGuidedLoss: multi-term training objective with auto 3-stage scheduler
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Sequence, Union
import numpy as np

# ============================================================
# Utility functions for label adjacency matrix
# ============================================================


def _label_adj_stats(A: torch.Tensor, name: str = "label_adj") -> str:
    """
    Return a one-line diagnostic summary string for a label adjacency matrix.

    Supports both dense and sparse (COO) adjacency tensors.  Useful for
    logging/debugging the state of the label graph at various pipeline stages.

    Args:
        A:    Label adjacency matrix (dense or sparse COO).
        name: Tag string prepended to the summary for identification.

    Returns:
        A single-line string containing shape, nnz, density, degree
        statistics, and value range.
    """
    if A is None:
        return f"[{name}] None"

    with torch.no_grad():
        if A.is_sparse:
            A_ = A.coalesce()
            L = A_.size(0)
            nnz = int(A_._nnz())
            density = nnz / float(max(1, L * L))
            deg = torch.sparse.sum(A_, dim=1).to_dense()
            vals = A_.values()
            msg = (
                f"[{name}] sparse shape={tuple(A_.shape)} nnz={nnz} density={density:.4f} "
                f"deg(min/mean/max)=({deg.min().item():.3f}/{deg.mean().item():.3f}/{deg.max().item():.3f}) "
                f"val(min/max)=({vals.min().item():.3f}/{vals.max().item():.3f})"
            )
            return msg
        else:
            A_ = A
            L = A_.size(0)
            nnz = int((A_ != 0).sum().item())
            density = nnz / float(max(1, L * L))
            deg = A_.sum(dim=1)
            sym_err = (A_ - A_.t()).abs().mean().item()
            msg = (
                f"[{name}] dense shape={tuple(A_.shape)} nnz={nnz} density={density:.4f} "
                f"deg(min/mean/max)=({deg.min().item():.3f}/{deg.mean().item():.3f}/{deg.max().item():.3f}) "
                f"sym_err(mean|A-A^T|)={sym_err:.6f} "
                f"val(min/max)=({A_.min().item():.3f}/{A_.max().item():.3f})"
            )
            return msg


def _stabilize_dense_label_adj(
    A: torch.Tensor,
    *,
    symmetrize: bool = True,
    clamp_nonneg: bool = True,
) -> torch.Tensor:
    """
    Apply numerical stabilization to a dense label adjacency matrix.

    Operations performed (in order):
      1. Cast to float32.
      2. Optionally symmetrize: A <- 0.5 * (A + A^T).
      3. Optionally clamp negative entries to zero.
      4. Replace NaN / +Inf / -Inf with 0.

    Args:
        A:            Dense label adjacency tensor of shape [L, L].
        symmetrize:   Whether to force symmetry via averaging with transpose.
        clamp_nonneg: Whether to clamp all values to be non-negative.

    Returns:
        Stabilized dense float32 tensor of the same shape.
    """
    A = A.float()
    if symmetrize:
        A = 0.5 * (A + A.t())
    if clamp_nonneg:
        A = A.clamp(min=0.0)
    # Replace any remaining NaN or infinite values with 0
    A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    return A


# ============================================================
# Label-GCN prediction head
# ============================================================


class LabelGCNPredictor(nn.Module):
    """
    Label co-occurrence GCN used as a prediction head.

    Propagates learnable label embeddings through a label co-occurrence graph
    via GCN layers, then computes per-label logits for each node via an inner
    product between the (projected) node embedding and the propagated label
    embeddings.

    This module is structurally analogous to the label_gcn inside
    LabelAwareEdgeScorer, but operates at prediction time rather than at edge
    scoring time.

    Args:
        node_dim:       Dimensionality of input node embeddings (encoder output).
        num_labels:     Total number of labels L.
        label_emb_dim:  Dimensionality of label embeddings.
        num_gcn_layers: Number of GCN propagation layers (1–2 is typically sufficient).
        proj_hidden:    Hidden size of the node projection MLP.
                        If None, a single linear layer is used instead.
        dropout:        Dropout probability applied after activations.
    """

    def __init__(
        self,
        node_dim: int,
        num_labels: int,
        label_emb_dim: int = 64,
        num_gcn_layers: int = 1,
        proj_hidden: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.dropout_p = dropout

        # Node projection: map node_emb into the same space as label embeddings
        proj_out = label_emb_dim
        if proj_hidden is not None:
            self.node_proj = nn.Sequential(
                nn.Linear(node_dim, proj_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(proj_hidden, proj_out),
            )
        else:
            self.node_proj = nn.Sequential(
                nn.Linear(node_dim, proj_out),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        # Learnable label embeddings (randomly initialised, refined by GCN)
        self.label_embedding = nn.Embedding(num_labels, label_emb_dim)

        # Stack of linear layers used as GCN weight matrices
        self.label_gcn = nn.ModuleList([
            nn.Linear(label_emb_dim, label_emb_dim, bias=False)
            for _ in range(num_gcn_layers)
        ])

        # Normalised label adjacency stored as a non-trainable buffer (set externally)
        self.register_buffer("_label_adj_norm", None, persistent=True)
        self._warned_no_label_adj = False

        # Per-label output bias
        self.label_bias = nn.Parameter(torch.zeros(num_labels))

        self._init_weights()

    def _init_weights(self):
        """Initialise weights with small values to keep early-training stable."""
        nn.init.normal_(self.label_embedding.weight, std=0.02)
        nn.init.zeros_(self.label_bias)
        for layer in self.label_gcn:
            nn.init.xavier_uniform_(layer.weight, gain=0.5)

    @torch.no_grad()
    def set_label_adj(
        self,
        label_adj: torch.Tensor,
        normalize: bool = True,
        add_self_loops: bool = True,
        *,
        symmetrize: bool = True,
        clamp_nonneg: bool = True,
        verbose: bool = False,
    ):
        """
        Pre-process and store the normalised label adjacency matrix.

        The adjacency is first converted to a dense float32 tensor, stabilised,
        then optionally augmented with self-loops and symmetrically normalised
        (D^{-1/2} A D^{-1/2}).  The result is stored as a sparse COO buffer so
        that torch.spmm can be used efficiently during forward passes.

        Args:
            label_adj:       Raw label co-occurrence matrix [L, L] (dense or sparse).
            normalize:       Apply symmetric normalisation.
            add_self_loops:  Add identity before normalisation.
            symmetrize:      Force symmetry via (A + A^T) / 2.
            clamp_nonneg:    Clamp negative entries to zero.
            verbose:         Print diagnostic statistics after setting.
        """
        L = self.num_labels
        dev = self.label_embedding.weight.device
        A = label_adj

        # Convert sparse input to dense for uniform processing (L is typically small)
        if A.is_sparse:
            A = A.coalesce().to_dense()
        A = A.to(device=dev, dtype=torch.float32)
        A = _stabilize_dense_label_adj(A, symmetrize=symmetrize, clamp_nonneg=clamp_nonneg)

        # Guard against all-zero matrices by falling back to identity
        if int((A != 0).sum().item()) == 0:
            A = torch.eye(L, device=dev, dtype=torch.float32)

        # Convert back to sparse COO for efficient spmm
        idx = (A != 0).nonzero(as_tuple=False).t()
        val = A[idx[0], idx[1]]
        A = torch.sparse_coo_tensor(idx, val, (L, L), device=dev).coalesce()

        if add_self_loops:
            eye = torch.arange(L, device=dev)
            I = torch.sparse_coo_tensor(
                torch.stack([eye, eye], dim=0),
                torch.ones(L, device=dev),
                (L, L),
            ).coalesce()
            A = (A + I).coalesce()

        if normalize:
            # Symmetric normalisation: D^{-1/2} A D^{-1/2}
            deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1e-12)
            deg_inv_sqrt = deg.pow(-0.5)
            r, c = A.indices()
            v = A.values() * deg_inv_sqrt[r] * deg_inv_sqrt[c]
            A = torch.sparse_coo_tensor(A.indices(), v, A.size(), device=dev).coalesce()

        self._label_adj_norm = A

        if verbose:
            print(_label_adj_stats(self._label_adj_norm, name="PredGCN/_label_adj_norm"))

    def _get_label_emb(self) -> torch.Tensor:
        """
        Return GCN-propagated label embeddings of shape [L, label_emb_dim].

        If no adjacency matrix has been set, the raw (unpropagated) embeddings
        are returned, effectively degrading to a standard embedding inner-product
        classifier.
        """
        h = self.label_embedding.weight  # [L, d]

        if self._label_adj_norm is not None:
            A = self._label_adj_norm
            for layer in self.label_gcn:
                h = layer(h)            # Linear transform
                h = torch.spmm(A, h)   # Neighbourhood aggregation (sparse matmul)
                h = F.leaky_relu(h, 0.2)
                if self.dropout_p > 0:
                    h = F.dropout(h, p=self.dropout_p, training=self.training)

        return h  # [L, d]

    def forward(self, node_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-label logits via node–label inner products.

        Args:
            node_emb: Node embeddings from the encoder, shape [N, node_dim].

        Returns:
            logits: Raw (pre-sigmoid) logits of shape [N, L].
        """
        node_h = self.node_proj(node_emb)      # [N, label_emb_dim]
        label_h = self._get_label_emb()        # [L, label_emb_dim]

        logits = node_h @ label_h.t()          # [N, L] inner product
        logits = logits + self.label_bias      # Add per-label bias

        return logits


# ============================================================
# Graph encoder building blocks
# ============================================================


class SAGEConvLayer(nn.Module):
    """
    Single GraphSAGE layer with mean aggregation (PyG edge convention: src -> dst).

    Concatenates the target node's own features with the weighted mean of
    its neighbours' features, then applies a linear projection.

    Args:
        in_dim:  Input feature dimensionality.
        out_dim: Output feature dimensionality.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # Input is [self || mean_neighbour], so 2 * in_dim
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x:           Node feature matrix [N, in_dim].
            edge_index:  Edge indices [2, E] (src, dst).
            edge_weight: Optional per-edge scalar weights [E].

        Returns:
            Updated node features [N, out_dim].
        """
        src, dst = edge_index
        N = x.size(0)

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype)
        else:
            edge_weight = edge_weight.to(dtype=x.dtype)

        # Weighted source messages
        msg = x[src] * edge_weight.unsqueeze(-1)

        # Compute weighted in-degree for mean normalisation
        deg = torch.zeros(N, device=x.device, dtype=x.dtype)
        deg.scatter_add_(0, dst, edge_weight)
        deg = deg.clamp(min=1e-6)

        # Aggregate messages into destination nodes
        agg = torch.zeros(N, x.size(1), device=x.device, dtype=x.dtype)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand(-1, x.size(1)), msg)
        agg = agg / deg.unsqueeze(-1)  # Mean aggregation

        # Concatenate self features with aggregated neighbour features
        out = torch.cat([x, agg], dim=-1)
        return self.linear(out)


class EdgeScorer(nn.Module):
    """
    MLP-based edge importance scorer.

    Given node embeddings, concatenates the two endpoint representations for
    each edge and passes them through a small MLP to produce a scalar
    importance score per edge.

    Args:
        node_dim:   Dimensionality of node embeddings.
        hidden_dim: Hidden layer size of the MLP.
    """

    def __init__(self, node_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )
        # Initialise the output layer near zero for training stability
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.1)

    def forward(self, node_emb: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_emb:   Node embeddings [N, node_dim].
            edge_index: Edge indices [2, E].

        Returns:
            Scalar importance score per edge, shape [E].
        """
        row, col = edge_index
        edge_feat = torch.cat([node_emb[row], node_emb[col]], dim=-1)
        return self.mlp(edge_feat).squeeze(-1)


class LabelAwareEdgeScorer(nn.Module):
    """
    Label-conditioned edge importance scorer using a low-rank bilinear model.

    For each label (or a subset of labels), this module produces edge scores by
    computing a bilinear product between a label-specific vector and a shared
    edge representation.  An optional label co-occurrence GCN further enriches
    the label embeddings before scoring.

    The scorer returns raw (pre-sigmoid) logits.

    Usage examples::

        scores = scorer(h, edge_index, label_id=3)                # (E,)
        scores = scorer(h, edge_index, label_ids=[3, 7, 9])       # (E, 3)
        scores = scorer(h, edge_index, return_all_labels=True)    # (E, L)

    Args:
        node_dim:               Dimensionality of input node embeddings.
        num_labels:             Total number of labels L.
        edge_hidden:            Hidden size for the edge encoder MLP.
        label_emb_dim:          Dimensionality of label embeddings.
        rank:                   Rank of the bilinear interaction (low-rank factorisation).
        use_label_correlation:  Whether to propagate label embeddings through a GCN.
        label_gcn_layers:       Number of GCN layers (used when use_label_correlation=True).
        dropout:                Dropout probability inside the label GCN.
    """

    def __init__(
        self,
        node_dim: int,
        num_labels: int,
        edge_hidden: int = 64,
        label_emb_dim: int = 32,
        rank: int = 16,
        use_label_correlation: bool = False,
        label_gcn_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_labels = int(num_labels)
        self.rank = int(rank)
        self.use_label_correlation = bool(use_label_correlation)
        self.label_gcn_layers = int(label_gcn_layers)
        self.dropout = float(dropout)

        # Shared edge encoder: maps concatenated endpoint features to a hidden vector
        self.edge_encoder = nn.Sequential(
            nn.Linear(node_dim * 2, edge_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(edge_hidden, edge_hidden),
            nn.LeakyReLU(0.2),
        )
        # Project edge hidden vector down to rank-r space
        self.edge_to_rank = nn.Linear(edge_hidden, rank, bias=False)

        # Per-label embeddings
        self.label_embedding = nn.Embedding(self.num_labels, label_emb_dim)
        # Project label embedding to rank-r space (Tanh keeps magnitudes bounded)
        self.label_to_rank = nn.Sequential(
            nn.Linear(label_emb_dim, rank, bias=True),
            nn.Tanh(),
        )
        # Per-label scalar bias
        self.label_bias = nn.Embedding(self.num_labels, 1)

        # Normalised label adjacency (set externally via set_label_adj)
        self.register_buffer("_label_adj_norm", None, persistent=True)
        self._warned_no_label_adj = False

        # Optional label correlation GCN (only instantiated when requested)
        if self.use_label_correlation:
            self.label_gcn = nn.ModuleList(
                [nn.Linear(label_emb_dim, label_emb_dim, bias=False)
                 for _ in range(self.label_gcn_layers)]
            )

        self._init_weights()

    def _init_weights(self):
        """Initialise weights conservatively to prevent early instability."""
        nn.init.normal_(self.label_embedding.weight, std=0.1)
        nn.init.xavier_uniform_(self.edge_to_rank.weight, gain=0.05)
        lin = self.label_to_rank[0]
        nn.init.xavier_uniform_(lin.weight, gain=0.05)
        nn.init.zeros_(lin.bias)
        nn.init.zeros_(self.label_bias.weight)

    @torch.no_grad()
    def set_label_adj(
        self,
        label_adj: torch.Tensor,
        normalize: bool = True,
        add_self_loops: bool = True,
        *,
        symmetrize: bool = True,
        clamp_nonneg: bool = True,
        verbose: bool = False,
    ):
        """
        Pre-process and store the normalised label adjacency for GCN propagation.

        Raises RuntimeError if use_label_correlation=False, since the adjacency
        is only meaningful when label correlation is enabled.

        Args:
            label_adj:       Raw label co-occurrence matrix [L, L].
            normalize:       Apply symmetric D^{-1/2} A D^{-1/2} normalisation.
            add_self_loops:  Add self-loops before normalisation.
            symmetrize:      Symmetrise by averaging with transpose.
            clamp_nonneg:    Clamp negative entries to zero.
            verbose:         Print diagnostic statistics after processing.
        """
        if not self.use_label_correlation:
            raise RuntimeError("use_label_correlation=False, but set_label_adj() was called.")

        L = self.num_labels
        dev = self.label_embedding.weight.device

        A = label_adj
        if A.is_sparse:
            A = A.coalesce().to_dense()
        else:
            A = A.to(device=dev, dtype=torch.float32)

        A = A.to(device=dev, dtype=torch.float32)
        A = _stabilize_dense_label_adj(A, symmetrize=symmetrize, clamp_nonneg=clamp_nonneg)

        # Guard against degenerate all-zero adjacency
        if int((A != 0).sum().item()) == 0:
            A = torch.eye(L, device=dev, dtype=torch.float32)

        if add_self_loops:
            A = A + torch.eye(L, device=dev, dtype=A.dtype)

        if normalize:
            deg = A.sum(dim=1).clamp(min=1e-12)
            deg_inv_sqrt = deg.pow(-0.5)
            A = deg_inv_sqrt[:, None] * A * deg_inv_sqrt[None, :]

        # Store as sparse COO for efficient spmm during forward
        idx = (A != 0).nonzero(as_tuple=False).t()
        if idx.numel() == 0:
            # Fallback: identity if no non-zero entries remain
            idx = torch.stack([torch.arange(L, device=dev), torch.arange(L, device=dev)], dim=0)
            val = torch.ones(L, device=dev, dtype=A.dtype)
        else:
            val = A[idx[0], idx[1]]

        A_sparse = torch.sparse_coo_tensor(idx, val, (L, L), device=dev).coalesce()
        self._label_adj_norm = A_sparse

        if verbose:
            print(_label_adj_stats(A, name="LA/raw_or_norm_dense_before_sparse"))
            print(_label_adj_stats(self._label_adj_norm, name="LA/_label_adj_norm"))

    def _get_label_rank_matrix(self) -> torch.Tensor:
        """
        Return the rank-r label representation matrix of shape [L, rank].

        If label correlation is enabled and an adjacency has been set, the raw
        label embeddings are first propagated through the label GCN before being
        projected to rank-r space.  Otherwise, raw embeddings are used directly
        with a one-time warning.
        """
        emb = self.label_embedding.weight

        if self.use_label_correlation and (self._label_adj_norm is None):
            if not self._warned_no_label_adj:
                print("[Warning][LabelAwareEdgeScorer] use_label_correlation=True but _label_adj_norm is None. "
                    "Falling back to raw label embeddings (no label-graph propagation).")
                self._warned_no_label_adj = True

        if self.use_label_correlation and (self._label_adj_norm is not None):
            h = emb
            A = self._label_adj_norm
            for layer in self.label_gcn:
                h = layer(h)
                h = torch.spmm(A, h)
                h = F.leaky_relu(h, 0.2)
                if self.dropout > 0:
                    h = F.dropout(h, p=self.dropout, training=self.training)
            emb = h

        label_r = self.label_to_rank(emb)
        return label_r

    def forward(
        self,
        node_emb: torch.Tensor,
        edge_index: torch.Tensor,
        label_id: Optional[int] = None,
        label_ids: Optional[Union[Sequence[int], torch.Tensor]] = None,
        return_all_labels: bool = False,
        chunk_size: int = 200_000,
    ) -> torch.Tensor:
        """
        Compute label-conditioned edge importance scores (raw logits).

        Exactly one of label_id, label_ids, or return_all_labels must be
        specified.  For large edge counts the all-labels path supports chunked
        computation to limit peak memory usage.

        Args:
            node_emb:          Node embeddings [N, node_dim].
            edge_index:        Edge indices [2, E].
            label_id:          Single label index -> output shape [E].
            label_ids:         List / tensor of K label indices -> output [E, K].
            return_all_labels: If True, return scores for all L labels -> [E, L].
            chunk_size:        Maximum number of edges to process at once when
                               returning all-label scores.

        Returns:
            Edge score tensor.  Shape depends on the query mode (see above).
        """
        if (label_id is None) and (label_ids is None) and (not return_all_labels):
            raise ValueError("Must specify label_id or label_ids or return_all_labels=True")
        if (label_id is not None) and (label_ids is not None):
            raise ValueError("Specify only one of label_id or label_ids")

        # Encode each edge as the concatenation of its endpoint embeddings
        row, col = edge_index
        edge_feat = torch.cat([node_emb[row], node_emb[col]], dim=-1)
        edge_h = self.edge_encoder(edge_feat)   # [E, edge_hidden]
        edge_r = self.edge_to_rank(edge_h)      # [E, rank]
        label_r_all = self._get_label_rank_matrix()  # [L, rank]

        # --- Single label query ---
        if label_id is not None:
            lid = int(label_id)
            lr = label_r_all[lid: lid + 1]                          # [1, rank]
            b = self.label_bias.weight[lid: lid + 1].view(1)
            return (edge_r @ lr.t()).squeeze(-1) + b                # [E]

        # --- Multiple label query ---
        if label_ids is not None:
            if isinstance(label_ids, torch.Tensor):
                lids = label_ids.to(device=node_emb.device, dtype=torch.long).view(-1)
            else:
                lids = torch.tensor(list(label_ids), device=node_emb.device, dtype=torch.long).view(-1)
            lr = label_r_all.index_select(0, lids)                  # [K, rank]
            b = self.label_bias.weight.index_select(0, lids).view(1, -1)  # [1, K]
            return edge_r @ lr.t() + b                              # [E, K]

        # --- All-labels query (optionally chunked to save memory) ---
        L = self.num_labels
        b_all = self.label_bias.weight.view(1, L)

        if chunk_size is None or edge_r.size(0) <= chunk_size:
            return edge_r @ label_r_all.t() + b_all                # [E, L]

        outs = []
        E = edge_r.size(0)
        for s in range(0, E, chunk_size):
            e = min(s + chunk_size, E)
            outs.append(edge_r[s:e] @ label_r_all.t() + b_all)
        return torch.cat(outs, dim=0)                               # [E, L]


# ============================================================
# Graph encoders
# ============================================================


class BalancedSAGEEncoder(nn.Module):
    """
    Multi-layer GraphSAGE encoder with balanced residual connections.

    Produces node embeddings of shape [N, hid_dim] suitable for downstream
    prediction and edge scoring tasks.

    The forward pass requires a pre-built sparse adjacency matrix and the
    corresponding in-degree vector so that mean aggregation can be applied
    without recomputation.

    Args:
        in_dim:     Dimensionality of raw input node features.
        hid_dim:    Hidden (and output) embedding dimensionality.
        num_layers: Number of SAGE message-passing layers.
        dropout:    Dropout probability applied after each activation.
    """

    def __init__(self, in_dim, hid_dim, num_layers=3, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Input projection and normalisation
        self.input_proj = nn.Linear(in_dim, hid_dim)
        self.input_bn = nn.BatchNorm1d(hid_dim)

        # Per-layer SAGE linear transforms and batch norms
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hid_dim * 2, hid_dim))
            self.bns.append(nn.BatchNorm1d(hid_dim))

        # Global skip connection from raw features to output
        self.skip_h = nn.Linear(in_dim, hid_dim, bias=False)

    def forward(self, x, A_sparse, deg_vec=None):
        """
        Args:
            x:        Raw node feature matrix [N, in_dim].
            A_sparse: Pre-built sparse adjacency [N, N] for spmm.
            deg_vec:  Optional in-degree vector [N, 1] used for mean normalisation.
                      If None, raw spmm output is used (equivalent to sum aggregation).

        Returns:
            Node embeddings [N, hid_dim].
        """
        # Initial projection
        h = self.input_proj(x)
        h = self.input_bn(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        for i in range(self.num_layers):
            # Neighbourhood aggregation via sparse matrix multiply
            neigh = torch.spmm(A_sparse, h)
            if deg_vec is not None:
                neigh = neigh / deg_vec  # Mean normalisation

            # Self + neighbour concatenation -> linear projection
            h = torch.cat([h, neigh], dim=1)
            h = self.layers[i](h)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Residual: add a small fraction of the global skip connection
        h = h + 0.2 * self.skip_h(x)
        return h

# ============================================================
# Main model
# ============================================================


class SelfExplainableGNN(nn.Module):
    """
    Self-Explainable Multi-label GNN (SEMGNN).

    Combines a sufficiency-based edge scorer (EdgeScorer / LabelAwareEdgeScorer)
    with gradient-guided necessity scores to produce both predictions and
    human-interpretable per-label edge importance masks without any external
    explainer model.

    Key components:
      - BalancedSAGEEncoder: produces node embeddings from the full graph.
      - EdgeScorer: base (label-agnostic) edge importance MLP.
      - LabelAwareEdgeScorer (optional): label-conditioned low-rank edge scorer.
      - LabelGCNPredictor (optional): label-GCN augmented prediction head.
      - Gradient guidance: necessity scores derived via autograd.

    Args:
        in_dim:                  Input node feature dimensionality.
        hidden_dim:              Encoder hidden / output dimensionality.
        out_dim:                 Number of labels (output classes).
        num_layers:              Number of SAGE encoder layers.
        dropout:                 Dropout probability throughout the model.
        explain_hidden:          Hidden size for the base EdgeScorer MLP.
        temperature:             Sigmoid temperature for soft edge masks (lower = sharper).
        use_gradient_guidance:   Whether to blend gradient necessity into the final mask.
        grad_weight:             Blend weight for gradient necessity (0 = no gradient).
        use_label_aware:         Whether to instantiate LabelAwareEdgeScorer.
        label_edge_hidden:       Edge encoder hidden size for LabelAwareEdgeScorer.
        label_emb_dim:           Label embedding dimensionality for LabelAwareEdgeScorer.
        label_rank:              Low-rank factorisation rank for LabelAwareEdgeScorer.
        label_use_correlation:   Enable label GCN inside LabelAwareEdgeScorer.
        label_gcn_layers:        GCN layers inside LabelAwareEdgeScorer.
        label_dropout:           Dropout inside LabelAwareEdgeScorer's label GCN.
        use_label_gcn_pred:      Whether to instantiate LabelGCNPredictor.
        label_pred_emb_dim:      Label embedding size for LabelGCNPredictor.
        label_pred_gcn_layers:   GCN layers for LabelGCNPredictor.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        dropout: float = 0.3,
        explain_hidden: int = 64,
        temperature: float = 1.0,
        use_gradient_guidance: bool = True,
        grad_weight: float = 0.3,
        use_label_aware: bool = False,
        label_edge_hidden: int = 64,
        label_emb_dim: int = 32,
        label_rank: int = 16,
        label_use_correlation: bool = False,
        label_gcn_layers: int = 1,
        label_dropout: float = 0.0,
        use_label_gcn_pred: bool = False,
        label_pred_emb_dim: int = 64,
        label_pred_gcn_layers: int = 1,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.temperature = temperature
        self.use_gradient_guidance = use_gradient_guidance
        self.grad_weight = grad_weight
        self.use_label_gcn_pred = bool(use_label_gcn_pred)
        self.pred_label_head: Optional[LabelGCNPredictor] = None
        self.pred_rel_gate_raw: Optional[nn.Parameter] = None

        # Dense normalised label adjacency used for explanation-time correlation refinement
        self.register_buffer("_label_adj_explain_refine", None, persistent=True)

        # Gate controlling how much the label-aware scorer contributes to edge scores
        # Initialised very negative so it starts near zero (sigmoid(-2.5) ≈ 0.076)
        self.la_gate_raw = nn.Parameter(torch.tensor(-2.5))

        # Stage scale: set externally to gradually ramp in the label-aware component
        self.register_buffer("la_stage_scale", torch.tensor(0.0), persistent=True)

        # Graph encoder (BalancedSAGE)
        self.encoder = BalancedSAGEEncoder(
            in_dim=in_dim,
            hid_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Base MLP prediction head (always present)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        # Optional label-GCN prediction head (gated residual on top of base predictor)
        self.pred_label_head = None
        self.pred_rel_gate_raw = None

        if self.use_label_gcn_pred:
            self.pred_label_head = LabelGCNPredictor(
                node_dim=hidden_dim,
                num_labels=out_dim,
                label_emb_dim=label_pred_emb_dim,
                num_gcn_layers=label_pred_gcn_layers,
                proj_hidden=hidden_dim,
                dropout=dropout,
            )
            # Gate initialised at -4 so the GCN head starts near zero contribution
            self.pred_rel_gate_raw = nn.Parameter(torch.tensor(-4.0))

        # Base (label-agnostic) edge importance scorer
        self.edge_scorer = EdgeScorer(hidden_dim, explain_hidden)

        # Optional label-aware edge scorer
        self.use_label_aware = bool(use_label_aware)
        self.label_edge_scorer: Optional[LabelAwareEdgeScorer] = None
        if self.use_label_aware:
            self.label_edge_scorer = LabelAwareEdgeScorer(
                node_dim=hidden_dim,
                num_labels=out_dim,
                edge_hidden=label_edge_hidden,
                label_emb_dim=label_emb_dim,
                rank=label_rank,
                use_label_correlation=label_use_correlation,
                label_gcn_layers=label_gcn_layers,
                dropout=label_dropout,
            )

        # Global skip connection from raw features (bypasses all message passing)
        self.skip = nn.Linear(in_dim, hidden_dim, bias=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def build_sparse_adj(self, edge_index, edge_weight, num_nodes, dtype, device):
        """
        Build a sparse COO adjacency matrix and the corresponding in-degree vector.

        The adjacency is stored as (dst, src) rows for column-major spmm
        compatibility with BalancedSAGEEncoder.

        Args:
            edge_index:  [2, E] edge index tensor (src, dst convention).
            edge_weight: Optional [E] edge weight tensor.
            num_nodes:   Number of nodes N.
            dtype:       Desired tensor dtype.
            device:      Target device.

        Returns:
            Tuple (A_sparse, deg) where deg has shape [N, 1].
        """
        src, dst = edge_index
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=device, dtype=dtype)
        else:
            edge_weight = edge_weight.to(device=device, dtype=dtype)

        # Note: indices stored as (dst, src) for row-based aggregation in spmm
        indices = torch.stack([dst, src], dim=0)
        A = torch.sparse_coo_tensor(indices, edge_weight, (num_nodes, num_nodes), device=device).coalesce()

        # In-degree per node for mean normalisation
        deg = torch.zeros(num_nodes, device=device, dtype=dtype)
        deg.scatter_add_(0, dst, edge_weight)
        deg = deg.clamp(min=1e-12).unsqueeze(-1)
        return A, deg

    def encode(self, x, edge_index, edge_weight=None):
        """
        Encode nodes using the BalancedSAGE encoder.

        Args:
            x:           Node features [N, in_dim].
            edge_index:  Edge indices [2, E].
            edge_weight: Optional edge weights [E] (used as soft masks).

        Returns:
            Node embeddings [N, hidden_dim].
        """
        N = x.size(0)
        A, deg = self.build_sparse_adj(edge_index, edge_weight, N, x.dtype, x.device)
        return self.encoder(x, A_sparse=A, deg_vec=deg)

    def predict(self, node_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute label logits from node embeddings.

        If use_label_gcn_pred is True, the LabelGCNPredictor logits are added
        to the base predictor logits via a learned sigmoid gate.

        Args:
            node_emb: Node embeddings [N, hidden_dim].

        Returns:
            Raw logits [N, out_dim].
        """
        logits = self.predictor(node_emb)  # Base MLP

        if self.pred_label_head is not None:
            logits_rel = self.pred_label_head(node_emb)
            alpha_rel = torch.sigmoid(self.pred_rel_gate_raw)
            logits = logits + alpha_rel * logits_rel

        return logits

    # ------------------------------------------------------------------
    # Edge mask computation
    # ------------------------------------------------------------------

    def compute_edge_mask_soft(self, node_emb, edge_index, *,
                               label_id=None, label_ids=None,
                               return_all_labels=False, chunk_size=200_000):
        """
        Compute soft (continuous) edge importance mask in [0, 1].

        Base scores from EdgeScorer are optionally augmented by
        LabelAwareEdgeScorer delta scores, controlled by a sigmoid gate and an
        external stage scale factor.

        Args:
            node_emb:          Node embeddings [N, hidden_dim].
            edge_index:        Edge indices [2, E].
            label_id:          Query a single label (output [E]).
            label_ids:         Query multiple labels (output [E, K]).
            return_all_labels: Query all L labels (output [E, L]).
            chunk_size:        Chunk size for the all-labels path.

        Returns:
            Soft edge mask tensor (sigmoid applied).
        """
        base_scores = self.edge_scorer(node_emb, edge_index)   # (E,)

        use_label = (label_id is not None) or (label_ids is not None) or bool(return_all_labels)
        if use_label and (self.label_edge_scorer is not None):
            # Optionally detach node_emb to prevent gradients flowing into encoder
            # through the label-aware scorer during training
            la_in = node_emb.detach() if (self.training and getattr(self, "_la_detach", False)) else node_emb
            delta_scores = self.label_edge_scorer(
                la_in, edge_index,
                label_id=label_id,
                label_ids=label_ids,
                return_all_labels=return_all_labels,
                chunk_size=chunk_size,
            )
            alpha = torch.sigmoid(self.la_gate_raw) * float(self.la_stage_scale.item())

            if delta_scores.dim() == 1:
                edge_scores = base_scores + alpha * delta_scores
            else:
                # Broadcast base_scores over the label dimension
                edge_scores = base_scores.unsqueeze(-1) + alpha * delta_scores
        else:
            edge_scores = base_scores

        return torch.sigmoid(edge_scores / self.temperature)

    def compute_aggregated_mask_with_k_label_sampling(
        self,
        node_emb: torch.Tensor,
        edge_index: torch.Tensor,
        num_samples: int = 6,
    ):
        """
        Compute an aggregated edge mask by uniformly sampling K labels.

        Used during training to reduce memory when computing label-aware scores
        for all L labels is prohibitively expensive.  Sampled label scores are
        averaged with uniform weights.

        Args:
            node_emb:    Node embeddings [N, hidden_dim].
            edge_index:  Edge indices [2, E].
            num_samples: Number of labels to sample (clamped to min(num_samples, L)).

        Returns:
            Aggregated soft edge mask [E].
        """
        base_scores = self.edge_scorer(node_emb, edge_index)   # (E,)

        if self.label_edge_scorer is None:
            return self.compute_edge_mask_soft(node_emb, edge_index)

        L = self.out_dim
        num_samples = min(num_samples, L)
        device = node_emb.device

        # Random label subset without replacement
        sampled_labels = torch.randperm(L, device=device)[:num_samples].tolist()

        la_in = node_emb.detach() if (self.training and getattr(self, "_la_detach", False)) else node_emb
        delta_scores = self.label_edge_scorer(
            la_in, edge_index, label_ids=sampled_labels
        )  # [E, num_samples]

        # Uniform average across sampled labels
        w = torch.ones(num_samples, device=device) / float(num_samples)
        delta_agg = (delta_scores * w.view(1, -1)).sum(dim=1)  # [E]
        alpha = torch.sigmoid(self.la_gate_raw) * float(self.la_stage_scale.item())

        mask_aggregated = torch.sigmoid((base_scores + alpha * delta_agg) / self.temperature)
        return mask_aggregated

    # ------------------------------------------------------------------
    # Necessity (gradient-based) scores
    # ------------------------------------------------------------------

    def compute_necessity_scores(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask_soft: torch.Tensor,
        targets: torch.Tensor,
        train_mask: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None,
        label_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Estimate edge necessity via gradient magnitude w.r.t. the edge mask.

        Necessity captures how much removing an edge would hurt the prediction,
        complementing the sufficiency signal from EdgeScorer.  Gradients are
        rank-normalised to [0, 1] for stable combination with soft mask scores.

        Args:
            x:              Node features [N, in_dim].
            edge_index:     Edge indices [2, E].
            edge_mask_soft: Current soft edge mask [E] (used as the differentiable weight).
            targets:        Binary label targets [N, L] (or [N] for a single label).
            train_mask:     Boolean node mask restricting the loss computation.
            pos_weight:     Per-label positive class weights for BCE loss.
            label_id:       If specified, compute necessity only for this label.

        Returns:
            Rank-normalised necessity scores [E] in [0, 1], detached.
        """
        with torch.enable_grad():
            # Clone mask so gradients do not affect the original computation graph
            mask_for_grad = edge_mask_soft.detach().clone().requires_grad_(True)
            node_emb_masked = self.encode(x, edge_index, edge_weight=mask_for_grad)
            logits_masked = self.predict(node_emb_masked)

            # Compute BCE loss (optionally restricted to train nodes / a single label)
            if train_mask is None:
                if label_id is not None:
                    loss_for_grad = F.binary_cross_entropy_with_logits(
                        logits_masked[:, label_id], targets[:, label_id],
                        pos_weight=pos_weight[label_id] if pos_weight is not None else None
                    )
                else:
                    loss_for_grad = F.binary_cross_entropy_with_logits(
                        logits_masked, targets, pos_weight=pos_weight
                    )
            else:
                if label_id is not None:
                    loss_for_grad = F.binary_cross_entropy_with_logits(
                        logits_masked[train_mask, label_id], targets[train_mask, label_id],
                        pos_weight=pos_weight[label_id] if pos_weight is not None else None
                    )
                else:
                    loss_for_grad = F.binary_cross_entropy_with_logits(
                        logits_masked[train_mask], targets[train_mask], pos_weight=pos_weight
                    )

            # Impact = |∂L/∂m_e * m_e|  (gradient × mask value)
            raw_grad = torch.autograd.grad(loss_for_grad, mask_for_grad, create_graph=False)[0]
            impact = (raw_grad * mask_for_grad).abs()

            # Rank normalisation to [0, 1]
            E = impact.numel()
            if E > 1:
                ranks = torch.argsort(torch.argsort(impact))
                grad_scores = ranks.float() / (E - 1)
            else:
                grad_scores = torch.zeros_like(impact)
            return grad_scores.detach()

    def compute_necessity_scores_node_label(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        node_idx: int,
        label_id: int,
        edge_mask_soft: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute node-and-label specific edge necessity scores.

        This is a finer-grained variant of compute_necessity_scores that
        focuses on the loss contribution of a single (node, label) pair.
        The resulting scores indicate which edges are most important for
        the model to assign label `label_id` to node `node_idx`.

        Args:
            X:              Node features [N, in_dim].
            edge_index:     Edge indices [2, E].
            node_idx:       Target node index.
            label_id:       Target label index.
            edge_mask_soft: Pre-computed soft mask [E]; computed internally if None.
            pos_weight:     Per-label positive class weights.

        Returns:
            Normalised necessity scores [E] in [0, 1], detached.
        """
        node_idx = int(node_idx)
        label_id = int(label_id)

        with torch.enable_grad():
            if edge_mask_soft is None:
                node_emb_full = self.encode(X, edge_index)
                edge_mask_soft = self.compute_edge_mask_soft(node_emb_full, edge_index)

            mask_for_grad = edge_mask_soft.detach().clone().requires_grad_(True)
            node_emb_masked = self.encode(X, edge_index, edge_weight=mask_for_grad)
            logits = self.predict(node_emb_masked)

            # Loss: negative log-probability that node_idx belongs to label_id
            if pos_weight is None:
                prob = torch.sigmoid(logits[node_idx, label_id])
                loss = -torch.log(prob + 1e-12)
            else:
                target = torch.ones((), device=logits.device, dtype=logits.dtype)
                loss = F.binary_cross_entropy_with_logits(
                    logits[node_idx, label_id], target,
                    pos_weight=pos_weight[label_id],
                )

            # Edges with large negative gradient (removing them hurts the prediction)
            # have high necessity
            grad = torch.autograd.grad(loss, mask_for_grad, create_graph=False)[0]
            impact = (-grad * mask_for_grad).relu()
            # Min-max normalisation to [0, 1]
            impact = (impact - impact.min()) / (impact.max() - impact.min() + 1e-9)
            return impact.detach()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_explanation: bool = False,
        compute_gradient: bool = False,
        compute_removed: bool = True,
        targets: Optional[torch.Tensor] = None,
        train_mask: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning predictions and (optionally) explanation signals.

        During training with return_explanation=True, the model additionally
        encodes the graph with the soft edge mask applied (masked encoding) and
        optionally computes the 'removed' encoding (1 - mask) for the delete loss.

        During inference with compute_gradient=True, gradient necessity scores
        are blended with the soft mask to produce a combined explanation mask.

        Args:
            x:                  Node features [N, in_dim].
            edge_index:         Edge indices [2, E].
            return_explanation: If True, compute and return edge masks.
            compute_gradient:   If True (and return_explanation=True), compute
                                gradient necessity scores.
            compute_removed:    If True (training only), compute the 'removed'
                                graph encoding for the delete loss term.
            targets:            Binary label targets [N, L] (needed for gradient).
            train_mask:         Boolean node mask [N] (needed for gradient).
            pos_weight:         Per-label positive class weights [L].

        Returns:
            Dict with keys:
                - "logits":         Full-graph predictions [N, L].
                - "node_emb":       Full-graph node embeddings [N, hidden_dim].
                - "logits_masked":  Masked-graph predictions (training only).
                - "node_emb_masked":Masked-graph embeddings (training only).
                - "edge_mask":      Soft / combined edge mask [E].
                - "logits_removed": Removed-graph predictions (training + compute_removed).
                - "grad_scores":    Gradient necessity scores (if compute_gradient).
                - "edge_mask_combined": Blended mask (inference + gradient).
        """
        node_emb_full = self.encode(x, edge_index)
        logits_full = self.predict(node_emb_full)

        result = {
            "logits": logits_full,
            "node_emb": node_emb_full,
        }

        if return_explanation:
            # Choose mask computation strategy
            if self.training and getattr(self, "_use_k_label_sampling", False):
                edge_mask_soft = self.compute_aggregated_mask_with_k_label_sampling(
                    node_emb=node_emb_full,
                    edge_index=edge_index,
                    num_samples=getattr(self, "_num_label_samples", 6),
                )
            else:
                edge_mask_soft = self.compute_edge_mask_soft(node_emb_full, edge_index)

            if self.training:
                # Masked encoding: encode with soft mask applied as edge weights
                node_emb_masked = self.encode(x, edge_index, edge_weight=edge_mask_soft)
                logits_masked = self.predict(node_emb_masked)

                result["logits_masked"] = logits_masked
                result["node_emb_masked"] = node_emb_masked
                result["edge_mask"] = edge_mask_soft

                if compute_removed:
                    # Complement mask: edges that were suppressed become active
                    edge_weight_removed = (1.0 - edge_mask_soft).clamp(0.0, 1.0)
                    node_emb_removed = self.encode(x, edge_index, edge_weight=edge_weight_removed)
                    logits_removed = self.predict(node_emb_removed)
                    result["logits_removed"] = logits_removed

                if compute_gradient and targets is not None and train_mask is not None:
                    grad_scores = self.compute_necessity_scores(
                        x, edge_index, edge_mask_soft,
                        targets, train_mask, pos_weight
                    )
                    result["grad_scores"] = grad_scores
            else:
                # Inference: optionally blend gradient necessity into the final mask
                if self.use_gradient_guidance and compute_gradient:
                    if targets is not None and train_mask is not None:
                        grad_scores = self.compute_necessity_scores(
                            x, edge_index, edge_mask_soft,
                            targets, train_mask, pos_weight
                        )
                        edge_mask_combined = (
                            (1 - self.grad_weight) * edge_mask_soft +
                            self.grad_weight * grad_scores
                        )
                        result["grad_scores"] = grad_scores
                        result["edge_mask_combined"] = edge_mask_combined
                        result["edge_mask"] = edge_mask_combined
                    else:
                        result["edge_mask"] = edge_mask_soft
                else:
                    result["edge_mask"] = edge_mask_soft

        return result

    # ------------------------------------------------------------------
    # Explanation utilities
    # ------------------------------------------------------------------

    @torch.no_grad()
    def refine_label_masks_with_correlation(
        self,
        label_masks: Dict[int, torch.Tensor],
        *,
        beta: float = 0.2,
        topk_neighbors: Optional[int] = None,
        zero_diag: bool = True,
        clamp01: bool = True,
    ) -> Dict[int, torch.Tensor]:
        """
        Post-hoc label correlation refinement of per-label edge masks.

        Performs one step of graph smoothing on the label mask matrix using the
        stored label adjacency:
            S_refined = (1 - beta) * S + beta * S @ A_sub

        where A_sub is the row/column sub-matrix of the full label adjacency
        restricted to the active labels, and S is the [E, K] stacked mask matrix.

        Args:
            label_masks:      Dict mapping label_id -> edge_mask [E].
            beta:             Interpolation weight in [0, 1] (0 = no refinement).
            topk_neighbors:   If set, only keep the top-k strongest label
                              correlations per label (sparsifies A_sub).
            zero_diag:        Zero out the diagonal of A_sub (no self-refinement).
            clamp01:          Clamp refined masks to [0, 1].

        Returns:
            Refined label_masks dict with identical keys and shapes.
        """
        if (self._label_adj_explain_refine is None) or (label_masks is None) or (len(label_masks) <= 1):
            return label_masks

        active_labels = sorted([int(k) for k in label_masks.keys()])
        K = len(active_labels)

        # Stack all active label masks into a [E, K] matrix
        S = torch.stack([label_masks[l] for l in active_labels], dim=1)
        device = S.device
        dtype = S.dtype

        # Extract the sub-adjacency for active labels only
        A_full = self._label_adj_explain_refine.to(device=device, dtype=dtype)
        idx = torch.tensor(active_labels, device=device, dtype=torch.long)
        A = A_full.index_select(0, idx).index_select(1, idx)  # [K, K]

        if zero_diag:
            A = A.clone()
            A.fill_diagonal_(0.0)

        if (topk_neighbors is not None) and (topk_neighbors > 0) and (topk_neighbors < K):
            k = int(topk_neighbors)
            vals, inds = torch.topk(A, k=k, dim=1)
            mask = torch.zeros_like(A, dtype=torch.bool)
            rows = torch.arange(K, device=device).unsqueeze(1).expand_as(inds)
            mask[rows, inds] = True
            A = A.masked_fill(~mask, 0.0)

        # Row-normalise A so each label receives a convex combination of neighbours
        deg = A.sum(dim=1, keepdim=True).clamp(min=1e-12)
        A = A / deg

        beta = float(max(0.0, min(1.0, beta)))
        S_ref = (1.0 - beta) * S + beta * (S @ A)

        if clamp01:
            S_ref = S_ref.clamp(0.0, 1.0)

        out = {}
        for j, l in enumerate(active_labels):
            out[int(l)] = S_ref[:, j]
        return out

    @torch.no_grad()
    def compute_full_graph_edge_mask_per_label(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        label_id: int,
        Y: Optional[torch.Tensor] = None,
        train_mask: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the full-graph edge mask for a single label.

        If gradient guidance is enabled and ground-truth labels are provided,
        the soft mask is blended with gradient necessity scores.

        Args:
            X:          Node features [N, in_dim].
            edge_index: Edge indices [2, E].
            label_id:   Target label index.
            Y:          Binary label targets [N, L] (needed for gradient guidance).
            train_mask: Boolean node mask [N].
            pos_weight: Per-label positive class weights [L].

        Returns:
            Edge mask [E] in [0, 1].
        """
        self.eval()

        node_emb = self.encode(X, edge_index)
        edge_mask_soft = self.compute_edge_mask_soft(
            node_emb, edge_index,
            # Only pass label_id to label_edge_scorer if it is available
            label_id=int(label_id) if self.label_edge_scorer is not None else None,
        )

        if Y is not None and train_mask is not None and self.use_gradient_guidance:
            grad_scores = self.compute_necessity_scores(
                X, edge_index, edge_mask_soft,
                Y, train_mask, pos_weight,
                label_id=label_id,
            )
            return (1 - self.grad_weight) * edge_mask_soft + self.grad_weight * grad_scores
        else:
            return edge_mask_soft

    @torch.no_grad()
    def compute_full_graph_edge_masks_multilabel(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        active_labels: List[int],
        Y: Optional[torch.Tensor] = None,
        train_mask: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None,
        *,
        max_elements: int = 20_000_000,
        use_corr_refine: bool = False,
        corr_refine_beta: float = 0.2,
        corr_refine_topk: Optional[int] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Compute full-graph edge masks for a list of labels simultaneously.

        When the LabelAwareEdgeScorer is available and the total tensor size
        (E * K) is within max_elements, all label scores are computed in a
        single batched call for efficiency.  Otherwise, labels are processed
        individually.

        Args:
            X:               Node features [N, in_dim].
            edge_index:      Edge indices [2, E].
            active_labels:   List of label indices to compute masks for.
            Y:               Binary label targets [N, L].
            train_mask:      Boolean node mask [N].
            pos_weight:      Per-label positive class weights [L].
            max_elements:    Maximum E*K for batched scoring; falls back to
                             per-label loop beyond this threshold.
            use_corr_refine: Apply label correlation refinement after computing masks.
            corr_refine_beta:  Refinement blend weight (see refine_label_masks_with_correlation).
            corr_refine_topk:  Top-k neighbours for sparse refinement.

        Returns:
            Dict mapping each label_id -> edge_mask [E].
        """
        self.eval()
        active_labels = [int(l) for l in active_labels]
        label_masks: Dict[int, torch.Tensor] = {}

        # No label-aware scorer: fall back to per-label computation
        if self.label_edge_scorer is None:
            for label_id in active_labels:
                label_masks[label_id] = self.compute_full_graph_edge_mask_per_label(
                    X, edge_index, label_id, Y, train_mask, pos_weight
                )
            return label_masks

        node_emb = self.encode(X, edge_index)
        E = int(edge_index.size(1))
        K = int(len(active_labels))

        # Fall back to per-label loop if memory budget would be exceeded
        if K <= 1 or (E * K > int(max_elements)):
            for label_id in active_labels:
                label_masks[label_id] = self.compute_full_graph_edge_mask_per_label(
                    X, edge_index, label_id, Y, train_mask, pos_weight
                )
            return label_masks

        # Batched scoring for all active labels in one call -> [E, K]
        masks_soft = self.compute_edge_mask_soft(
            node_emb, edge_index,
            label_ids=active_labels,
        )

        for j, label_id in enumerate(active_labels):
            edge_mask_soft = masks_soft[:, j]

            if Y is not None and train_mask is not None and self.use_gradient_guidance:
                grad_scores = self.compute_necessity_scores(
                    X, edge_index, edge_mask_soft,
                    Y, train_mask, pos_weight,
                    label_id=int(label_id),
                )
                edge_mask = (1 - self.grad_weight) * edge_mask_soft + self.grad_weight * grad_scores
            else:
                edge_mask = edge_mask_soft

            label_masks[int(label_id)] = edge_mask

        # Optional post-hoc label correlation refinement
        if use_corr_refine and (len(label_masks) > 1):
            label_masks = self.refine_label_masks_with_correlation(
                label_masks,
                beta=corr_refine_beta,
                topk_neighbors=corr_refine_topk,
                zero_diag=True,
                clamp01=True,
            )

        return label_masks

    @torch.no_grad()
    def set_label_adj(
        self,
        label_adj,
        normalize=True,
        add_self_loops=True,
        *,
        symmetrize=True,
        clamp_nonneg=True,
        verbose=False,
    ):
        """
        Distribute the label adjacency matrix to all sub-modules that need it.

        This method:
          1. Forwards the adjacency to LabelAwareEdgeScorer (if present and
             use_label_correlation=True).
          2. Forwards it to LabelGCNPredictor (if present).
          3. Stores a row-normalised dense copy for explanation-time correlation
             refinement in _label_adj_explain_refine.

        Args:
            label_adj:       Raw label co-occurrence matrix [L, L].
            normalize:       Apply symmetric normalisation in sub-modules.
            add_self_loops:  Add self-loops in sub-modules.
            symmetrize:      Symmetrise the adjacency.
            clamp_nonneg:    Clamp negative values to zero.
            verbose:         Print diagnostic info.
        """
        if verbose:
            A_dbg = label_adj.coalesce().to_dense() if label_adj.is_sparse else label_adj
            A_dbg = A_dbg.float()
            print(_label_adj_stats(A_dbg, name="Input/label_adj(before_set)"))

        # Propagate to LabelAwareEdgeScorer if label correlation is enabled
        if (
            self.label_edge_scorer is not None
            and getattr(self.label_edge_scorer, "use_label_correlation", False)
        ):
            self.label_edge_scorer.set_label_adj(
                label_adj,
                normalize=normalize,
                add_self_loops=add_self_loops,
                symmetrize=symmetrize,
                clamp_nonneg=clamp_nonneg,
                verbose=verbose,
            )

        # Propagate to LabelGCNPredictor
        if self.pred_label_head is not None:
            self.pred_label_head.set_label_adj(
                label_adj,
                normalize=normalize,
                add_self_loops=add_self_loops,
                symmetrize=symmetrize,
                clamp_nonneg=clamp_nonneg,
                verbose=verbose,
            )

        # Build a row-normalised dense copy for explanation-time refinement
        # (no self-loops, no symmetric normalisation – just row-stochastic)
        A_ref = label_adj.coalesce().to_dense() if label_adj.is_sparse else label_adj
        A_ref = A_ref.float()
        A_ref = 0.5 * (A_ref + A_ref.t())          # Symmetrise
        A_ref = A_ref.clamp(min=0.0)
        A_ref = torch.nan_to_num(A_ref, nan=0.0, posinf=0.0, neginf=0.0)
        deg = A_ref.sum(dim=1).clamp(min=1e-12)
        A_ref = A_ref / deg[:, None]               # Row-normalise

        self._label_adj_explain_refine = A_ref.to(
            device=self.skip.weight.device, dtype=self.skip.weight.dtype
        )

    def explain_node_with_per_label_mask(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        node_idx: int,
        k_hop: int = 3,
        label_ids: Optional[List[int]] = None,
        label_masks: Optional[Dict[int, torch.Tensor]] = None,
        Y: Optional[torch.Tensor] = None,
        train_mask: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None,
        min_edges: int = 12,
        max_frac: float = 0.30,
        cap: int = 200,
        use_node_local_necessity: bool = True,
        alpha: float = 0.6,
    ) -> Dict:
        """
        Generate a per-label node explanation using k-hop subgraph analysis.

        For each predicted (or specified) label, important edges are selected
        by combining:
          - Global full-graph label masks (pre-computed or computed here).
          - Local node-and-label necessity scores computed on the k-hop subgraph.

        The combination weight is controlled by `alpha`:
            group_score = alpha * global_score + (1 - alpha) * local_score

        The top-M edges per label are selected adaptively via compute_adaptive_topM.
        Directed edges are mapped to undirected (u, v) pairs by averaging mask
        scores across both directions.

        Args:
            X:                       Full-graph node features [N, in_dim].
            edge_index:              Full-graph edge indices [2, E].
            node_idx:                Index of the node to explain.
            k_hop:                   Neighbourhood radius for subgraph extraction.
            label_ids:               Labels to explain; if None, predicted labels are used.
            label_masks:             Pre-computed full-graph label masks (avoids recomputation).
            Y:                       Binary label targets [N, L] for gradient guidance.
            train_mask:              Boolean node mask [N].
            pos_weight:              Per-label positive class weights [L].
            min_edges:               Minimum number of edges to select per label.
            max_frac:                Maximum fraction of subgraph edges to select.
            cap:                     Hard cap on the number of selected edges.
            use_node_local_necessity:Whether to blend in local necessity scores.
            alpha:                   Global vs local blend weight (0 = pure local).

        Returns:
            Dict with keys:
                - "node_idx":              Input node index.
                - "predictions":           List of predicted label indices.
                - "probabilities":         Full-graph prediction probabilities [L].
                - "per_label_explanations":Dict[label_id -> {important_edges, ...}].
                - "unified_explanation":   Union of all per-label important edges.
                - "subgraph_nodes":        Nodes in the k-hop subgraph.
                - "subgraph_edge_index":   Edge index of the k-hop subgraph.
                - "E_sub":                 Number of directed edges in the subgraph.
                - "used_per_label_masks":  Always True.
        """
        from torch_geometric.utils import k_hop_subgraph
        from eval.evaluate import compute_adaptive_topM

        self.eval()
        device = X.device
        N = X.size(0)
        node_idx = int(node_idx)

        # Extract k-hop subgraph around the target node
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=k_hop,
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=N,
        )

        node_l = mapping.item()     # Local index of node_idx within the subgraph
        x_sub = X[subset]
        E_sub = sub_edge_index.size(1)

        # Determine which labels to explain
        if label_ids is None:
            res_full = self(X, edge_index, return_explanation=False)
            probs_full = torch.sigmoid(res_full["logits"][node_idx])
            pred_labels = (probs_full > 0.5).nonzero(as_tuple=True)[0].tolist()
            if not pred_labels:
                pred_labels = [int(torch.argmax(probs_full))]
        else:
            pred_labels = [int(l) for l in label_ids]

        res_full = self(X, edge_index, return_explanation=False)
        probs_full = torch.sigmoid(res_full["logits"][node_idx]).detach().cpu().numpy()

        # ---- Build directed-to-undirected edge mapping ----
        # Try to reuse a pre-built global registry; fall back to building on the fly
        from utils.gt_registry import gt_registry
        E_dir = edge_index.size(1)

        if (gt_registry.is_initialized and
                getattr(gt_registry, "edge2id", None) is not None and
                getattr(gt_registry, "edge_index", None) is not None):
            edge2id = gt_registry.edge2id
            E_undir = int(gt_registry.edge_index.shape[0])
            dir2undir = getattr(gt_registry, "dir2undir", None)

            if dir2undir is None:
                ei = edge_index.detach().cpu().numpy()
                dir2undir = np.empty(E_dir, dtype=np.int64)
                for i in range(E_dir):
                    u = int(ei[0, i]); v = int(ei[1, i])
                    a, b = (u, v) if u < v else (v, u)
                    dir2undir[i] = edge2id[(a, b)]
        else:
            # Build edge2id and dir2undir from scratch
            ei = edge_index.detach().cpu().numpy()
            edge2id = {}
            dir2undir = np.empty(E_dir, dtype=np.int64)
            next_id = 0
            for i in range(E_dir):
                u = int(ei[0, i]); v = int(ei[1, i])
                a, b = (u, v) if u < v else (v, u)
                key = (a, b)
                eid = edge2id.get(key, None)
                if eid is None:
                    eid = next_id
                    edge2id[key] = eid
                    next_id += 1
                dir2undir[i] = eid
            E_undir = int(next_id)

        def directed_to_undirected_mean(mask_dir: torch.Tensor) -> torch.Tensor:
            """Average directed edge mask scores into undirected edge scores."""
            d2u = torch.from_numpy(dir2undir).to(mask_dir.device).long()
            s = torch.zeros(E_undir, device=mask_dir.device, dtype=mask_dir.dtype)
            c = torch.zeros(E_undir, device=mask_dir.device, dtype=mask_dir.dtype)
            s.scatter_add_(0, d2u, mask_dir)
            c.scatter_add_(0, d2u, torch.ones_like(mask_dir))
            return s / c.clamp(min=1)

        # Compute full-graph label masks if not provided
        if label_masks is None:
            print(f"Computing per-label full graph masks for {len(pred_labels)} labels...")
            label_masks = self.compute_full_graph_edge_masks_multilabel(
                X, edge_index, pred_labels, Y, train_mask, pos_weight,
                use_corr_refine=True,
                corr_refine_beta=0.2,
                corr_refine_topk=5,
            )

        per_label_explanations = {}
        all_important_edges = set()

        # Encode the subgraph for local necessity computation
        node_emb_sub = self.encode(x_sub, sub_edge_index)

        # Pre-compute subgraph soft mask (label-agnostic) when no label scorer exists
        edge_mask_soft_sub = None
        if self.label_edge_scorer is None:
            edge_mask_soft_sub = self.compute_edge_mask_soft(node_emb_sub, sub_edge_index)

        for label_id in pred_labels:
            if label_id not in label_masks:
                print(f"Warning: label {label_id} not in label_masks, skipping")
                continue

            full_graph_mask_label = label_masks[label_id]

            # Map full-graph directed mask to undirected scores
            if full_graph_mask_label.numel() == E_dir:
                full_undir = directed_to_undirected_mean(full_graph_mask_label)
            elif full_graph_mask_label.numel() == E_undir:
                full_undir = full_graph_mask_label
            else:
                raise ValueError(
                    f"Mask size mismatch: got {full_graph_mask_label.numel()}, "
                    f"E_dir={E_dir}, E_undir={E_undir}"
                )

            # Group directed subgraph edges into undirected (u, v) pairs
            groups = {}
            for i in range(E_sub):
                u_sub = int(sub_edge_index[0, i])
                v_sub = int(sub_edge_index[1, i])
                u_full = int(subset[u_sub])
                v_full = int(subset[v_sub])
                a, b = (u_full, v_full) if u_full < v_full else (v_full, u_full)
                groups.setdefault((a, b), []).append(i)

            G_sub = len(groups)
            if G_sub == 0:
                continue

            group_keys = list(groups.keys())
            a = float(max(0.0, min(1.0, alpha)))

            # Global scores: look up full-graph undirected mask score for each group
            global_scores = torch.tensor(
                [float(full_undir[edge2id[k]]) if k in edge2id else 0.0 for k in group_keys],
                device=device, dtype=torch.float32,
            )

            if use_node_local_necessity:
                # Local necessity scores computed on the subgraph for this (node, label)
                local_nec_dir = self.compute_necessity_scores_node_label(
                    X=x_sub,
                    edge_index=sub_edge_index,
                    node_idx=int(node_l),
                    label_id=int(label_id),
                    edge_mask_soft=(
                        edge_mask_soft_sub
                        if edge_mask_soft_sub is not None
                        else self.compute_edge_mask_soft(
                            node_emb_sub, sub_edge_index, label_id=int(label_id)
                        )
                    ),
                    pos_weight=pos_weight,
                )

                # Average directed necessity scores within each undirected group
                local_scores = torch.tensor(
                    [float(local_nec_dir[groups[k]].mean()) for k in group_keys],
                    device=device, dtype=torch.float32,
                )
                # Blend global sufficiency and local necessity
                group_scores = a * global_scores + (1.0 - a) * local_scores
            else:
                group_scores = global_scores

            # Adaptive top-M edge selection
            top_m = compute_adaptive_topM(G_sub, min_edges, max_frac, cap)
            top_m = min(top_m, G_sub)

            _, top_g_idx = torch.topk(group_scores, top_m)
            selected_edges_undirected = [group_keys[j] for j in top_g_idx.detach().cpu().tolist()]

            per_label_explanations[label_id] = {
                "important_edges": [list(e) for e in selected_edges_undirected],
                "num_edges": int(top_m),
                "sparsity": 1.0 - (top_m / float(G_sub)),
                "G_sub_undirected": int(G_sub),
                "top_m": int(top_m),
            }

            for (u, v) in selected_edges_undirected:
                all_important_edges.add((int(u), int(v)))

        # Union of all per-label explanations
        unified_explanation = {
            'important_edges': list(all_important_edges),
            'num_edges': len(all_important_edges),
        }

        return {
            'node_idx': node_idx,
            'predictions': pred_labels,
            'probabilities': probs_full,
            'per_label_explanations': per_label_explanations,
            'unified_explanation': unified_explanation,
            'subgraph_nodes': subset.cpu().numpy(),
            'subgraph_edge_index': sub_edge_index.cpu().numpy(),
            'E_sub': E_sub,
            'used_per_label_masks': True,
        }


# ============================================================
# Training loss
# ============================================================


class GradientGuidedLoss(nn.Module):
    """
    Multi-term training objective for SelfExplainableGNN with an automatic
    3-stage learning schedule.

    Loss terms:
      1. **loss_pred**          BCE on the full-graph prediction against ground truth.
      2. **loss_teacher**       Confidence-weighted BCE between masked and full logits
                                (teacher distillation: masked graph learns from full graph).
      3. **loss_delete**        Encourages suppressed-edge predictions to drop below
                                the teacher confidence threshold (necessity supervision).
      4. **loss_masked_target** Warm-up BCE on masked predictions against ground truth;
                                decays to zero as training progresses (stage 0 only).
      5. **loss_mask_reg**      Adaptive sparsity regulariser on the soft edge mask.

    Three-stage schedule (driven by EMA of teacher confidence and teacher loss):
      - Stage 0 (warm-up):  small lambda_teacher, no delete loss, large masked_target.
      - Stage 1 (ramp-up):  lambda_teacher grows, masked_target zeroed out.
      - Stage 2 (full):     delete loss introduced, mask_reg at maximum.

    Args:
        lambda_mask_reg:         Weight for the mask regularisation term.
        lambda_teacher:          Maximum weight for the teacher distillation term.
        lambda_delete:           Maximum weight for the delete / necessity term.
        lambda_masked_target:    Initial weight for the masked-graph GT term (stage 0).
        lambda_masked_target_floor: Minimum floor for lambda_masked_target (unused in auto mode).
        teacher_conf:            Confidence threshold for identifying "confident" predictions.
        ema_conf_tau:            EMA decay used for the confidence gate computation.
        delete_margin:           Unused margin parameter (reserved for future use).
        pos_weight:              Per-label positive class weights [L] for BCE.
        use_focal:               Replace standard BCE with focal loss for loss_pred.
        focal_gamma:             Focusing parameter for focal loss.
        sparsity_mode:           "fixed" or "adaptive_degree" for target ratio computation.
        base_target_ratio:       Base sparsity target ratio (fixed mode).
        min_target_ratio:        Minimum per-edge target sparsity (adaptive mode).
        max_target_ratio:        Maximum per-edge target sparsity (adaptive mode).
    """

    def __init__(
        self,
        lambda_mask_reg: float = 1.0,
        lambda_teacher: float = 0.5,
        lambda_delete: float = 0.2,
        lambda_masked_target: float = 0.5,
        lambda_masked_target_floor: float = 0.0,
        teacher_conf: float = 0.6,
        ema_conf_tau: float = 0.35,
        delete_margin: float = 0.15,
        pos_weight: Optional[torch.Tensor] = None,
        use_focal: bool = False,
        focal_gamma: float = 1.0,
        sparsity_mode: str = "adaptive_degree",
        base_target_ratio: float = 0.10,
        min_target_ratio: float = 0.02,
        max_target_ratio: float = 0.30,
    ):
        super().__init__()
        self.lambda_mask_reg = lambda_mask_reg
        self.lambda_teacher = lambda_teacher
        self.lambda_delete = lambda_delete
        self.lambda_masked_target_init = float(lambda_masked_target)
        self.lambda_masked_target = float(lambda_masked_target)
        self.lambda_masked_target_floor = float(lambda_masked_target_floor)
        self.teacher_conf = teacher_conf
        self.ema_conf_tau = float(ema_conf_tau)
        self.delete_margin = float(delete_margin)
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.pos_weight = pos_weight

        self.sparsity_mode = sparsity_mode
        self.base_target_ratio = base_target_ratio
        self.min_target_ratio = min_target_ratio
        self.max_target_ratio = max_target_ratio
        self._is_bidirected_cached = None

        # ---------- Auto 3-stage scheduler state ----------
        # Store maximum values so stage scaling always scales relative to these
        self.lambda_teacher_max = float(self.lambda_teacher)
        self.lambda_delete_max  = float(self.lambda_delete)
        self.lambda_mask_reg_max  = float(self.lambda_mask_reg)

        self._stage = 0          # Current training stage (0, 1, or 2)
        self._epoch_cnt = 0      # Epoch counter

        # EMA trackers for model confidence and teacher loss magnitude
        self._ema_beta = 0.98
        self._ema_conf = None
        self._ema_teacher = None
        self._ema_teacher_init = None  # Stores initial teacher loss for relative improvement

        # Hit counters and patience for stage transitions
        self._hit0 = 0
        self._hit1 = 0
        self._patience = 5

    def _update_ema(self, conf_value: float, teacher_value: float):
        """
        Update exponential moving averages for confidence and teacher loss.

        Args:
            conf_value:    Current batch average prediction confidence in [0, 1].
            teacher_value: Current teacher loss scalar.
        """
        b = self._ema_beta
        if self._ema_conf is None:
            self._ema_conf = conf_value
        else:
            self._ema_conf = b * self._ema_conf + (1.0 - b) * conf_value

        if self._ema_teacher is None:
            self._ema_teacher = teacher_value
            self._ema_teacher_init = max(teacher_value, 1e-8)
        else:
            self._ema_teacher = b * self._ema_teacher + (1.0 - b) * teacher_value

    def step_auto3stage(self):
        """
        Advance the automatic 3-stage scheduler by one epoch.

        Should be called ONCE at the beginning of each training epoch.

        Stage transitions:
          - 0 -> 1: when EMA confidence exceeds 0.25 for `_patience` consecutive
                    epochs, or after 30 epochs unconditionally.
          - 1 -> 2: when EMA confidence exceeds 0.45 for `_patience` consecutive
                    epochs, or after 60 epochs unconditionally.

        Lambda values are set as smooth functions of confidence and teacher
        improvement gates to avoid abrupt transitions.
        """
        self._epoch_cnt += 1
        # If EMA has not been initialised yet, use conservative stage-0 defaults
        if (self._ema_conf is None) or (self._ema_teacher is None) or (self._ema_teacher_init is None):
            self._stage = 0
            self.lambda_masked_target = self.lambda_masked_target_init
            self.lambda_teacher = 0.2 * self.lambda_teacher_max
            self.lambda_delete  = 0.0
            self.lambda_mask_reg  = 0.1 * self.lambda_mask_reg_max
            return

        ema_conf = float(self._ema_conf)
        ema_teacher = float(self._ema_teacher)
        t0 = float(self._ema_teacher_init)

        # Teacher improvement fraction: how much the teacher loss has dropped
        teacher_improve = max(0.0, min(1.0, 1.0 - ema_teacher / (t0 + 1e-12)))

        # Adaptive confidence threshold that tightens as confidence grows
        tau_conf = max(float(self.teacher_conf), min(0.85, 0.55 + 0.5 * (ema_conf - 0.5)))
        conf_gate = max(0.0, min(1.0, (ema_conf - tau_conf) / (1.0 - tau_conf + 1e-12)))

        # Stage transition logic with patience
        if self._stage == 0:
            cond = (ema_conf > 0.25)
            self._hit0 = self._hit0 + 1 if cond else 0
            if self._hit0 >= self._patience or self._epoch_cnt >= 30:
                self._stage = 1
        if self._stage == 1:
            cond = (ema_conf > 0.45)
            self._hit1 = self._hit1 + 1 if cond else 0
            if self._hit1 >= self._patience or self._epoch_cnt >= 60:
                self._stage = 2

        # Set lambda values according to current stage
        if self._stage == 0:
            p = max(conf_gate, teacher_improve)
            self.lambda_masked_target = self.lambda_masked_target_init * (1.0 - p)
            self.lambda_teacher = (0.2 + 0.6 * conf_gate) * self.lambda_teacher_max
            self.lambda_delete  = 0.0
            self.lambda_mask_reg  = (0.05 + 0.15 * teacher_improve) * self.lambda_mask_reg_max
        elif self._stage == 1:
            self.lambda_masked_target = 0.0
            self.lambda_teacher = (0.4 + 0.6 * teacher_improve) * self.lambda_teacher_max
            self.lambda_delete  = 0.0
            self.lambda_mask_reg  = (0.2 + 0.5 * teacher_improve) * self.lambda_mask_reg_max
        else:  # Stage 2: full training with delete loss enabled
            self.lambda_masked_target = 0.0
            self.lambda_teacher = 1.0 * self.lambda_teacher_max
            p_del = max(0.0, min(1.0, (conf_gate - 0.3) / (1.0 - 0.3 + 1e-12)))
            self.lambda_delete = p_del * self.lambda_delete_max
            self.lambda_mask_reg = (0.7 + 0.3 * p_del) * self.lambda_mask_reg_max

    def forward(
        self,
        logits_full: torch.Tensor,
        logits_masked: torch.Tensor,
        edge_mask: torch.Tensor,
        targets: torch.Tensor,
        train_mask: torch.Tensor,
        edge_index: torch.Tensor,
        grad_scores: Optional[torch.Tensor] = None,
        edge_mask_soft: Optional[torch.Tensor] = None,
        logits_removed: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the total training loss and return a detailed metrics dict.

        Args:
            logits_full:    Full-graph logits [N, L].
            logits_masked:  Masked-graph logits [N, L].
            edge_mask:      Soft edge mask [E].
            targets:        Binary label targets [N, L].
            train_mask:     Boolean node mask [N].
            edge_index:     Edge indices [2, E].
            grad_scores:    Optional gradient necessity scores [E] (enables
                            additional diagnostics when provided).
            edge_mask_soft: Alias for edge_mask (unused internally, kept for API
                            compatibility with older training loops).
            logits_removed: Logits from the complement-mask encoding [N, L]
                            (required for the delete loss term).

        Returns:
            Dict containing:
                - "loss_total":  Weighted sum of all loss terms.
                - "loss_pred":   Full-graph BCE loss.
                - "loss_teacher":Teacher distillation loss.
                - "loss_delete": Delete / necessity loss.
                - "loss_masked_target": Warm-up masked GT loss.
                - "loss_mask_reg": Mask sparsity regularisation loss.
                - Mask and training diagnostic scalars.
                - Optional gradient score diagnostics.
        """
        N = logits_full.size(0)
        device = edge_mask.device

        # ========== 1. Full-graph prediction loss (against ground truth) ==========
        if self.use_focal:
            # Focal loss: down-weight easy examples via (1 - pt)^gamma
            bce = F.binary_cross_entropy_with_logits(
                logits_full[train_mask],
                targets[train_mask],
                reduction='none',
                pos_weight=self.pos_weight,
            )
            probs = torch.sigmoid(logits_full[train_mask])
            pt = torch.where(targets[train_mask] == 1, probs, 1 - probs)
            focal_weight = (1 - pt) ** self.focal_gamma
            loss_pred = (focal_weight * bce).mean()
        else:
            loss_pred = F.binary_cross_entropy_with_logits(
                logits_full[train_mask],
                targets[train_mask],
                pos_weight=self.pos_weight,
            )

        # ========== 2. Teacher distillation loss (masked -> full soft labels) ==========
        # Detach full logits so gradients only flow through the masked branch
        p_teacher = torch.sigmoid(logits_full[train_mask]).detach()

        bce_teacher = F.binary_cross_entropy_with_logits(
            logits_masked[train_mask],
            p_teacher,
            reduction="none",
        )
        # Weight by confidence: high-confidence predictions contribute more
        w = (p_teacher - 0.5).abs() * 2.0
        loss_teacher = (bce_teacher * w).mean()

        # Update EMA trackers for the stage scheduler
        with torch.no_grad():
            conf_value = float(((p_teacher - 0.5).abs() * 2.0).mean().item())
            teacher_value = float(loss_teacher.detach().item())
            self._update_ema(conf_value, teacher_value)

        # ========== 3. Delete loss (suppressed-edge predictions should drop) ==========
        loss_delete = torch.tensor(0.0, device=device)
        if (logits_removed is not None) and (self.lambda_delete > 0.0):
            p_removed = torch.sigmoid(logits_removed[train_mask])
            # Only penalise confident positive predictions (necessity signal)
            conf = (p_teacher > self.teacher_conf).float()
            loss_delete = (p_removed * conf).sum() / (conf.sum() + 1e-8)

        # ========== 4. Masked-graph warm-up loss (stage 0 only) ==========
        loss_masked_target = torch.tensor(0.0, device=device)
        if self.lambda_masked_target > 0:
            bce_t = F.binary_cross_entropy_with_logits(
                logits_masked[train_mask],
                targets[train_mask].float(),
                pos_weight=self.pos_weight,
                reduction="none",
            )
            gate = (p_teacher - 0.5).abs() * 2.0  # Confidence gate
            loss_masked_target = (bce_t * gate).mean()

        # ========== 5. Mask sparsity regularisation ==========
        edge_targets = self.compute_adaptive_targets(edge_index, edge_mask, N)
        loss_mask_reg = (
            F.mse_loss(edge_mask, edge_targets)                          # Target sparsity
            + 0.1 * (edge_mask * (1 - edge_mask)).mean()                 # Entropy penalty
            + 0.1 * F.relu(0.5 - edge_mask.max()) ** 2                  # Ensure at least some edges are selected
        )

        mask_max = edge_mask.max()

        # ========== Total loss ==========
        loss_total = (
            loss_pred
            + self.lambda_teacher * loss_teacher
            + self.lambda_delete * loss_delete
            + self.lambda_masked_target * loss_masked_target
            + self.lambda_mask_reg * loss_mask_reg
        )

        # Diagnostic scalars for logging
        mask_mean = edge_mask.mean()
        mask_high = (edge_mask > 0.5).float().mean()
        target_mean = edge_targets.mean()
        target_std = edge_targets.std()

        dev = loss_total.device

        def _nan_if_none(x):
            """Return NaN float if x is None, else float(x)."""
            return float("nan") if x is None else float(x)

        result = {
            "loss_total": loss_total,
            "loss_pred": loss_pred,
            "loss_teacher": loss_teacher,
            "loss_delete": loss_delete,
            "loss_masked_target": loss_masked_target,
            "loss_mask_reg": loss_mask_reg,
            "mask_mean": mask_mean,
            "mask_max": mask_max,
            "mask_sparsity": mask_high,
            "target_mean": target_mean,
            "target_std": target_std,
            # Current (scheduled) lambda values for monitoring
            "lambda_masked_target_current": torch.tensor(float(self.lambda_masked_target), device=dev),
            "stage": torch.tensor(int(self._stage), device=dev),
            "lambda_teacher_current": torch.tensor(float(self.lambda_teacher), device=dev),
            "lambda_delete_current": torch.tensor(float(self.lambda_delete), device=dev),
            "lambda_mask_reg_current": torch.tensor(float(self.lambda_mask_reg), device=dev),
            "ema_conf": torch.tensor(_nan_if_none(self._ema_conf), device=dev),
            "ema_teacher": torch.tensor(_nan_if_none(self._ema_teacher), device=dev),
            # Weighted loss contributions
            "teacher_w": self.lambda_teacher * loss_teacher,
            "delete_w":  self.lambda_delete  * loss_delete,
            "mask_reg_w":  self.lambda_mask_reg  * loss_mask_reg,
        }

        # Optional gradient score diagnostics
        if grad_scores is not None:
            result["grad_mean"] = grad_scores.mean()
            result["grad_max"] = grad_scores.max()
            result["grad_sparsity"] = (grad_scores > 0.5).float().mean()

            # Mask entropy: measures sharpness of the mask distribution
            mask_entropy = -(
                edge_mask * torch.log(edge_mask + 1e-8) +
                (1 - edge_mask) * torch.log(1 - edge_mask + 1e-8)
            ).mean()
            result["mask_entropy"] = mask_entropy
            result["mask_quality"] = 1.0 - (mask_entropy / 0.693)  # Normalised to [0, 1]

            with torch.no_grad():
                # Pearson correlation between soft mask and gradient necessity
                correlation = torch.corrcoef(
                    torch.stack([edge_mask, grad_scores])
                )[0, 1]
                result["soft_grad_corr"] = correlation

        return result

    def compute_adaptive_targets(
        self,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        N: int,
    ) -> torch.Tensor:
        """
        Compute per-edge sparsity target ratios for the mask regularisation loss.

        Two modes are supported:

        - ``"fixed"``: All edges receive the same target ratio (base_target_ratio).
        - ``"adaptive_degree"``: High-degree nodes get lower target ratios (their
          neighbourhoods should be more selective), while low-degree nodes can
          retain a larger fraction of their edges.  The mapping is log-linear in
          degree, clamped to [min_target_ratio, max_target_ratio].

        Args:
            edge_index: Edge indices [2, E].
            edge_mask:  Soft edge mask [E] (used only to infer device/dtype).
            N:          Number of nodes.

        Returns:
            Per-edge target ratio tensor [E] in [min_target_ratio, max_target_ratio].
        """
        device = edge_index.device
        src, dst = edge_index

        if self.sparsity_mode == "fixed":
            return torch.full_like(edge_mask, self.base_target_ratio)

        elif self.sparsity_mode == "adaptive_degree":
            deg_out = torch.bincount(src, minlength=N).float()
            deg_in = torch.bincount(dst, minlength=N).float()

            # Cache the undirectedness check (expensive for large graphs)
            if self._is_bidirected_cached is None:
                try:
                    from torch_geometric.utils import is_undirected
                    self._is_bidirected_cached = bool(is_undirected(edge_index))
                except Exception:
                    self._is_bidirected_cached = False

            degree = deg_out + deg_in
            if self._is_bidirected_cached:
                # For undirected graphs, each edge is counted twice
                degree = 0.5 * degree

            degree = degree.to(device).clamp(min=1.0)
            log_degree = torch.log(degree + 1.0)

            # Normalise log-degree to [0, 1]
            log_min = log_degree.min()
            log_max = log_degree.max()
            if log_max > log_min:
                norm_degree = (log_degree - log_min) / (log_max - log_min)
            else:
                norm_degree = torch.zeros_like(log_degree)

            # Higher degree -> lower target ratio (more selective mask)
            node_targets = (
                self.max_target_ratio
                - (self.max_target_ratio - self.min_target_ratio) * norm_degree
            )
            # Per-edge target is the average of its two endpoint targets
            edge_targets = (node_targets[src] + node_targets[dst]) / 2.0
            return edge_targets

        else:
            raise ValueError(f"Unknown sparsity_mode: {self.sparsity_mode}")