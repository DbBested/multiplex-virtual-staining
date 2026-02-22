"""MarkerGNN: Graph Attention Network for marker-marker relationship modeling.

Implements a pure PyTorch GATv2-based graph attention network that models
biological relationships between 4 marker nodes (DAPI, Lap2, Marker/Ki67,
Hematoxylin). Operates on pooled transformer features and produces a gated
refinement signal.

No PyTorch Geometric dependency. Pure PyTorch implementation for a 4-node
fully-connected graph.

References:
    - GATv2: "How Attentive are Graph Attention Networks?" (Brody et al., 2021)
    - AdaLN-Zero gating: DiT (Peebles & Xie, 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Marker node ordering (consistent with CHANNEL_NAMES in jit_trainer.py)
MARKER_NODES = ['DAPI', 'Lap2', 'Marker', 'Hematoxylin']
NUM_MARKER_NODES = 4


def build_biological_prior() -> torch.Tensor:
    """Build 4x4 biological prior matrix for marker relationships.

    Node ordering: [DAPI=0, Lap2=1, Marker=2, Hematoxylin=3]
    Values represent prior belief of relationship strength.
    Self-loops are zero (handled by residual connections in GNN layers).
    Matrix is symmetric: relationship strength is bidirectional.

    Returns:
        prior: (4, 4) symmetric tensor with zero diagonal and non-negative
            values encoding biological relationship strengths.
    """
    prior = torch.zeros(NUM_MARKER_NODES, NUM_MARKER_NODES)
    DAPI, LAP2, MARKER, HEMA = 0, 1, 2, 3

    # DAPI <-> Lap2: strong (nuclear DNA <-> nuclear envelope)
    prior[DAPI, LAP2] = prior[LAP2, DAPI] = 0.8
    # DAPI <-> Hematoxylin: strong (both mark nuclei)
    prior[DAPI, HEMA] = prior[HEMA, DAPI] = 0.7
    # Lap2 <-> Hematoxylin: moderate
    prior[LAP2, HEMA] = prior[HEMA, LAP2] = 0.5
    # Marker(Ki67) <-> DAPI: weak (Ki67 is sparse)
    prior[MARKER, DAPI] = prior[DAPI, MARKER] = 0.3
    # Marker(Ki67) <-> Lap2: weak
    prior[MARKER, LAP2] = prior[LAP2, MARKER] = 0.2
    # Marker(Ki67) <-> Hematoxylin: weak
    prior[MARKER, HEMA] = prior[HEMA, MARKER] = 0.2

    return prior


class GATv2Layer(nn.Module):
    """Single GATv2 message-passing layer for small fully-connected graphs.

    Implements GATv2 dynamic attention:
        e_ij = a^T LeakyReLU(W_l h_i + W_r h_j)

    Unlike GAT (v1), this computes attention AFTER the nonlinearity,
    enabling dynamic (query-dependent) attention scores.

    Args:
        in_dim: Input feature dimension per node.
        out_dim: Output feature dimension per node. Must be divisible by n_heads.
        n_heads: Number of attention heads.
        dropout: Dropout rate applied to attention weights.
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert out_dim % n_heads == 0, f"out_dim ({out_dim}) must be divisible by n_heads ({n_heads})"

        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads

        # Separate projections for source and target (GATv2 requirement)
        self.W_l = nn.Linear(in_dim, out_dim, bias=False)   # left / query projection
        self.W_r = nn.Linear(in_dim, out_dim, bias=False)   # right / key projection

        # Per-head attention vector
        self.a = nn.Parameter(torch.empty(n_heads, self.head_dim))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

        # Initialize attention vector with xavier_uniform
        # unsqueeze to create a 2D tensor for xavier_uniform_, then squeeze back
        nn.init.xavier_uniform_(self.a.unsqueeze(-1))

    def forward(self, h: torch.Tensor, edge_bias: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of GATv2 layer.

        Args:
            h: Node features of shape (B, N, D) where N is the number of
                marker nodes (4) and D is the feature dimension.
            edge_bias: Optional edge bias of shape (N, N) encoding biological
                prior weights. Added to attention logits before softmax.

        Returns:
            Updated node features of shape (B, N, out_dim).
        """
        B, N, _ = h.shape
        H, d = self.n_heads, self.head_dim

        # Project to multi-head space: (B, N, H, d)
        g_l = self.W_l(h).view(B, N, H, d)
        g_r = self.W_r(h).view(B, N, H, d)

        # GATv2 pairwise attention: LeakyReLU THEN dot product
        # g_l_i: (B, N, 1, H, d) broadcast with g_r_j: (B, 1, N, H, d)
        pairwise = self.leaky_relu(
            g_l.unsqueeze(2) + g_r.unsqueeze(1)
        )  # (B, N_i, N_j, H, d)

        # Attention scores: dot with per-head attention vector
        e = (pairwise * self.a).sum(dim=-1)  # (B, N, N, H)

        # Add biological prior bias if provided
        if edge_bias is not None:
            # edge_bias: (N, N) -> (1, N, N, 1) for broadcasting
            e = e + edge_bias.unsqueeze(0).unsqueeze(-1)

        # Softmax over source nodes (dim=2 = j dimension)
        alpha = F.softmax(e, dim=2)  # (B, N, N, H)
        alpha = self.dropout(alpha)

        # Aggregate: weighted sum of value features (g_r serves as values)
        # alpha: (B, N_i, N_j, H), g_r: (B, N_j, H, d)
        h_prime = torch.einsum('bijk,bjkd->bikd', alpha,
                               g_r.view(B, N, H, d))  # (B, N, H, d)

        # Concat heads and apply layer norm
        h_prime = h_prime.reshape(B, N, -1)  # (B, N, out_dim)
        h_prime = self.norm(h_prime)

        return h_prime


class MarkerGNN(nn.Module):
    """Graph Attention Network for marker-marker relationship modeling.

    Models biological relationships between 4 marker nodes (DAPI, Lap2,
    Marker/Ki67, Hematoxylin) using GATv2 attention with learnable edge
    weights initialized from biological priors.

    Architecture:
        1. Pool spatial tokens to global representation
        2. Project to 4 marker-specific node features
        3. GATv2 message passing with residual connections
        4. Project refined nodes back to hidden_size
        5. Gated residual broadcast to all tokens

    The gate is initialized to zero so the module acts as identity at
    initialization, ensuring stable integration into pretrained models.

    Args:
        hidden_size: Transformer hidden dimension (default: 768).
        node_dim: Per-node feature dimension (default: 192 = hidden_size // 4).
        n_heads: Number of attention heads in GATv2 layers (default: 4).
        n_layers: Number of GATv2 message-passing layers (default: 2).
        dropout: Dropout rate in GATv2 attention (default: 0.1).
        use_bio_prior: Whether to initialize edge bias from biological priors.
            If True, edge_bias is a learnable nn.Parameter starting from
            build_biological_prior(). If False, edge_bias is a zero buffer.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        node_dim: int = 192,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_bio_prior: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.node_dim = node_dim
        self.n_layers = n_layers

        # Project pooled features to per-marker node features
        self.to_nodes = nn.Sequential(
            nn.Linear(hidden_size, NUM_MARKER_NODES * node_dim),
            nn.SiLU(),
        )

        # GATv2 layers for message passing
        self.gnn_layers = nn.ModuleList([
            GATv2Layer(node_dim, node_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Project refined nodes back to hidden_size
        self.from_nodes = nn.Sequential(
            nn.Linear(NUM_MARKER_NODES * node_dim, hidden_size),
        )

        # Biological prior as learnable bias or zero buffer
        if use_bio_prior:
            self.edge_bias = nn.Parameter(build_biological_prior())
        else:
            self.register_buffer('edge_bias', torch.zeros(NUM_MARKER_NODES, NUM_MARKER_NODES))

        # Gated residual: zero-init for stable integration (AdaLN-Zero pattern)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: pool -> GNN -> gate -> residual.

        Args:
            x: Transformer output tokens of shape (B, num_tokens, hidden_size).

        Returns:
            Refined tokens of shape (B, num_tokens, hidden_size). At
            initialization (gate=0), returns x unchanged (identity).
        """
        # 1. Pool spatial tokens to global representation
        x_pooled = x.mean(dim=1)  # (B, hidden_size)

        # 2. Project to marker nodes
        nodes = self.to_nodes(x_pooled)  # (B, NUM_MARKER_NODES * node_dim)
        nodes = nodes.view(-1, NUM_MARKER_NODES, self.node_dim)  # (B, 4, node_dim)

        # 3. GNN message passing with residual connections
        for layer in self.gnn_layers:
            nodes = nodes + layer(nodes, self.edge_bias)

        # 4. Project back to hidden_size
        nodes_flat = nodes.reshape(-1, NUM_MARKER_NODES * self.node_dim)  # (B, 4*node_dim)
        refinement = self.from_nodes(nodes_flat)  # (B, hidden_size)

        # 5. Gated residual broadcast to all tokens
        # torch.tanh bounds gate between -1 and 1
        return x + torch.tanh(self.gate) * refinement.unsqueeze(1)
