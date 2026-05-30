"""
Traffic Node Encoder: Packet-Level Numerical Embedding + Intra-Node Fusion.

Two-stage pipeline:
  Stage 1 (TrafficNumericalEmbedding):
    - Decouples magnitude and direction of each packet size
    - Soft-quantizes via learnable Gaussian kernels (robust to numerical drift)
    - Combines kernel-weighted magnitude embedding with sign embedding

  Stage 2 (IntraNodeEncoder):
    - Multi-scale 1D-CNN over the sequence of packet embeddings
    - Dual-source FiLM modulation: local statistics + global prototype
    - Attention pooling over the sequence dimension

References:
  The Gaussian-kernel soft-quantization design is inspired by the
  "statistical prototype" mechanism in FedSTAR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


# ═══════════════════════════════════════════════════════════════════
# Stage 1: Numerical Embedding (soft-quantization via Gaussian kernels)
# ═══════════════════════════════════════════════════════════════════

class TrafficNumericalEmbedding(nn.Module):
    """
    Maps raw packet sizes (e.g., 1448, -32) to dense embeddings.

    Design rationale:
      - Magnitude and sign are decoupled — sign carries uplink/downlink semantics
      - Learnable Gaussian anchors provide soft quantization, robust to
        small numerical perturbations (1448 vs 1450 map similarly)
      - K-Means initialization prevents cold-start collapse
    """

    def __init__(self, num_kernels=64, embedding_dim=128, min_sigma=1.0):
        super().__init__()
        self.num_kernels = num_kernels
        self.embedding_dim = embedding_dim
        self.min_sigma = min_sigma

        # Learnable anchor centers (μ) and widths (σ)
        self.mu = nn.Parameter(torch.randn(num_kernels), requires_grad=True)
        self.log_sigma = nn.Parameter(torch.zeros(num_kernels), requires_grad=True)

        # Magnitude embedding (per-kernel lookup)
        self.kernel_embeddings = nn.Embedding(num_kernels, embedding_dim)
        # Direction embedding: 0=Pad, 1=Downlink(negative), 2=Uplink(positive)
        self.sign_embeddings = nn.Embedding(3, embedding_dim)

        # Fusion projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)

    def init_anchors_with_kmeans(self, raw_values):
        """
        Initialize Gaussian anchor centers via K-Means on the training set's
        absolute packet sizes.  Prevents cold-start issues where anchors
        drift far from the data distribution.

        Args:
            raw_values: list of floats/ints — all packet sizes from training data
        """
        print(f"[*] Initializing anchors with K-Means on {len(raw_values)} samples...")
        abs_values = np.abs(np.array(raw_values)).reshape(-1, 1)

        kmeans = KMeans(n_clusters=self.num_kernels, n_init=10, random_state=42)
        kmeans.fit(abs_values)

        centers = np.sort(kmeans.cluster_centers_.flatten())

        with torch.no_grad():
            self.mu.copy_(torch.from_numpy(centers).float())
            if len(centers) > 1:
                avg_dist = np.mean(centers[1:] - centers[:-1])
                self.log_sigma.fill_(np.log(avg_dist + 1e-5))
        print("[*] Anchor initialization complete.")

    def get_sigma(self):
        return F.softplus(self.log_sigma) + self.min_sigma

    def forward(self, x):
        """
        Args:
            x: [Batch, Seq_Len] raw packet sizes (e.g., 1448, -32, 0=pad)
        Returns:
            [Batch, Seq_Len, embedding_dim]
        """
        # ── A. Decouple magnitude and direction ──
        x_abs = torch.abs(x)
        x_sign = torch.sign(x)  # -1, 0, 1

        sign_indices = torch.zeros_like(x_sign, dtype=torch.long)
        sign_indices[x_sign < 0] = 1   # downlink
        sign_indices[x_sign > 0] = 2   # uplink

        # ── B. Soft-quantization: Gaussian similarity to each anchor ──
        x_expanded = x_abs.unsqueeze(-1)          # [B, S, 1]
        mu_expanded = self.mu.view(1, 1, -1)      # [1, 1, K]
        sigma = self.get_sigma().view(1, 1, -1)

        dist_sq = (x_expanded - mu_expanded) ** 2
        weights = torch.exp(-dist_sq / (2 * sigma ** 2))   # [B, S, K]
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # ── C. Weighted lookup + sign embedding ──
        mag_emb = torch.matmul(weights, self.kernel_embeddings.weight)  # [B, S, D]
        sign_emb = self.sign_embeddings(sign_indices)                   # [B, S, D]

        # ── D. Fuse ──
        return self.output_proj(mag_emb + sign_emb)


# ═══════════════════════════════════════════════════════════════════
# Stage 2: Intra-Node Feature Encoding (CNN + Dual-Source FiLM)
# ═══════════════════════════════════════════════════════════════════

class IntraNodeEncoder(nn.Module):
    """
    Fuses per-packet embeddings and flow-level statistics into a single
    node representation.

    Key components:
      - Multi-scale 1D-CNN captures local burst patterns at 3/5/7 granularities
      - Dual-source FiLM: local statistics (z_v) + global prototype (P_c)
        jointly modulate CNN features via γ, β
      - Attention pooling discards padding positions automatically
    """

    def __init__(self, input_dim=128, hidden_dim=256, stat_dim=5,
                 prototype_dim=76, prototype_mlp_hidden=64):
        super().__init__()

        # ── Multi-scale 1D-CNN ──
        self.conv3 = nn.Conv1d(input_dim, hidden_dim // 3, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(input_dim, hidden_dim // 3, kernel_size=5, padding=2)
        rem = hidden_dim - 2 * (hidden_dim // 3)
        self.conv7 = nn.Conv1d(input_dim, rem, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(hidden_dim)

        # ── Dual-Source FiLM Generator ──
        # Local branch: flow statistics → h_local
        self.mlp_local = nn.Sequential(
            nn.Linear(stat_dim, prototype_mlp_hidden),
            nn.ReLU(),
            nn.Linear(prototype_mlp_hidden, prototype_mlp_hidden),
        )
        # Global branch: prototype → h_global
        self.mlp_global = nn.Sequential(
            nn.Linear(prototype_dim, prototype_mlp_hidden),
            nn.ReLU(),
            nn.Linear(prototype_mlp_hidden, prototype_mlp_hidden),
        )
        # Fusion: [h_local || h_global || h_local - h_global] → γ, β
        self.mlp_fuse = nn.Sequential(
            nn.Linear(prototype_mlp_hidden * 3, prototype_mlp_hidden),
            nn.ReLU(),
            nn.Linear(prototype_mlp_hidden, hidden_dim * 2),
        )

        # ── Attention Pooling ──
        self.attn_score = nn.Linear(hidden_dim, 1)

    def forward(self, seq_emb, stat_features, global_prototype=None):
        """
        Args:
            seq_emb:         [B, Seq_Len, input_dim]  from Stage 1
            stat_features:   [B, stat_dim]            normalized flow statistics
            global_prototype: [B, prototype_dim] or None.
                If None, falls back to single-source FiLM (local stats only).

        Returns:
            node_vector: [B, hidden_dim]
        """
        # ── A. Multi-scale CNN ──
        x = seq_emb.permute(0, 2, 1)           # [B, D, S]
        x3 = F.relu(self.conv3(x))
        x5 = F.relu(self.conv5(x))
        x7 = F.relu(self.conv7(x))
        h_seq = torch.cat([x3, x5, x7], dim=1) # [B, H, S]
        h_seq = self.bn(h_seq)

        # ── B. FiLM modulation ──
        if global_prototype is not None:
            # Dual-source (FedSTAR)
            h_local = self.mlp_local(stat_features)
            h_global = self.mlp_global(global_prototype)
            h_diff = h_local - h_global              # explicit deviation encoding
            h_fuse = torch.cat([h_local, h_global, h_diff], dim=-1)
            film_params = self.mlp_fuse(h_fuse)
        else:
            # Single-source fallback (statistics only)
            h_local = self.mlp_local(stat_features)
            h_global = torch.zeros_like(h_local)
            h_diff = torch.zeros_like(h_local)
            h_fuse = torch.cat([h_local, h_global, h_diff], dim=-1)
            film_params = self.mlp_fuse(h_fuse)

        gamma, beta = torch.chunk(film_params, 2, dim=1)
        gamma = gamma.unsqueeze(2)   # [B, H, 1]
        beta = beta.unsqueeze(2)

        h_modulated = h_seq * (1 + gamma) + beta

        # ── C. Attention pooling ──
        h_modulated = h_modulated.permute(0, 2, 1)    # [B, S, H]
        scores = self.attn_score(h_modulated)          # [B, S, 1]
        weights = F.softmax(scores, dim=1)

        h_node = torch.sum(h_modulated * weights, dim=1)  # [B, H]
        return h_node


# ═══════════════════════════════════════════════════════════════════
# Top-level wrapper
# ═══════════════════════════════════════════════════════════════════

class TrafficNodeModel(nn.Module):
    """
    Convenience wrapper combining Stage 1 (numerical embedding) and
    Stage 2 (intra-node encoder).
    """

    def __init__(self, num_kernels=64, embedding_dim=128, hidden_dim=256, stat_dim=5):
        super().__init__()
        self.embedder = TrafficNumericalEmbedding(num_kernels, embedding_dim)
        self.encoder = IntraNodeEncoder(embedding_dim, hidden_dim, stat_dim)

    def init_anchors(self, raw_values):
        """Expose K-Means anchor initialization to external callers."""
        self.embedder.init_anchors_with_kmeans(raw_values)

    def forward(self, raw_seq, raw_stats, global_prototype=None):
        """
        Args:
            raw_seq:  [B, Seq_Len]   raw packet size sequence
            raw_stats:[B, Stat_Dim]  flow-level statistics
            global_prototype: [B, Prototype_Dim] or None (FedSTAR)
        Returns:
            node_vector: [B, hidden_dim]
        """
        seq_vectors = self.embedder(raw_seq)
        node_vector = self.encoder(seq_vectors, raw_stats,
                                   global_prototype=global_prototype)
        return node_vector
