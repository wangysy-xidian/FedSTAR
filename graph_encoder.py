"""
Traffic Graph Model with R-GAT Encoder and Dual-Source FiLM Modulation.

Architecture:
  1. Node Encoder (TrafficNodeModel): packet-level numerical embedding + intra-node fusion
  2. Edge Encoder: edge type embedding + sinusoidal time-difference encoding
  3. Graph Encoder: multi-layer GATv2Conv with edge features
  4. Readout: GlobalAttention pooling → MLP classifier

Supports two-stage prototype-guided inference (FedSTAR).
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GlobalAttention

from node_encoder import TrafficNodeModel


class TimeEncoder(nn.Module):
    """
    Encodes continuous time differences (dt) into vectors via sinusoidal
    positional encoding (Transformer-style), handling spans from ms to seconds.
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-np.log(10000.0) / embedding_dim)
        )
        self.register_buffer('div_term_tensor', self.div_term)

    def forward(self, dt):
        """
        Args:
            dt: [Num_Edges, 1] time differences
        Returns:
            encoding: [Num_Edges, embedding_dim]
        """
        pe = dt * self.div_term_tensor
        encoding = torch.zeros(dt.shape[0], self.embedding_dim, device=dt.device)
        encoding[:, 0::2] = torch.sin(pe)
        encoding[:, 1::2] = torch.cos(pe)
        return encoding


class TrafficGraphModel(nn.Module):
    """
    End-to-end traffic graph classification model.

    Pipeline:
      raw packet sequences + stats → NodeEncoder → R-GAT → GlobalAttention → logits

    FedSTAR extension:
      Accepts global_prototypes or per_graph_prototypes for dual-source FiLM modulation
      during the node encoding stage.
    """

    def __init__(self, num_classes,
                 num_kernels=64, embed_dim=128, hidden_dim=256,
                 gat_heads=4, gat_layers=2):
        super().__init__()

        # ── Node Encoder (packet-level numerical embedding + intra-node fusion) ──
        self.node_processor = TrafficNodeModel(
            num_kernels=num_kernels,
            embedding_dim=embed_dim,
            hidden_dim=hidden_dim,
            stat_dim=5
        )

        # ── Edge Encoder (type + time) ──
        # Edge types: 0=Burst, 1=NextStart, 2=NextEnd
        self.edge_type_emb = nn.Embedding(3, hidden_dim)
        self.time_encoder = TimeEncoder(hidden_dim)
        self.edge_fusion = nn.Linear(hidden_dim, hidden_dim)

        # ── Graph Encoder (R-GAT) ──
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            GATv2Conv(hidden_dim, hidden_dim // gat_heads,
                      heads=gat_heads,
                      edge_dim=hidden_dim,
                      concat=True)
        )
        for _ in range(gat_layers - 1):
            self.gat_layers.append(
                GATv2Conv(hidden_dim, hidden_dim // gat_heads,
                          heads=gat_heads,
                          edge_dim=hidden_dim,
                          concat=True)
            )

        # ── Readout & Classifier ──
        self.readout_gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.pool = GlobalAttention(gate_nn=self.readout_gate)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, data, global_prototypes=None, per_graph_prototypes=None):
        """
        Args:
            data: PyG Batch object with:
                - x_seq:    [Total_Nodes, Seq_Len]   packet size sequences
                - x_stats:  [Total_Nodes, Stat_Dim]  statistical features
                - edge_index: [2, Num_Edges]
                - edge_attr:  [Num_Edges, 2]  (type, time_diff)
                - batch:    [Total_Nodes]  node→graph mapping
                - y:        [Batch_Size]   graph labels (training only)
            global_prototypes: [num_classes, prototype_dim] or None.
                Looked up via data.y during training.
            per_graph_prototypes: [batch_size, prototype_dim] or None.
                Direct per-graph prototypes for two-stage inference.
                Takes precedence over global_prototypes.

        Returns:
            logits: [Batch_Size, num_classes]
        """

        # ── 0. Prototype expansion (FedSTAR) ──
        node_prototype = None
        if per_graph_prototypes is not None:
            graph_prototypes = per_graph_prototypes          # [B, D_p]
            node_prototype = graph_prototypes[data.batch]    # [N, D_p]
        elif global_prototypes is not None and hasattr(data, 'y'):
            graph_prototypes = global_prototypes[data.y]     # [B, D_p]
            node_prototype = graph_prototypes[data.batch]    # [N, D_p]

        # ── 1. Node Encoding ──
        node_embeddings = self.node_processor(
            data.x_seq, data.x_stats,
            global_prototype=node_prototype
        )

        # ── 2. Edge Encoding ──
        edge_types = data.edge_attr[:, 0].long()
        time_diffs = data.edge_attr[:, 1].unsqueeze(1)

        e_type_emb = self.edge_type_emb(edge_types)
        e_time_emb = self.time_encoder(time_diffs)
        edge_embeddings = self.edge_fusion(e_type_emb + e_time_emb)

        # ── 3. Graph Encoding (R-GAT) ──
        x = node_embeddings
        for layer in self.gat_layers:
            x = layer(x, data.edge_index, edge_attr=edge_embeddings)
            x = F.elu(x)

        # ── 4. Readout ──
        graph_emb = self.pool(x, data.batch)

        # ── 5. Classification ──
        logits = self.classifier(graph_emb)

        return logits
