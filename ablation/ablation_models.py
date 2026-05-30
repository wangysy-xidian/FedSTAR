"""
FedSTAR Ablation Study — Model Variant Definitions
===================================================
M0: Base Seq  — 1D-CNN, no graph, no prototype
M1: Graph Only — GAT inference, no soft-quantisation, no prototype FiLM
M2: Node Only  — soft-quantisation + prototype FiLM, no graph structure
M3: FedSTAR    — full framework (TrafficGraphModel)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GlobalAttention, global_mean_pool
import numpy as np

from node_encoder import TrafficNumericalEmbedding, IntraNodeEncoder
from graph_encoder import TimeEncoder, TrafficGraphModel as M3_FedSTAR


# ═══════════════════════════════════════════════════════════════════
# M0: Base Seq — standard multi-scale 1D-CNN (no graph, no prototype)
# ═══════════════════════════════════════════════════════════════════

class M0_BaseSeq(nn.Module):
    """No soft-quantisation, no graph, no prototype. Pure 1D-CNN + mean pool."""

    def __init__(self, num_classes, max_seq_len=256, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.vocab_size = 2000
        self.embed = nn.Embedding(self.vocab_size, 64, padding_idx=0)

        h3 = hidden_dim // 3
        h5 = hidden_dim // 3
        h7 = hidden_dim - h3 - h5
        self.conv3 = nn.Conv1d(64, h3, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(64, h5, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(64, h7, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.attn = nn.Linear(hidden_dim, 1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, data, global_prototypes=None, per_graph_prototypes=None):
        x = data.x_seq.long().clamp(0, self.vocab_size - 1)
        x = self.embed(x)
        x = x.permute(0, 2, 1)
        x3 = F.relu(self.conv3(x))
        x5 = F.relu(self.conv5(x))
        x7 = F.relu(self.conv7(x))
        x = torch.cat([x3, x5, x7], dim=1)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        scores = self.attn(x).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        h = torch.sum(x * weights.unsqueeze(-1), dim=1)

        graph_emb = global_mean_pool(h, data.batch)
        return self.classifier(graph_emb)


# ═══════════════════════════════════════════════════════════════════
# M1: Graph Only — GAT inference, no soft-quantisation, no FiLM
# ═══════════════════════════════════════════════════════════════════

class M1_GraphOnly(nn.Module):
    """
    Retains GAT graph reasoning and edge encoding, but:
      - Numerical embedding: standard Embedding (no soft-quantisation)
      - Node encoding: simple 1D-CNN (no prototype FiLM)
    """

    def __init__(self, num_classes, hidden_dim=256, gat_heads=4, gat_layers=2):
        super().__init__()
        self.vocab_size = 2000
        self.embed = nn.Embedding(self.vocab_size, hidden_dim, padding_idx=0)

        # Simple 1D-CNN sequence encoder (replaces IntraNodeEncoder with FiLM)
        h3 = hidden_dim // 3
        h5 = hidden_dim // 3
        h7 = hidden_dim - h3 - h5
        self.conv3 = nn.Conv1d(hidden_dim, h3, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(hidden_dim, h5, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(hidden_dim, h7, kernel_size=7, padding=3)
        self.seq_bn = nn.BatchNorm1d(hidden_dim)
        self.seq_attn = nn.Linear(hidden_dim, 1)

        # Statistical feature fusion (linear, not FiLM)
        self.stat_proj = nn.Linear(5, hidden_dim)

        # Edge encoding (same as FedSTAR)
        self.edge_type_emb = nn.Embedding(3, hidden_dim)
        self.time_encoder = TimeEncoder(hidden_dim)
        self.edge_fusion = nn.Linear(hidden_dim, hidden_dim)

        # GAT
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            GATv2Conv(hidden_dim, hidden_dim // gat_heads,
                      heads=gat_heads, edge_dim=hidden_dim, concat=True))
        for _ in range(gat_layers - 1):
            self.gat_layers.append(
                GATv2Conv(hidden_dim, hidden_dim // gat_heads,
                          heads=gat_heads, edge_dim=hidden_dim, concat=True))

        # Readout
        self.readout_gate = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.pool = GlobalAttention(gate_nn=self.readout_gate)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(), nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, data, global_prototypes=None, per_graph_prototypes=None):
        # 1. Standard embedding → sequence encoding (no soft-quant, no FiLM)
        x = data.x_seq.long().clamp(0, self.vocab_size - 1)
        x = self.embed(x)
        x = x.permute(0, 2, 1)
        x3 = F.relu(self.conv3(x))
        x5 = F.relu(self.conv5(x))
        x7 = F.relu(self.conv7(x))
        x = torch.cat([x3, x5, x7], dim=1)
        x = self.seq_bn(x)
        x = x.permute(0, 2, 1)
        scores = self.seq_attn(x).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        node_emb = torch.sum(x * weights.unsqueeze(-1), dim=1)

        # Statistical feature fusion (not FiLM)
        if hasattr(data, 'x_stats') and data.x_stats is not None:
            stat_emb = self.stat_proj(data.x_stats)
            node_emb = node_emb + stat_emb

        # 2. Edge encoding
        edge_types = data.edge_attr[:, 0].long()
        time_diffs = data.edge_attr[:, 1].unsqueeze(1)
        e_type_emb = self.edge_type_emb(edge_types)
        e_time_emb = self.time_encoder(time_diffs)
        edge_emb = self.edge_fusion(e_type_emb + e_time_emb)

        # 3. GAT
        h = node_emb
        for layer in self.gat_layers:
            h = layer(h, data.edge_index, edge_attr=edge_emb)
            h = F.elu(h)

        # 4. Readout + Classify
        graph_emb = self.pool(h, data.batch)
        return self.classifier(graph_emb)


# ═══════════════════════════════════════════════════════════════════
# M2: Node Only — soft-quantisation + prototype FiLM, no graph
# ═══════════════════════════════════════════════════════════════════

class M2_NodeOnly(nn.Module):
    """
    Retains TrafficNumericalEmbedding (soft-quantisation) and
    IntraNodeEncoder (dual-source FiLM), but removes GAT graph reasoning.
    Direct mean pool → classify.
    """

    def __init__(self, num_classes, num_kernels=64, embed_dim=128, hidden_dim=256,
                 prototype_dim=76):
        super().__init__()
        self.num_kernels = num_kernels
        self.node_processor = TrafficNumericalEmbedding(num_kernels, embed_dim)
        self.node_encoder = IntraNodeEncoder(
            input_dim=embed_dim, hidden_dim=hidden_dim, stat_dim=5,
            prototype_dim=prototype_dim,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(), nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def init_anchors(self, raw_values):
        self.node_processor.init_anchors_with_kmeans(raw_values)

    def forward(self, data, global_prototypes=None, per_graph_prototypes=None):
        # Prototype expansion
        node_proto = None
        if per_graph_prototypes is not None:
            node_proto = per_graph_prototypes[data.batch]
        elif global_prototypes is not None and hasattr(data, 'y'):
            graph_protos = global_prototypes[data.y]
            node_proto = graph_protos[data.batch]

        # Soft-quantisation
        seq_emb = self.node_processor(data.x_seq)

        # Node encoding (dual-source FiLM)
        stat = data.x_stats if hasattr(data, 'x_stats') else torch.zeros(
            data.x_seq.size(0), 5, device=data.x_seq.device)
        h = self.node_encoder(seq_emb, stat, global_prototype=node_proto)

        # No graph — direct global pooling
        graph_emb = global_mean_pool(h, data.batch)
        return self.classifier(graph_emb)
