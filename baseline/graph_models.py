"""
Graph Baseline Models — GCN and GAT for federated traffic classification.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool

from baseline.config import BASELINE_CONFIG


# ═══════════════════════════════════════════════════════════════════
# Vanilla GCN Baseline
# ═══════════════════════════════════════════════════════════════════

class GCNBaseline(nn.Module):
    """Multi-layer GCN with mean pooling readout."""

    def __init__(
        self,
        node_feature_dim: int = None,
        num_classes: int = None,
        hidden_dim: int = None,
        num_layers: int = None,
        dropout: float = None,
    ):
        super().__init__()
        self.node_feature_dim = node_feature_dim or BASELINE_CONFIG["NODE_FEATURE_DIM"]
        self.num_classes = num_classes or BASELINE_CONFIG["NUM_CLASSES"]
        self.hidden_dim = hidden_dim or BASELINE_CONFIG["HIDDEN_DIM"]
        self.num_layers = num_layers if num_layers is not None else BASELINE_CONFIG["NUM_LAYERS"]
        self.dropout_rate = dropout if dropout is not None else BASELINE_CONFIG["DROPOUT"]

        # Input projection: handle variable input dimensions
        _trg_input_dim = 55  # x_seq(50) + x_stats(5)
        self.input_proj = (
            nn.Linear(_trg_input_dim, self.node_feature_dim)
            if _trg_input_dim != self.node_feature_dim else nn.Identity()
        )

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(self.num_layers):
            out_dim = self.hidden_dim
            in_dim = self.node_feature_dim if i == 0 else self.hidden_dim
            self.convs.append(GCNConv(in_dim, out_dim))
            self.bns.append(nn.BatchNorm1d(out_dim))

        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)

    def _resolve_node_features(self, data):
        """Handle both data.x and data.x_seq+data.x_stats formats."""
        if hasattr(data, "x") and data.x is not None:
            return data.x
        parts = []
        if hasattr(data, "x_seq") and data.x_seq is not None:
            parts.append(data.x_seq)
        if hasattr(data, "x_stats") and data.x_stats is not None:
            parts.append(data.x_stats)
        if not parts:
            raise AttributeError(
                "Graph data has neither 'x' nor 'x_seq'/'x_stats' fields."
            )
        x = torch.cat(parts, dim=-1)
        return self.input_proj(x)

    def forward(self, data):
        x = self._resolve_node_features(data)
        edge_index = data.edge_index
        batch = data.batch

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            if x.size(0) > 1 or not self.training:
                x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = global_mean_pool(x, batch)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ═══════════════════════════════════════════════════════════════════
# Vanilla GAT Baseline
# ═══════════════════════════════════════════════════════════════════

class GATBaseline(nn.Module):
    """Multi-layer GAT with mean+max pooling readout."""

    def __init__(
        self,
        node_feature_dim: int = None,
        num_classes: int = None,
        hidden_dim: int = None,
        num_layers: int = None,
        gat_heads: int = None,
        dropout: float = None,
    ):
        super().__init__()
        self.node_feature_dim = node_feature_dim or BASELINE_CONFIG["NODE_FEATURE_DIM"]
        self.num_classes = num_classes or BASELINE_CONFIG["NUM_CLASSES"]
        self.hidden_dim = hidden_dim or BASELINE_CONFIG["HIDDEN_DIM"]
        self.num_layers = num_layers if num_layers is not None else BASELINE_CONFIG["NUM_LAYERS"]
        self.gat_heads = gat_heads or BASELINE_CONFIG["GAT_HEADS"]
        self.dropout_rate = dropout if dropout is not None else BASELINE_CONFIG["DROPOUT"]

        _trg_input_dim = 55
        self.input_proj = (
            nn.Linear(_trg_input_dim, self.node_feature_dim)
            if _trg_input_dim != self.node_feature_dim else nn.Identity()
        )

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(self.num_layers):
            is_last = (i == self.num_layers - 1)
            if is_last:
                out_dim = self.hidden_dim
                heads = self.gat_heads
                concat = False
            else:
                out_dim = self.hidden_dim // self.gat_heads
                heads = self.gat_heads
                concat = True

            in_dim = self.node_feature_dim if i == 0 else bn_dim
            self.convs.append(
                GATConv(in_dim, out_dim, heads=heads, concat=concat,
                        dropout=self.dropout_rate)
            )
            bn_dim = out_dim * heads if concat else out_dim
            self.bns.append(nn.BatchNorm1d(bn_dim))

        pooled_dim = self.hidden_dim * 2

        self.fc1 = nn.Linear(pooled_dim, self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)

    def _resolve_node_features(self, data):
        if hasattr(data, "x") and data.x is not None:
            return data.x
        parts = []
        if hasattr(data, "x_seq") and data.x_seq is not None:
            parts.append(data.x_seq)
        if hasattr(data, "x_stats") and data.x_stats is not None:
            parts.append(data.x_stats)
        if not parts:
            raise AttributeError(
                "Graph data has neither 'x' nor 'x_seq'/'x_stats' fields."
            )
        x = torch.cat(parts, dim=-1)
        return self.input_proj(x)

    def forward(self, data):
        x = self._resolve_node_features(data)
        edge_index = data.edge_index
        batch = data.batch

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            if x.size(0) > 1 or not self.training:
                x = bn(x)
            x = F.elu(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ═══════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════

def create_graph_model(model_type: str, **kwargs) -> nn.Module:
    model_type = model_type.lower().strip()
    if model_type == "gcn":
        return GCNBaseline(**kwargs)
    elif model_type == "gat":
        return GATBaseline(**kwargs)
    else:
        raise ValueError(f"Unknown graph model type: '{model_type}'")
