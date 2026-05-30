"""Baseline Models for Federated Traffic Classification.

Provides 1D-CNN, BiLSTM, GCN, GAT baseline models with FedAvg framework.
"""

from baseline.sequence_models import CNN1D, BiLSTMClassifier
from baseline.sequence_dataset import PacketSequenceDataset
from baseline.config import BASELINE_CONFIG

# Graph models: delayed import (requires torch-geometric)
try:
    from baseline.graph_models import GCNBaseline, GATBaseline
    from baseline.graph_dataset import TrafficGraphBaselineDataset
except ImportError:
    GCNBaseline = None
    GATBaseline = None
    TrafficGraphBaselineDataset = None

from baseline.baseline_client import BaselineFedClient, MODE_SEQUENCE, MODE_GNN
from baseline.baseline_server import BaselineFedServer

__all__ = [
    'CNN1D',
    'BiLSTMClassifier',
    'GCNBaseline',
    'GATBaseline',
    'PacketSequenceDataset',
    'TrafficGraphBaselineDataset',
    'BASELINE_CONFIG',
    'BaselineFedClient',
    'BaselineFedServer',
    'MODE_SEQUENCE',
    'MODE_GNN',
]
