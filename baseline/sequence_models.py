"""
Sequence Baseline Models

Includes:
  - CNN1D: 3-layer Conv1d + BatchNorm + MaxPool + FC
  - BiLSTMClassifier: Linear Embedding → BiLSTM → Attention Pooling → FC
  - create_sequence_model(): factory function

Constraints:
  - No attention / graph / transformer in CNN
  - BiLSTM allows attention pooling (standard configuration)
  - All models maintain session-level classification granularity
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.config import BASELINE_CONFIG


# ═══════════════════════════════════════════════════════════════════
# 1D-CNN Baseline
# ═══════════════════════════════════════════════════════════════════

class CNN1D(nn.Module):
    """
    Three-layer 1D-CNN sequence classifier.

    Architecture:
      Conv1D(k=7, s=2) → BN → ReLU → MaxPool(k=3, s=2)
      Conv1D(k=5, s=1) → BN → ReLU → MaxPool(k=3, s=2)
      Conv1D(k=3, s=1) → BN → ReLU
      AdaptiveAvgPool1d(16)
      Linear(H*16, H) → ReLU → Dropout
      Linear(H, num_classes)
    """

    def __init__(
        self,
        max_seq_len: int = None,
        num_classes: int = None,
        hidden_dim: int = None,
        dropout: float = None,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len or BASELINE_CONFIG["MAX_SEQ_LEN"]
        self.num_classes = num_classes or BASELINE_CONFIG["NUM_CLASSES"]
        self.hidden_dim = hidden_dim or BASELINE_CONFIG["HIDDEN_DIM"]
        self.dropout_rate = dropout if dropout is not None else BASELINE_CONFIG["DROPOUT"]

        self.conv1 = nn.Conv1d(1, self.hidden_dim, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)

        self.conv2 = nn.Conv1d(
            self.hidden_dim, self.hidden_dim * 2, kernel_size=5, stride=1, padding=2
        )
        self.bn2 = nn.BatchNorm1d(self.hidden_dim * 2)

        self.conv3 = nn.Conv1d(
            self.hidden_dim * 2, self.hidden_dim, kernel_size=3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm1d(self.hidden_dim)

        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(output_size=16)

        self.fc1 = nn.Linear(self.hidden_dim * 16, self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, L) → (B, 1, L)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# ═══════════════════════════════════════════════════════════════════
# BiLSTM Baseline
# ═══════════════════════════════════════════════════════════════════

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM sequence classifier.

    Architecture:
      Linear Embedding (1 → H)
      BiLSTM(H, num_layers=2, bidirectional=True)
      Attention Pooling (softmax over time)
      Linear(2H, H) → ReLU → Dropout
      Linear(H, num_classes)
    """

    def __init__(
        self,
        max_seq_len: int = None,
        num_classes: int = None,
        hidden_dim: int = None,
        num_layers: int = 2,
        dropout: float = None,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len or BASELINE_CONFIG["MAX_SEQ_LEN"]
        self.num_classes = num_classes or BASELINE_CONFIG["NUM_CLASSES"]
        self.hidden_dim = hidden_dim or BASELINE_CONFIG["HIDDEN_DIM"]
        self.num_layers = num_layers
        self.dropout_rate = dropout if dropout is not None else BASELINE_CONFIG["DROPOUT"]

        self.embedding = nn.Linear(1, self.hidden_dim)

        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0.0,
        )

        self.attn_fc = nn.Linear(self.hidden_dim * 2, 1)

        self.fc1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)                  # (B, L) → (B, L, 1)
        x = self.embedding(x)                # (B, L, H)

        lstm_out, _ = self.lstm(x)           # (B, L, 2H)

        attn_scores = self.attn_fc(lstm_out)        # (B, L, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        x = torch.sum(lstm_out * attn_weights, dim=1)  # (B, 2H)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# ═══════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════

def create_sequence_model(
    model_type: str,
    max_seq_len: Optional[int] = None,
    num_classes: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """
    Sequence model factory.

    Args:
        model_type: "cnn1d" or "bilstm"
    """
    model_type = model_type.lower().strip()

    if model_type in ("cnn1d", "cnn", "cnn_1d"):
        return CNN1D(
            max_seq_len=max_seq_len, num_classes=num_classes,
            hidden_dim=hidden_dim, **kwargs,
        )
    elif model_type in ("bilstm", "lstm", "bi_lstm"):
        return BiLSTMClassifier(
            max_seq_len=max_seq_len, num_classes=num_classes,
            hidden_dim=hidden_dim, **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown sequence model type: '{model_type}'. "
            f"Supported: 'cnn1d', 'bilstm'"
        )
