"""
Baseline Federated Client

Supports two input modes:
  - MODE_SEQUENCE: sequence models (CNN1D / BiLSTM), standard DataLoader
  - MODE_GNN: graph models (GCN / GAT), PyG DataLoader

Constraints:
  - Fixed FedAvg aggregation
  - Model architecture is the only variable
  - No personalised FL / FedDyn / SCAFFOLD / complex federated optimisers
"""

import copy
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader

from baseline.config import BASELINE_CONFIG

logger = logging.getLogger(__name__)

MODE_SEQUENCE = "sequence"
MODE_GNN = "gnn"


class BaselineFedClient:
    """
    Baseline federated learning client.

    Args:
        client_id: integer client identifier
        model:     nn.Module (CNN1D / BiLSTMClassifier / GCNBaseline / GATBaseline)
        dataset:   PyTorch Dataset (sequence) or list of PyG Data (graph)
        mode:      "sequence" or "gnn"
        config:    configuration dict (defaults to BASELINE_CONFIG)
        device:    torch.device
    """

    def __init__(self, client_id, model, dataset, mode=MODE_SEQUENCE,
                 config=None, device=None):
        self.client_id = client_id
        self.model = model
        self.dataset = dataset
        self.mode = mode
        self.config = config if config is not None else BASELINE_CONFIG
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.local_epochs = self.config.get("LOCAL_EPOCHS", 5)
        self.batch_size = self.config.get("LOCAL_BATCH_SIZE", 32)
        self.lr = self.config.get("LOCAL_LR", 0.001)
        self.weight_decay = self.config.get("LOCAL_WEIGHT_DECAY", 1e-4)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        if self.mode == MODE_GNN:
            try:
                from torch_geometric.loader import DataLoader as PyGDataLoader
                self.train_loader = PyGDataLoader(
                    self.dataset, batch_size=self.batch_size, shuffle=True
                )
            except ImportError:
                raise ImportError(
                    "GNN mode requires torch_geometric. Install: pip install torch-geometric"
                )
        else:
            self.train_loader = TorchDataLoader(
                self.dataset, batch_size=self.batch_size, shuffle=True
            )

    def local_train(self, global_state_dict):
        """
        Client local training (FedAvg).

        Args:
            global_state_dict: global model parameters

        Returns:
            dict with state_dict, num_samples, accuracy
        """
        self.model.load_state_dict(global_state_dict, strict=True)
        self.model.train()

        total_samples = 0
        total_correct = 0

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for batch in self.train_loader:
                # Parse batch (sequence vs graph)
                if self.mode == MODE_GNN:
                    batch = batch.to(self.device)
                    inputs = batch
                    labels = batch.y
                else:
                    if isinstance(batch, (list, tuple)):
                        seq, labels = batch
                        inputs = seq.to(self.device)
                        labels = labels.to(self.device)
                    else:
                        inputs = batch.to(self.device)
                        labels = batch.y.to(self.device) if hasattr(batch, "y") else None

                self.optimizer.zero_grad()
                logits = self.model(inputs)
                loss = nn.CrossEntropyLoss()(logits, labels)
                loss.backward()
                self.optimizer.step()

                bs = labels.size(0)
                epoch_samples += bs
                epoch_loss += loss.item() * bs
                _, pred = torch.max(logits, 1)
                epoch_correct += (pred == labels).sum().item()

            total_samples = epoch_samples
            total_correct = epoch_correct
            logger.info(
                f"Client {self.client_id} Epoch {epoch + 1}: "
                f"Loss={epoch_loss / max(epoch_samples, 1):.4f}, "
                f"Acc={epoch_correct / max(epoch_samples, 1) * 100:.2f}%"
            )

        accuracy = total_correct / max(total_samples, 1)

        return {
            "state_dict": copy.deepcopy(self.model.state_dict()),
            "num_samples": total_samples,
            "accuracy": accuracy,
        }

    def evaluate(self, test_loader):
        """Evaluate the model on a given data loader."""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in test_loader:
                if self.mode == MODE_GNN:
                    batch = batch.to(self.device)
                    inputs = batch
                    labels = batch.y
                else:
                    if isinstance(batch, (list, tuple)):
                        seq, labels = batch
                        inputs = seq.to(self.device)
                        labels = labels.to(self.device)
                    else:
                        inputs = batch.to(self.device)
                        labels = batch.y.to(self.device) if hasattr(batch, "y") else None

                logits = self.model(inputs)
                loss = nn.CrossEntropyLoss()(logits, labels)

                bs = labels.size(0)
                total_samples += bs
                total_loss += loss.item() * bs

                _, pred = torch.max(logits, 1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        if not all_labels:
            return {"accuracy": 0.0, "f1": 0.0, "loss": 0.0, "num_samples": 0}

        acc = accuracy_score(all_labels, all_preds)
        _, _, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )
        avg_loss = total_loss / max(total_samples, 1)

        return {
            "accuracy": acc, "f1": f1, "loss": avg_loss,
            "num_samples": total_samples
        }

    def get_model_params(self):
        return copy.deepcopy(self.model.state_dict())

    def set_model_params(self, state_dict):
        self.model.load_state_dict(state_dict, strict=True)
