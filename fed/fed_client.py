"""
FedSTAR Federated Client
(S)tatistical (T)raffic-prototype guided (A)lignment and (R)efinement

Key improvements over vanilla FedAvg:
  - Dual-source FiLM training: injects global prototypes into the forward pass
  - Local prototype computation: extracts statistical prototypes post-training
  - Two-stage inference: avg prototype → soft predictions → weighted prototype → logits
"""

import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score

from fed.fed_config import FED_CONFIG
from fed.prototype import compute_client_prototypes

logger = logging.getLogger(__name__)

MODE_GNN = "gnn"
MODE_MULTIMODAL = "multimodal"


class FedClient:
    """
    A single federated client holding a local data partition.

    Responsibilities:
      - Local training with dual-source FiLM modulation
      - Post-training local prototype computation
      - Two-stage evaluation with global prototypes
    """

    def __init__(self, client_id, model, dataset, config=None, mode=MODE_GNN,
                 device=None, raw_entries=None, label_encoder=None):
        """
        Args:
            client_id:     integer client identifier
            model:         local copy of the global model
            dataset:       TrafficGraphDataset for this client
            config:        configuration dict (defaults to FED_CONFIG)
            mode:          MODE_GNN or MODE_MULTIMODAL
            device:        torch device
            raw_entries:   list of raw JSONL dicts for prototype computation
            label_encoder: fitted sklearn LabelEncoder
        """
        self.client_id = client_id
        self.model = model
        self.dataset = dataset
        self.config = config if config is not None else FED_CONFIG
        self.mode = mode
        self.device = device if device is not None else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.raw_entries = raw_entries if raw_entries is not None else []
        self.label_encoder = label_encoder

        self.local_epochs = self.config.get("LOCAL_EPOCHS", 5)
        self.batch_size = self.config.get("LOCAL_BATCH_SIZE", 32)
        self.lr = self.config.get("LOCAL_LR", 0.001)
        self.weight_decay = self.config.get("LOCAL_WEIGHT_DECAY", 1e-4)
        self.use_prototypes = self.config.get("USE_PROTOTYPES", True)
        self.num_classes = None  # set during local_train

        self.use_dp = self.config.get("USE_DP", False)
        self.dp_max_grad_norm = self.config.get("DP_MAX_GRAD_NORM", 1.0)
        self.dp_noise_multiplier = self.config.get("DP_NOISE_MULTIPLIER", 1.1)

        self.agg_algo = self.config.get("AGGREGATION", "fedavg").lower()
        self.mu = self.config.get("FEDPROX_MU", 0.01)

        if self.dataset is not None and len(self.dataset) > 0:
            self.train_loader = DataLoader(
                self.dataset, batch_size=self.batch_size, shuffle=True
            )
        else:
            self.train_loader = []

    def _reset_optimizer(self):
        """Reset optimizer before each local_train to avoid state leakage."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def _unpack_batch(self, batch):
        """Unified batch unpacking for different PyG batch formats."""
        if isinstance(batch, (list, tuple)):
            batch = [b.to(self.device) if torch.is_tensor(b) else b for b in batch]
            inputs, labels = batch[0], batch[1]
        elif hasattr(batch, "x") and hasattr(batch, "y"):
            inputs, labels = batch.to(self.device), batch.y
        else:
            inputs = batch.to(self.device)
            labels = batch.y.to(self.device) if hasattr(batch, "y") else None
        return inputs, labels

    def local_train(self, global_state_dict, global_prototypes=None,
                    server_control_variate=None):
        """
        Local training with dual-source FiLM modulation.

        Args:
            global_state_dict:      global model parameters
            global_prototypes:      [num_classes, D_p] global prototype table
            server_control_variate: SCAFFOLD control variate (optional)

        Returns:
            dict with:
              - state_dict:        updated local model parameters
              - num_samples:       number of training samples
              - accuracy:          local training accuracy
              - local_prototypes:  {class_idx: np.ndarray} per-class prototypes
        """
        self.model.load_state_dict(global_state_dict, strict=True)
        self.model.train()
        self._reset_optimizer()

        if global_prototypes is not None:
            self.num_classes = global_prototypes.shape[0]

        if not self.train_loader:
            return {
                "state_dict": copy.deepcopy(self.model.state_dict()),
                "num_samples": 0,
                "accuracy": 0.0,
                "local_prototypes": {},
            }

        global_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }
        total_samples, total_correct = 0, 0

        for epoch in range(self.local_epochs):
            epoch_correct, epoch_samples = 0, 0
            for batch in self.train_loader:
                inputs, labels = self._unpack_batch(batch)

                self.optimizer.zero_grad()

                # Dual-source forward: inject global prototypes
                outputs = self.model(inputs, global_prototypes=global_prototypes)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs

                loss = nn.CrossEntropyLoss()(logits, labels)

                if self.agg_algo == "fedprox":
                    proximal_term = 0.0
                    for name, param in self.model.named_parameters():
                        proximal_term += (
                            (param - global_params[name]) ** 2
                        ).sum()
                    loss += (self.mu / 2.0) * proximal_term

                loss.backward()

                if self.use_dp:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.dp_max_grad_norm
                    )
                    with torch.no_grad():
                        for p in self.model.parameters():
                            if p.grad is not None:
                                p.grad += torch.normal(
                                    0,
                                    self.dp_noise_multiplier * self.dp_max_grad_norm,
                                    size=p.grad.shape, device=self.device
                                )

                self.optimizer.step()

                bs = labels.size(0)
                epoch_samples += bs
                _, pred = torch.max(logits, 1)
                epoch_correct += (pred == labels).sum().item()

            total_samples = epoch_samples
            total_correct = epoch_correct

        accuracy = total_correct / max(total_samples, 1)

        # Post-training: compute local prototypes
        local_prototypes = {}
        if self.use_prototypes and self.raw_entries and self.label_encoder is not None:
            self.model.eval()
            with torch.no_grad():
                local_prototypes = compute_client_prototypes(
                    self.model, self.raw_entries, self.label_encoder,
                    self.config, self.device
                )

        return {
            "state_dict": copy.deepcopy(self.model.state_dict()),
            "num_samples": total_samples,
            "accuracy": accuracy,
            "local_prototypes": local_prototypes,
        }

    def evaluate_two_stage(self, data_loader, global_prototypes):
        """
        Two-stage prototype-modulated inference.

        Stage 1: mean prototype → soft predictions α_c
        Stage 2: Σ_c α_c · P_c → final logits

        Args:
            data_loader:      PyG DataLoader
            global_prototypes: [num_classes, D_p]

        Returns:
            dict: accuracy, predictions, labels
        """
        self.model.eval()
        num_classes = global_prototypes.shape[0]
        device = self.device

        mean_prototype = global_prototypes.mean(dim=0, keepdim=True)  # [1, D_p]

        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = self._unpack_batch(batch)
                batch_size = labels.size(0) if labels is not None else 1

                # Stage 1: mean prototype → soft predictions
                mean_proto_batch = mean_prototype.expand(batch_size, -1).to(device)
                logits_stage1 = self.model(
                    inputs, per_graph_prototypes=mean_proto_batch
                )
                if isinstance(logits_stage1, tuple):
                    logits_stage1 = logits_stage1[0]
                alpha = F.softmax(logits_stage1, dim=-1)

                # Stage 2: weighted prototype → final logits
                weighted_proto = torch.matmul(alpha, global_prototypes)
                logits_stage2 = self.model(
                    inputs, per_graph_prototypes=weighted_proto
                )
                if isinstance(logits_stage2, tuple):
                    logits_stage2 = logits_stage2[0]

                preds = logits_stage2.argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                if labels is not None:
                    all_labels.extend(labels.cpu().tolist())

        accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0
        return {"accuracy": accuracy, "predictions": all_preds, "labels": all_labels}

    def get_control_variate(self):
        """Return zero-initialised control variate for SCAFFOLD."""
        return {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
        }
