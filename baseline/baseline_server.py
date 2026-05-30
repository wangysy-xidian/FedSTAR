"""
Baseline FedAvg Server

Constraints:
  - Fixed FedAvg aggregation
  - Fixed Dirichlet data partition
  - Model architecture is the only variable
  - No personalised FL / FedDyn / SCAFFOLD / complex federated optimisers
"""

import copy
import os
import json
import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from baseline.config import BASELINE_CONFIG

logger = logging.getLogger(__name__)


class BaselineFedServer:
    """Standard FedAvg server for baseline comparison experiments."""

    def __init__(self, global_model: nn.Module, config: Dict = None):
        self.config = config if config is not None else BASELINE_CONFIG
        self.global_model = global_model
        self.global_state_dict = copy.deepcopy(global_model.state_dict())

        self.round = 0
        self.history: Dict[str, List] = {
            "round": [],
            "global_acc": [],
            "global_loss": [],
            "client_acc_mean": [],
            "client_acc_std": [],
        }

        self.output_dir = self.config.get("OUTPUT_DIR", "experiments/baseline_results")
        os.makedirs(self.output_dir, exist_ok=True)

    def select_clients(self, num_clients_total: int, fraction: float = 1.0) -> List[int]:
        num_selected = max(1, int(num_clients_total * fraction))
        selected = np.random.choice(
            num_clients_total, size=num_selected, replace=False
        ).tolist()
        logger.info(
            f"[Round {self.round+1}] Selected {len(selected)}/{num_clients_total} clients"
        )
        return selected

    def distribute_model(self, client_ids: List[int]) -> Dict[int, Dict]:
        return {cid: copy.deepcopy(self.global_state_dict) for cid in client_ids}

    def aggregate(
        self,
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        client_weights: Dict[int, int],
    ) -> Dict[str, torch.Tensor]:
        """FedAvg: sample-size-weighted average."""
        total_samples = sum(client_weights.values())
        new_state_dict = OrderedDict()

        first_key = list(client_updates.keys())[0]
        for key in client_updates[first_key]:
            tensor = client_updates[first_key][key]
            if torch.is_floating_point(tensor):
                new_state_dict[key] = torch.zeros_like(tensor, device="cpu")
            else:
                new_state_dict[key] = tensor.cpu().clone()

        for client_id, state_dict in client_updates.items():
            weight = client_weights[client_id] / total_samples
            for key in state_dict:
                if torch.is_floating_point(state_dict[key]):
                    new_state_dict[key] += weight * state_dict[key].cpu()

        return new_state_dict

    def update_global_model(self, new_state_dict: Dict[str, torch.Tensor]):
        self.global_state_dict = copy.deepcopy(new_state_dict)
        self.global_model.load_state_dict(self.global_state_dict)

    def evaluate(self, test_loader) -> Tuple[float, float]:
        """Evaluate the global model. Returns (accuracy, loss)."""
        self.global_model.eval()
        device = next(self.global_model.parameters()).device
        correct, total, total_loss = 0, 0, 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in test_loader:
                try:
                    batch = batch.to(device)
                    inputs, labels = batch, batch.y
                except Exception:
                    if isinstance(batch, (list, tuple)):
                        inputs, labels = batch[0].to(device), batch[1].to(device)
                    else:
                        inputs = batch.to(device)
                        labels = getattr(batch, "y", None)
                        if labels is not None:
                            labels = labels.to(device)

                logits = self.global_model(inputs)
                loss = criterion(logits, labels)
                total_loss += loss.item() * len(labels)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        self.global_model.train()
        return accuracy, avg_loss

    def log_round(self, global_acc: float, global_loss: float,
                  client_accuracies: List[float]):
        self.history["round"].append(self.round)
        self.history["global_acc"].append(global_acc)
        self.history["global_loss"].append(global_loss)
        if client_accuracies:
            self.history["client_acc_mean"].append(float(np.mean(client_accuracies)))
            self.history["client_acc_std"].append(float(np.std(client_accuracies)))
        else:
            self.history["client_acc_mean"].append(0.0)
            self.history["client_acc_std"].append(0.0)

        logger.info(
            f"[Round {self.round:03d}] "
            f"Global Acc={global_acc:.4f}  Loss={global_loss:.4f}  "
            f"Client Avg Acc={np.mean(client_accuracies):.4f} "
            f"± {np.std(client_accuracies):.4f}"
        )

    def save_checkpoint(self, ckpt_path: Optional[str] = None):
        path = ckpt_path or os.path.join(
            self.output_dir, f"global_round_{self.round:04d}.pth"
        )
        torch.save(
            {
                "round": self.round,
                "state_dict": self.global_state_dict,
                "config": self.config,
            },
            path,
        )
        logger.info(f"Checkpoint saved → {path}")

    def save_history(self, path: Optional[str] = None):
        path = path or os.path.join(self.output_dir, "training_history.json")
        serializable = {
            k: [float(v) if isinstance(v, (np.floating, torch.Tensor)) else v
                for v in vals]
            for k, vals in self.history.items()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"History saved → {path}")
