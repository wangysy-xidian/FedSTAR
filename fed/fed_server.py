"""
FedSTAR Federated Server
(S)tatistical (T)raffic-prototype guided (A)lignment and (R)efinement

Key improvements over vanilla FedAvg:
  - Maintains two parallel global states: model parameters + statistical prototypes
  - Prototype aggregation: per-class weighted average with temporal smoothing
    P_c^{(t)} = μ · P_c^{(t-1)} + (1-μ) · Σ_k w_{k,c} · p_{k,c}^{(t)}
  - Distributes global model + global prototypes each round;
    collects model updates + local prototypes
  - Evaluation supports two-stage prototype-modulated inference
"""

import os
import json
import copy
import logging
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from fed.fed_config import FED_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [FedServer] - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class FedServer:
    """
    Federated server that orchestrates training rounds.

    Responsibilities:
      - Client selection
      - Model distribution
      - Parameter aggregation (FedAvg / FedProx / SCAFFOLD)
      - Prototype aggregation with temporal smoothing
      - Global evaluation and checkpointing
    """

    def __init__(self, global_model: torch.nn.Module, num_classes: int,
                 config: Optional[Dict] = None, device: Optional[torch.device] = None):
        self.config = config if config is not None else FED_CONFIG
        self.global_model = global_model
        self.global_state_dict = copy.deepcopy(global_model.state_dict())
        self.server_control_variate: Optional[Dict[str, torch.Tensor]] = None
        self.round = 0
        self.device = device if device is not None else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── FedSTAR: global prototype state ──
        self.use_prototypes = self.config.get("USE_PROTOTYPES", True)
        self.prototype_dim = self.config.get("PROTOTYPE_DIM", 76)
        self.prototype_momentum = self.config.get("PROTOTYPE_MOMENTUM", 0.8)
        self.num_classes = num_classes
        self.global_prototypes = torch.zeros(
            num_classes, self.prototype_dim, device=self.device
        )

        self.history = {
            "round": [], "global_acc": [], "global_loss": [],
            "client_acc_avg": [], "client_acc_std": [],
        }

        save_dir = self.config.get("SAVE_DIR", "./results")
        os.makedirs(save_dir, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════
    # Client Selection & Model Distribution
    # ═══════════════════════════════════════════════════════════════

    def select_clients(self, num_clients_total: int) -> List[int]:
        strategy = self.config["CLIENT_SELECTION_STRATEGY"]
        fraction = self.config["FRACTION_SAMPLE"]
        num_selected = max(1, int(num_clients_total * fraction))
        if strategy == "random":
            selected = np.random.choice(
                num_clients_total, size=num_selected, replace=False
            ).tolist()
        elif strategy == "weighted":
            weights = self.config["CLIENT_WEIGHTS"]
            if weights is None:
                weights = np.ones(num_clients_total) / num_clients_total
            selected = np.random.choice(
                num_clients_total, size=num_selected, replace=False, p=weights
            ).tolist()
        else:
            raise ValueError(f"Unknown client selection strategy: {strategy}")
        logger.info(f"[Round {self.round + 1}] Selected clients: {selected}")
        return selected

    def distribute_model(self, client_ids: List[int]) -> Dict:
        return {
            client_id: copy.deepcopy(self.global_state_dict)
            for client_id in client_ids
        }

    def distribute_prototypes(self) -> torch.Tensor:
        """Return a copy of the current global prototypes [num_classes, D_p]."""
        return self.global_prototypes.clone()

    # ═══════════════════════════════════════════════════════════════
    # Model Parameter Aggregation
    # ═══════════════════════════════════════════════════════════════

    def aggregate(self, client_updates: Dict[int, Dict[str, torch.Tensor]],
                  client_weights: Dict[int, int]) -> Dict[str, torch.Tensor]:
        """Weighted average of client model parameters."""
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

    def aggregate_fedprox(self, client_updates: Dict, client_weights: Dict) -> Dict:
        return self.aggregate(client_updates, client_weights)

    def aggregate_scaffold(self, client_updates: Dict, client_weights: Dict,
                           client_control_variates: Dict) -> Tuple[Dict, Dict]:
        global_state_dict = self.aggregate(client_updates, client_weights)
        total_samples = sum(client_weights.values())
        if self.server_control_variate is None:
            self.server_control_variate = OrderedDict()
            first_key = list(client_updates.keys())[0]
            for key in client_updates[first_key]:
                self.server_control_variate[key] = torch.zeros_like(
                    client_updates[first_key][key], device="cpu"
                )
        for key in self.server_control_variate:
            self.server_control_variate[key].zero_()
            for client_id in client_updates:
                weight = client_weights[client_id] / total_samples
                self.server_control_variate[key] += (
                    weight * client_control_variates[client_id][key].cpu()
                )
        return global_state_dict, self.server_control_variate

    # ═══════════════════════════════════════════════════════════════
    # FedSTAR: Prototype Management
    # ═══════════════════════════════════════════════════════════════

    def init_prototypes(
        self,
        client_prototypes: Dict[int, Dict[int, np.ndarray]],
        client_class_counts: Dict[int, Dict[int, int]],
    ) -> torch.Tensor:
        """
        Initialise global prototypes (no temporal smoothing; Algorithm 1 lines 2–3).

        All clients compute initial local prototypes with the seed model;
        the server computes a weighted average as P_c^{(0)}.
        """
        aggregated = torch.zeros(
            self.num_classes, self.prototype_dim, device=self.device
        )
        class_total_weights = torch.zeros(self.num_classes, device=self.device)

        for client_id, proto_dict in client_prototypes.items():
            counts = client_class_counts.get(client_id, {})
            for class_idx, proto in proto_dict.items():
                if class_idx >= self.num_classes:
                    continue
                weight = counts.get(class_idx, 0)
                if weight <= 0:
                    continue
                proto_tensor = torch.from_numpy(proto).to(self.device)
                aggregated[class_idx] += weight * proto_tensor
                class_total_weights[class_idx] += weight

        for c in range(self.num_classes):
            if class_total_weights[c] > 0:
                self.global_prototypes[c] = aggregated[c] / class_total_weights[c]

        norms = self.global_prototypes.norm(dim=1)
        logger.info(
            f"[Init] Global prototypes: "
            f"non-zero classes={(class_total_weights > 0).sum().item()}, "
            f"norm min={norms.min().item():.4f} max={norms.max().item():.4f} "
            f"mean={norms.mean().item():.4f}"
        )
        return self.global_prototypes

    def aggregate_prototypes(
        self,
        client_prototypes: Dict[int, Dict[int, np.ndarray]],
        client_class_counts: Dict[int, Dict[int, int]],
    ) -> torch.Tensor:
        """
        Aggregate client prototypes with temporal smoothing.

        P_c^{(t)} = μ · P_c^{(t-1)} + (1-μ) · Σ_k w_{k,c} · p_{k,c}^{(t)}

        Classes without any samples in the current round retain their
        previous prototype values.
        """
        mu = self.prototype_momentum
        old_prototypes = self.global_prototypes.clone()

        aggregated = torch.zeros(
            self.num_classes, self.prototype_dim, device=self.device
        )
        class_total_weights = torch.zeros(self.num_classes, device=self.device)

        for client_id, proto_dict in client_prototypes.items():
            counts = client_class_counts.get(client_id, {})
            for class_idx, proto in proto_dict.items():
                if class_idx >= self.num_classes:
                    continue
                weight = counts.get(class_idx, 0)
                proto_tensor = torch.from_numpy(proto).to(self.device)
                aggregated[class_idx] += weight * proto_tensor
                class_total_weights[class_idx] += weight

        new_prototypes = old_prototypes.clone()
        for c in range(self.num_classes):
            if class_total_weights[c] > 0:
                avg_proto = aggregated[c] / class_total_weights[c]
                new_prototypes[c] = mu * old_prototypes[c] + (1 - mu) * avg_proto

        self.global_prototypes = new_prototypes
        return self.global_prototypes

    # ═══════════════════════════════════════════════════════════════
    # Model Update & Checkpointing
    # ═══════════════════════════════════════════════════════════════

    def update_global_model(self, new_state_dict: Dict[str, torch.Tensor]) -> None:
        self.global_state_dict = copy.deepcopy(new_state_dict)
        self.global_model.load_state_dict(self.global_state_dict)

    def save_checkpoint(self, round_num: int,
                        extra_info: Optional[Dict] = None) -> str:
        checkpoint = {
            "round": round_num,
            "model_state_dict": self.global_state_dict,
            "config": self.config,
            "history": self.history,
        }
        if self.use_prototypes:
            checkpoint["global_prototypes"] = self.global_prototypes.cpu()
        if extra_info is not None:
            checkpoint.update(extra_info)

        dataset_name = self.config.get("DATASET", "traffic")
        alpha_val = self.config.get("ALPHA", 0.1)
        prefix = self.config.get("SAVE_PREFIX", "fedstar")
        save_path = os.path.join(
            self.config["SAVE_DIR"],
            f"{prefix}_{dataset_name}_alpha{alpha_val}_round_{round_num:04d}.pth",
        )
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")
        return save_path

    def log_round(self, round_acc: float, round_loss: float,
                  client_accuracies: List[float]) -> None:
        self.history["round"].append(self.round)
        self.history["global_acc"].append(round_acc)
        self.history["global_loss"].append(round_loss)
        if client_accuracies:
            self.history["client_acc_avg"].append(np.mean(client_accuracies))
            self.history["client_acc_std"].append(np.std(client_accuracies))
        logger.info(
            f"[Round {self.round:04d}] Global Acc: {round_acc:.4f} | "
            f"Loss: {round_loss:.4f} | "
            f"Client Avg Acc: {np.mean(client_accuracies):.4f} "
            f"± {np.std(client_accuracies):.4f}"
        )

    def save_history(self) -> str:
        save_path = os.path.join(self.config["SAVE_DIR"], "training_history.json")
        serializable_history = {}
        for key, values in self.history.items():
            if isinstance(values, list) and len(values) > 0:
                if isinstance(values[0], (torch.Tensor, np.floating)):
                    serializable_history[key] = [float(v) for v in values]
                else:
                    serializable_history[key] = values
            else:
                serializable_history[key] = values
        with open(save_path, "w") as f:
            json.dump(serializable_history, f, indent=2)
        return save_path

    # ═══════════════════════════════════════════════════════════════
    # Federated Training Main Loop
    # ═══════════════════════════════════════════════════════════════

    def run_federation(
        self,
        clients: List,
        num_rounds: Optional[int] = None,
        test_loader: Optional[torch.utils.data.DataLoader] = None,
    ) -> None:
        num_rounds = num_rounds or self.config["COMMUNICATION_ROUNDS"]
        num_clients = len(clients)
        logger.info(
            f"Starting FedSTAR training: {num_clients} clients × {num_rounds} rounds"
        )
        if self.use_prototypes:
            logger.info(
                f"  Prototype dim: {self.prototype_dim}, "
                f"momentum μ: {self.prototype_momentum}"
            )

        for round_idx in range(num_rounds):
            self.round = round_idx + 1

            # 1. Select clients
            selected_client_ids = self.select_clients(num_clients)
            selected_clients = [clients[cid] for cid in selected_client_ids]

            # 2. Distribute global model + global prototypes
            distributed_models = self.distribute_model(selected_client_ids)
            global_prototypes = (
                self.distribute_prototypes() if self.use_prototypes else None
            )

            # 3. Client local training
            client_updates, client_weights, client_accuracies = {}, {}, []
            client_prototypes: Dict[int, Dict[int, np.ndarray]] = {}
            client_class_counts: Dict[int, Dict[int, int]] = {}

            for client_idx, client_id in enumerate(selected_client_ids):
                client = selected_clients[client_idx]
                local_result = client.local_train(
                    global_state_dict=distributed_models[client_id],
                    global_prototypes=global_prototypes,
                    server_control_variate=self.server_control_variate,
                )
                client_updates[client_id] = local_result["state_dict"]
                client_weights[client_id] = local_result["num_samples"]
                client_accuracies.append(local_result.get("accuracy", 0.0))

                if self.use_prototypes and "local_prototypes" in local_result:
                    client_prototypes[client_id] = local_result["local_prototypes"]
                    counts = {}
                    for entry in client.raw_entries:
                        c = int(client.label_encoder.transform(
                            [entry["output"]]
                        )[0])
                        counts[c] = counts.get(c, 0) + 1
                    client_class_counts[client_id] = counts

            # 4. Server aggregation
            # 4a. Model parameter aggregation
            agg_strategy = self.config["AGGREGATION"]
            if agg_strategy == "fedavg":
                new_state_dict = self.aggregate(client_updates, client_weights)
            elif agg_strategy == "fedprox":
                new_state_dict = self.aggregate_fedprox(client_updates, client_weights)
            elif agg_strategy == "scaffold":
                client_cv = {
                    cid: clients[cid].get_control_variate()
                    for cid in selected_client_ids
                }
                new_state_dict, _ = self.aggregate_scaffold(
                    client_updates, client_weights, client_cv
                )
            else:
                raise ValueError(f"Unknown aggregation strategy: {agg_strategy}")

            self.update_global_model(new_state_dict)

            # 4b. Prototype aggregation (FedSTAR)
            if self.use_prototypes and client_prototypes:
                self.aggregate_prototypes(client_prototypes, client_class_counts)

            # 5. Evaluation
            global_acc, global_loss = 0.0, 0.0
            if test_loader is not None:
                global_acc, global_loss = self._evaluate(test_loader)

            if self.round % self.config["LOG_INTERVAL"] == 0:
                self.log_round(global_acc, global_loss, client_accuracies)
            if self.round % self.config["SAVE_INTERVAL"] == 0:
                self.save_checkpoint(self.round)

        # Final save
        self.save_checkpoint(num_rounds, {"final": True})
        self.save_history()
        logger.info("FedSTAR training completed!")

    # ═══════════════════════════════════════════════════════════════
    # Evaluation (supports two-stage inference)
    # ═══════════════════════════════════════════════════════════════

    def _evaluate(self,
                  test_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        Evaluate the global model. Uses two-stage prototype-modulated
        inference when global prototypes are available.

        Stage 1: mean prototype → soft predictions α
        Stage 2: Σ_c α_c · P_c → final logits
        """
        self.global_model.eval()
        device = next(self.global_model.parameters()).device
        correct, total, total_loss = 0, 0, 0.0
        criterion = torch.nn.CrossEntropyLoss()

        prototypes = (
            self.global_prototypes.to(device)
            if self.global_prototypes is not None else None
        )

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                batch_size = batch.y.size(0) if hasattr(batch, 'y') else 1

                if self.use_prototypes and prototypes is not None:
                    # Two-stage inference (FedSTAR)
                    mean_proto = prototypes.mean(dim=0, keepdim=True)
                    mean_proto_batch = mean_proto.expand(batch_size, -1)
                    logits_s1 = self.global_model(
                        batch, per_graph_prototypes=mean_proto_batch
                    )
                    if isinstance(logits_s1, tuple):
                        logits_s1 = logits_s1[0]
                    alpha = F.softmax(logits_s1, dim=-1)

                    weighted_proto = torch.matmul(alpha, prototypes)
                    logits = self.global_model(
                        batch, per_graph_prototypes=weighted_proto
                    )
                    if isinstance(logits, tuple):
                        logits = logits[0]
                else:
                    logits = self.global_model(batch)
                    if isinstance(logits, tuple):
                        logits = logits[0]

                loss = criterion(logits, batch.y)
                total_loss += loss.item() * batch.y.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == batch.y).sum().item()
                total += batch.y.size(0)

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        self.global_model.train()
        return accuracy, avg_loss
