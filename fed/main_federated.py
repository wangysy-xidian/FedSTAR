"""
FedSTAR Federated Learning Entry Point
(S)tatistical (T)raffic-prototype guided (A)lignment and (R)efinement

Usage:
    python -m fed.main_federated \
        --data_dir /path/to/data \
        --dataset mydataset \
        --mode gnn \
        --partition label \
        --num_clients 25 \
        --rounds 50 \
        --alpha 0.1
"""

import os
import sys
import argparse
import logging
import copy
import re
import json
import numpy as np
import torch
from typing import Dict

# Allow running as both `python fed/main_federated.py` and `python -m fed.main_federated`
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fed.fed_config import FED_CONFIG
from fed.fed_server import FedServer
from fed.fed_client import FedClient, MODE_GNN, MODE_MULTIMODAL
from fed.data_partition import load_jsonl, partition_label_skew, partition_time_skew
from fed.prototype import compute_client_prototypes

from graph_encoder import TrafficGraphModel
from dataset import TrafficGraphDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [Federated] - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="FedSTAR Federated Training")

    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to directory containing train.jsonl and valid.jsonl")
    parser.add_argument("--train_file", type=str, default="train.jsonl",
                        help="Training file name (inside data_dir)")
    parser.add_argument("--valid_file", type=str, default="valid.jsonl",
                        help="Validation file name (inside data_dir)")
    parser.add_argument("--dataset", type=str, default="traffic",
                        help="Dataset name for logging and checkpoint naming")

    # Mode
    parser.add_argument("--mode", type=str, default="gnn",
                        choices=["gnn", "multimodal"],
                        help="Model mode (gnn uses TrafficGraphModel)")

    # Partitioning
    parser.add_argument("--partition", type=str, default="label",
                        choices=["label", "time", "quantity", "mixed", "feature"],
                        help="Non-IID partition strategy")
    parser.add_argument("--num_clients", type=int, default=FED_CONFIG["NUM_CLIENTS"])
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Dirichlet concentration (lower = more skewed)")

    # Training
    parser.add_argument("--rounds", type=int, default=FED_CONFIG["COMMUNICATION_ROUNDS"])
    parser.add_argument("--local_epochs", type=int, default=FED_CONFIG["LOCAL_EPOCHS"])
    parser.add_argument("--dp", action="store_true", help="Enable differential privacy")

    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="./results",
                        help="Directory for checkpoints and logs")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory for persistent PyG cache (default: <save_dir>/cache)")
    parser.add_argument("--gnn_checkpoint", type=str, default=None,
                        help="Path to pretrained GNN checkpoint")

    return parser.parse_args()


def build_global_model(num_classes: int, config: dict) -> torch.nn.Module:
    return TrafficGraphModel(
        num_classes=num_classes,
        num_kernels=config.get("num_kernels", 64),
        embed_dim=config.get("embed_dim", 128),
        hidden_dim=config.get("hidden_dim", 256),
        gat_heads=config.get("gat_heads", 4)
    )


def main():
    args = parse_args()

    # Build paths
    data_dir = os.path.abspath(args.data_dir)
    train_path = os.path.join(data_dir, args.train_file)
    valid_path = os.path.join(data_dir, args.valid_file)
    save_dir = os.path.abspath(args.save_dir)
    cache_base = os.path.abspath(args.cache_dir) if args.cache_dir else os.path.join(
        save_dir, "cache", f"{args.dataset}_alpha{args.alpha}_{args.num_clients}clients"
    )
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(cache_base, exist_ok=True)

    # Merge config
    config = copy.deepcopy(FED_CONFIG)
    config.update({
        "NUM_CLIENTS": args.num_clients,
        "COMMUNICATION_ROUNDS": args.rounds,
        "LOCAL_EPOCHS": args.local_epochs,
        "USE_DP": args.dp,
        "DATASET": args.dataset,
        "ALPHA": args.alpha,
        "SAVE_DIR": save_dir,
        "LOG_DIR": save_dir,
    })

    # Load and partition data
    logger.info("Loading and partitioning data...")
    raw_data = load_jsonl(train_path)

    if args.partition == "label":
        client_raw_data = partition_label_skew(
            raw_data, num_clients=args.num_clients, alpha=args.alpha
        )
    elif args.partition == "time":
        client_raw_data = partition_time_skew(
            raw_data, num_clients=args.num_clients
        )
    else:
        raise ValueError(f"Unknown partition strategy: {args.partition}")

    # Fit label encoder
    from sklearn.preprocessing import LabelEncoder
    train_data_all = load_jsonl(train_path)
    valid_data_all = load_jsonl(valid_path)
    all_labels = [entry["output"] for entry in train_data_all] + \
                 [entry["output"] for entry in valid_data_all]

    label_encoder = LabelEncoder().fit(all_labels)
    num_classes = len(label_encoder.classes_)

    # Build model
    global_model = build_global_model(num_classes, {
        "num_kernels": 64, "embed_dim": 128, "hidden_dim": 256, "gat_heads": 4
    })
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = global_model.to(device)

    # ── K-Means anchor initialisation ──
    # Collect all packet sizes from the training set, run K-Means to
    # initialise Gaussian anchor centres μ_k.
    logger.info("Initialising Gaussian anchors via K-Means...")
    all_pkt_sizes = []
    for entry in raw_data:
        seq_match = re.search(r"\[Seq:\s*(.*?)\]", entry["input"])
        if seq_match:
            for v in seq_match.group(1).split(","):
                v = v.strip()
                if v.lstrip("-").isdigit():
                    all_pkt_sizes.append(float(v))
    if all_pkt_sizes:
        global_model.node_processor.init_anchors(all_pkt_sizes)
        logger.info(f"  K-Means done on {len(all_pkt_sizes)} packet sizes")
    else:
        logger.warning("  No packet sizes found, anchors stay at random init")

    # ── Create clients with persistent cache ──
    # Each client gets its own subdirectory to avoid PyG cache collisions.
    clients = []
    for client_id in range(args.num_clients):
        client_dir = os.path.join(cache_base, f"client_{client_id}")
        os.makedirs(client_dir, exist_ok=True)

        client_file_path = os.path.join(client_dir, f"data_client_{client_id}.jsonl")

        # Write partition once; PyG caches the parsed graphs thereafter
        if not os.path.exists(client_file_path):
            with open(client_file_path, "w", encoding="utf-8") as f:
                for entry in client_raw_data[client_id]:
                    f.write(json.dumps(entry) + "\n")

        client_dataset = TrafficGraphDataset(
            client_file_path, label_encoder=label_encoder, fit_label=False
        )

        client_model = copy.deepcopy(global_model)
        mode = MODE_GNN if args.mode == "gnn" else MODE_MULTIMODAL
        clients.append(FedClient(
            client_id=client_id,
            model=client_model,
            dataset=client_dataset,
            config=config,
            mode=mode,
            device=device,
            raw_entries=client_raw_data[client_id],
            label_encoder=label_encoder,
        ))

    # Test loader
    from torch_geometric.loader import DataLoader
    test_loader = DataLoader(
        TrafficGraphDataset(valid_path, label_encoder=label_encoder, fit_label=False),
        batch_size=32, shuffle=False
    )

    # Server
    server = FedServer(
        global_model=global_model,
        num_classes=num_classes,
        config=config,
        device=device
    )

    # ── FedSTAR: prototype initialisation (Algorithm 1, lines 2–3) ──
    # All clients compute local prototypes with the initial global model;
    # the server aggregates them into P_c^{(0)}.
    if config["USE_PROTOTYPES"]:
        logger.info("Initialising global prototypes from all clients...")
        init_protos: Dict[int, Dict[int, np.ndarray]] = {}
        init_counts: Dict[int, Dict[int, int]] = {}

        for client_id in range(args.num_clients):
            client = clients[client_id]
            client.model.load_state_dict(global_model.state_dict(), strict=True)
            client.model.eval()
            with torch.no_grad():
                local_protos = compute_client_prototypes(
                    client.model, client.raw_entries, label_encoder, config, device
                )
            if local_protos:
                init_protos[client_id] = local_protos
                counts = {}
                for entry in client.raw_entries:
                    c = int(label_encoder.transform([entry["output"]])[0])
                    counts[c] = counts.get(c, 0) + 1
                init_counts[client_id] = counts

        server.init_prototypes(init_protos, init_counts)
        logger.info("Prototype initialisation complete.")

    # ── Train ──
    logger.info("=" * 60)
    logger.info(
        f"Starting FedSTAR Training [{args.dataset.upper()}] | "
        f"α={args.alpha} | {num_classes} classes"
    )
    logger.info(
        f"  Prototypes: {'ON' if config['USE_PROTOTYPES'] else 'OFF'} | "
        f"μ={config['PROTOTYPE_MOMENTUM']} | D_p={config['PROTOTYPE_DIM']}"
    )
    server.run_federation(
        clients=clients, num_rounds=args.rounds, test_loader=test_loader
    )


if __name__ == "__main__":
    main()
