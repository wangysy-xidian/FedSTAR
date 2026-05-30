"""
Baseline Federated Training Entry Point

Supports multiple model architectures:
  Sequence: cnn1d, bilstm
  Graph:    gcn, gat

Usage:
    python -m baseline.baseline_trainer \
        --model cnn1d --data_dir /path/to/data --alpha 0.5 --rounds 100
"""

import os
import sys
import copy
import argparse
import logging
import json
import torch
import numpy as np

# Allow running as both script and module
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.config import BASELINE_CONFIG
from baseline.sequence_models import create_sequence_model
from baseline.sequence_dataset import PacketSequenceDataset
from baseline.baseline_client import BaselineFedClient, MODE_SEQUENCE, MODE_GNN
from baseline.baseline_server import BaselineFedServer
from fed.data_partition import load_jsonl, partition_label_skew

try:
    from baseline.graph_models import create_graph_model
    from baseline.graph_dataset import TrafficGraphBaselineDataset
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [Baseline] - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Baseline Federated Training")
    p.add_argument("--model", type=str, required=True,
                   choices=["cnn1d", "bilstm", "gcn", "gat"])
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory containing train.jsonl and valid.jsonl")
    p.add_argument("--train_file", type=str, default="train.jsonl")
    p.add_argument("--valid_file", type=str, default="valid.jsonl")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--num_clients", type=int, default=BASELINE_CONFIG["NUM_CLIENTS"])
    p.add_argument("--rounds", type=int, default=BASELINE_CONFIG["COMMUNICATION_ROUNDS"])
    p.add_argument("--local_epochs", type=int, default=BASELINE_CONFIG["LOCAL_EPOCHS"])
    p.add_argument("--batch_size", type=int, default=BASELINE_CONFIG["LOCAL_BATCH_SIZE"])
    p.add_argument("--lr", type=float, default=BASELINE_CONFIG["LOCAL_LR"])
    p.add_argument("--device", type=str, default=BASELINE_CONFIG["DEVICE"])
    p.add_argument("--dataset", type=str, default="traffic")
    p.add_argument("--output_dir", type=str, default="experiments/baseline_results")
    return p.parse_args()


def _make_config(args) -> dict:
    cfg = copy.deepcopy(BASELINE_CONFIG)
    cfg.update({
        "NUM_CLIENTS": args.num_clients,
        "COMMUNICATION_ROUNDS": args.rounds,
        "LOCAL_EPOCHS": args.local_epochs,
        "LOCAL_BATCH_SIZE": args.batch_size,
        "LOCAL_LR": args.lr,
        "DIRICHLET_ALPHA": args.alpha,
        "DEVICE": args.device,
        "DATASET": args.dataset,
        "OUTPUT_DIR": args.output_dir,
    })
    return cfg


def build_clients(args, raw_data, label_map, mode):
    """Build client list from partitioned data."""
    client_raw = partition_label_skew(
        raw_data, num_clients=args.num_clients, alpha=args.alpha, seed=42
    )
    clients = []
    for cid in range(args.num_clients):
        if mode == MODE_SEQUENCE:
            client_ds = PacketSequenceDataset()
            client_ds.load_from_partition(client_raw[cid], label_map=dict(label_map))
            model = create_sequence_model(
                args.model,
                max_seq_len=BASELINE_CONFIG["MAX_SEQ_LEN"],
                num_classes=len(label_map),
                hidden_dim=BASELINE_CONFIG["HIDDEN_DIM"],
                dropout=BASELINE_CONFIG["DROPOUT"],
            )
        else:
            client_ds = TrafficGraphBaselineDataset(
                partition_data=client_raw[cid],
                label_map=dict(label_map),
                node_feature_dim=BASELINE_CONFIG["NODE_FEATURE_DIM"],
            )
            model = create_graph_model(
                args.model,
                node_feature_dim=BASELINE_CONFIG["NODE_FEATURE_DIM"],
                num_classes=len(label_map),
                hidden_dim=BASELINE_CONFIG["HIDDEN_DIM"],
                num_layers=BASELINE_CONFIG["NUM_LAYERS"],
                dropout=BASELINE_CONFIG["DROPOUT"],
            )
        client = BaselineFedClient(
            client_id=cid, model=model, dataset=client_ds,
            mode=mode, config=_make_config(args)
        )
        clients.append(client)
    return clients, client_raw


def build_valid_loader(args, label_map, mode):
    """Build validation data loader."""
    valid_path = os.path.join(args.data_dir, args.valid_file)
    if mode == MODE_SEQUENCE:
        ds = PacketSequenceDataset()
        ds.load_from_partition(load_jsonl(valid_path), label_map=dict(label_map))
        from torch.utils.data import DataLoader as TorchDataLoader
        return TorchDataLoader(ds, batch_size=args.batch_size, shuffle=False)
    else:
        ds = TrafficGraphBaselineDataset(
            jsonl_path=valid_path,
            label_map=dict(label_map),
            fit_label_map=False,
            node_feature_dim=BASELINE_CONFIG["NODE_FEATURE_DIM"],
        )
        from torch_geometric.loader import DataLoader as PyGLoader
        return PyGLoader(ds, batch_size=args.batch_size, shuffle=False)


def main():
    args = parse_args()
    mode = MODE_SEQUENCE if args.model in ("cnn1d", "bilstm") else MODE_GNN
    if mode == MODE_GNN and not HAS_PYG:
        logger.error("PyG not available. Install: pip install torch-geometric")
        sys.exit(1)

    data_dir = os.path.abspath(args.data_dir)
    train_path = os.path.join(data_dir, args.train_file)

    jsonl_data = load_jsonl(train_path)
    all_labels = sorted(set(e["output"] for e in jsonl_data))
    label_map = {lbl: i for i, lbl in enumerate(all_labels)}
    num_classes = len(label_map)

    # Build global model
    if mode == MODE_SEQUENCE:
        global_model = create_sequence_model(
            args.model,
            max_seq_len=BASELINE_CONFIG["MAX_SEQ_LEN"],
            num_classes=num_classes,
            hidden_dim=BASELINE_CONFIG["HIDDEN_DIM"],
            dropout=BASELINE_CONFIG["DROPOUT"],
        )
    else:
        global_model = create_graph_model(
            args.model,
            node_feature_dim=BASELINE_CONFIG["NODE_FEATURE_DIM"],
            num_classes=num_classes,
            hidden_dim=BASELINE_CONFIG["HIDDEN_DIM"],
            num_layers=BASELINE_CONFIG["NUM_LAYERS"],
            dropout=BASELINE_CONFIG["DROPOUT"],
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    global_model.to(device)

    # Build clients
    clients, client_raw = build_clients(args, jsonl_data, label_map, mode)

    # Validation loader
    valid_loader = build_valid_loader(args, label_map, mode)

    # Server
    server = BaselineFedServer(global_model=global_model, config=_make_config(args))

    # Training loop
    for rnd in range(1, args.rounds + 1):
        server.round = rnd
        selected_ids = server.select_clients(args.num_clients, fraction=0.6)
        models = server.distribute_model(selected_ids)

        updates, weights, accs = {}, {}, []
        for cid in selected_ids:
            result = clients[cid].local_train(models[cid])
            updates[cid] = result["state_dict"]
            weights[cid] = result["num_samples"]
            accs.append(result.get("accuracy", 0.0))

        global_state = server.aggregate(updates, weights)
        server.update_global_model(global_state)

        g_acc, g_loss = server.evaluate(valid_loader) if valid_loader else (0.0, 0.0)
        server.log_round(g_acc, g_loss, accs)
        if rnd % 10 == 0:
            server.save_checkpoint()

    final_path = os.path.join(
        server.output_dir,
        f"baseline_{args.model}_{args.dataset}_alpha{args.alpha}_round{args.rounds}.pth"
    )
    server.save_checkpoint(final_path)
    server.save_history()
    logger.info(f"Training complete! Results saved to: {final_path}")


if __name__ == "__main__":
    main()
