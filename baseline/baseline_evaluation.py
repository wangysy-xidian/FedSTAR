"""
Baseline Evaluation Script

Evaluates pre-trained baseline models with local/cross-client metrics.

Usage:
    python -m baseline.baseline_evaluation \
        --models cnn1d,bilstm,gcn,gat --data_dir /path/to/data --alpha 0.1
"""

import os
import sys
import argparse
import copy
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Allow running as both script and module
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.config import BASELINE_CONFIG
from baseline.sequence_models import create_sequence_model
from baseline.sequence_dataset import PacketSequenceDataset
from fed.data_partition import load_jsonl, partition_label_skew
from baseline.baseline_client import BaselineFedClient, MODE_SEQUENCE, MODE_GNN

try:
    from baseline.graph_models import create_graph_model
    from baseline.graph_dataset import TrafficGraphBaselineDataset
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

OUTPUT_DIR = BASELINE_CONFIG.get("OUTPUT_DIR", "experiments/baseline_results")


def evaluate_pretrained_model(
    model_name: str, alpha: float, dataset_name: str,
    num_clients: int = 5, rounds: int = 100, local_epochs: int = 5,
    data_dir: Optional[str] = None,
    train_file: str = "train.jsonl",
    valid_file: str = "valid.jsonl",
    checkpoint_dir: str = OUTPUT_DIR,
    ft_epochs: int = 0, ft_lr: float = 1e-4,
) -> Optional[Dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Locate checkpoint
    new_ckpt = os.path.join(
        checkpoint_dir,
        f"baseline_{model_name}_{dataset_name}_alpha{alpha}_round{rounds}.pth"
    )
    old_ckpt = os.path.join(
        checkpoint_dir,
        f"baseline_{model_name}_alpha{alpha}_round{rounds}.pth"
    )

    if os.path.exists(new_ckpt):
        ckpt_path = new_ckpt
    elif os.path.exists(old_ckpt):
        ckpt_path = old_ckpt
    else:
        print(f"Checkpoint not found: {new_ckpt}")
        return None
    print(f"Loaded checkpoint: {os.path.basename(ckpt_path)}")

    data_path = os.path.join(data_dir, train_file)
    valid_path = os.path.join(data_dir, valid_file)

    jsonl_data_train = load_jsonl(data_path)
    all_labels = sorted(set(e["output"] for e in jsonl_data_train))
    label_map = {lbl: i for i, lbl in enumerate(all_labels)}
    num_classes = len(label_map)

    mode = MODE_SEQUENCE if model_name in ("cnn1d", "bilstm") else MODE_GNN
    raw_data_for_partition = load_jsonl(valid_path)

    # Build validation loader
    if mode == MODE_SEQUENCE:
        valid_ds = PacketSequenceDataset()
        valid_ds.load_from_partition(
            load_jsonl(valid_path), label_map=dict(label_map)
        )
        from torch.utils.data import DataLoader as TorchDataLoader
        valid_loader = TorchDataLoader(valid_ds, batch_size=32, shuffle=False)
    else:
        valid_ds = TrafficGraphBaselineDataset(
            jsonl_path=valid_path, label_map=dict(label_map),
            node_feature_dim=BASELINE_CONFIG["NODE_FEATURE_DIM"]
        )
        from torch_geometric.loader import DataLoader as PyGLoader
        valid_loader = PyGLoader(valid_ds, batch_size=32, shuffle=False)

    client_raw = partition_label_skew(
        raw_data_for_partition, num_clients=num_clients, alpha=alpha, seed=42
    )

    # Build model
    if mode == MODE_SEQUENCE:
        global_model = create_sequence_model(
            model_name,
            max_seq_len=BASELINE_CONFIG["MAX_SEQ_LEN"],
            num_classes=num_classes,
            hidden_dim=BASELINE_CONFIG["HIDDEN_DIM"],
            dropout=BASELINE_CONFIG["DROPOUT"],
        )
    else:
        global_model = create_graph_model(
            model_name,
            node_feature_dim=BASELINE_CONFIG["NODE_FEATURE_DIM"],
            num_classes=num_classes,
            hidden_dim=BASELINE_CONFIG["HIDDEN_DIM"],
            num_layers=BASELINE_CONFIG["NUM_LAYERS"],
            dropout=BASELINE_CONFIG["DROPOUT"],
        )

    global_model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"]
    )
    global_model.to(device).eval()

    clients = []
    for cid in range(num_clients):
        if mode == MODE_SEQUENCE:
            ds = PacketSequenceDataset()
            ds.load_from_partition(client_raw[cid], label_map=dict(label_map))
        else:
            ds = TrafficGraphBaselineDataset(
                partition_data=client_raw[cid],
                label_map=dict(label_map),
                node_feature_dim=BASELINE_CONFIG["NODE_FEATURE_DIM"]
            )
        cfg = copy.deepcopy(BASELINE_CONFIG)
        cfg.update({
            "COMMUNICATION_ROUNDS": rounds,
            "DIRICHLET_ALPHA": alpha,
            "NUM_CLIENTS": num_clients,
        })
        clients.append(BaselineFedClient(
            cid, copy.deepcopy(global_model), ds,
            mode=mode, config=cfg, device=device
        ))

    global_metrics = _evaluate_model(global_model, valid_loader, mode, device)

    local_metrics_list, cross_metrics_list = [], []
    criterion = nn.CrossEntropyLoss()

    for cid in range(num_clients):
        client_model = copy.deepcopy(global_model)
        client = clients[cid]
        if ft_epochs > 0 and len(client.train_loader.dataset) > 0:
            client_model.train()
            optimizer = optim.AdamW(
                client_model.parameters(), lr=ft_lr, weight_decay=1e-4
            )
            for _ in range(ft_epochs):
                for batch in client.train_loader:
                    if isinstance(batch, (list, tuple)):
                        inputs, labels = batch[0].to(device), batch[1].to(device)
                    else:
                        inputs, labels = batch.to(device), batch.y.to(device)
                    optimizer.zero_grad()
                    loss = criterion(client_model(inputs), labels)
                    loss.backward()
                    optimizer.step()

        client_model.eval()
        local_metrics_list.append(
            _evaluate_model(client_model, client.train_loader, mode, device)
        )

        other_data = [
            d for oid in range(num_clients) if oid != cid
            for d in client_raw[oid]
        ]
        if other_data:
            cross_metrics_list.append(
                _evaluate_on_external_data(
                    client_model, other_data, label_map, mode, device
                )
            )
        else:
            cross_metrics_list.append({
                "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0
            })

    def _avg_metrics(met_list):
        if not met_list:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        return {
            k: float(np.mean([m[k] for m in met_list]))
            for k in ['accuracy', 'precision', 'recall', 'f1']
        }

    return {
        "model": model_name, "alpha": alpha, "dataset": dataset_name,
        "global": global_metrics,
        "local": _avg_metrics(local_metrics_list),
        "cross": _avg_metrics(cross_metrics_list),
    }


def _evaluate_model(model, loader, mode, device) -> Dict:
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                inputs, labels = batch[0].to(device), batch[1].to(device)
            else:
                inputs, labels = batch.to(device), batch.y.to(device)
            all_preds.extend(model(inputs).argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    if not all_labels:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    return {"accuracy": float(acc), "precision": float(p),
            "recall": float(r), "f1": float(f1)}


def _evaluate_on_external_data(model, data_entries, label_map, mode, device) -> Dict:
    if mode == MODE_SEQUENCE:
        ds = PacketSequenceDataset()
        ds.load_from_partition(data_entries, label_map=dict(label_map))
        from torch.utils.data import DataLoader as TorchDataLoader
        loader = TorchDataLoader(ds, batch_size=32, shuffle=False)
    else:
        ds = TrafficGraphBaselineDataset(
            partition_data=data_entries, label_map=dict(label_map),
            node_feature_dim=BASELINE_CONFIG["NODE_FEATURE_DIM"]
        )
        from torch_geometric.loader import DataLoader as PyGLoader
        loader = PyGLoader(ds, batch_size=32, shuffle=False)
    return _evaluate_model(model, loader, mode, device)


def print_result_table(results: List[Dict], ft_epochs: int):
    if not results:
        return
    print("\n" + "=" * 90)
    title_suffix = f"(FT: {ft_epochs} Epochs)" if ft_epochs > 0 else "(No FT)"
    print(f" Baseline Full Metrics {title_suffix}")
    print("=" * 90)
    for r in results:
        print(f"  {r['model'].upper():<6} [{r['dataset'].upper()}] [α={r['alpha']}]")
        print(f"    ├─ Global: Acc={r['global']['accuracy']:.4f} | "
              f"Pre={r['global']['precision']:.4f} | "
              f"Rec={r['global']['recall']:.4f} | F1={r['global']['f1']:.4f}")
        print(f"    ├─ Local : Acc={r['local']['accuracy']:.4f} | "
              f"Pre={r['local']['precision']:.4f} | "
              f"Rec={r['local']['recall']:.4f} | F1={r['local']['f1']:.4f}")
        print(f"    └─ Cross : Acc={r['cross']['accuracy']:.4f} | "
              f"Pre={r['cross']['precision']:.4f} | "
              f"Rec={r['cross']['recall']:.4f} | F1={r['cross']['f1']:.4f}")
        print("-" * 90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", type=str, default="0.1")
    parser.add_argument("--models", type=str, default="cnn1d,bilstm,gcn,gat")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing train.jsonl and valid.jsonl")
    parser.add_argument("--train_file", type=str, default="train.jsonl")
    parser.add_argument("--valid_file", type=str, default="valid.jsonl")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--ft_epochs", type=int, default=0)
    parser.add_argument("--ft_lr", type=float, default=1e-4)
    parser.add_argument("--dataset", type=str, default="traffic")
    parser.add_argument("--checkpoint_dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    model_names = [x.strip() for x in args.models.split(",") if x.strip()]

    print(f"\nBaseline evaluation | data_dir={args.data_dir} | "
          f"FT epochs: {args.ft_epochs}")

    results = []
    for mdl in model_names:
        for a in alphas:
            r = evaluate_pretrained_model(
                mdl, a, args.dataset, args.num_clients,
                args.rounds, args.local_epochs,
                data_dir=args.data_dir,
                train_file=args.train_file,
                valid_file=args.valid_file,
                checkpoint_dir=args.checkpoint_dir,
                ft_epochs=args.ft_epochs, ft_lr=args.ft_lr,
            )
            if r is not None:
                results.append(r)

    print_result_table(results, args.ft_epochs)


if __name__ == "__main__":
    main()
