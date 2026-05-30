"""
FedSTAR Model Evaluation (supports two-stage prototype inference).

Usage:
    python -m fed.eval_fedstar --ckpt <path> --data_dir <path> [--dataset mydata]
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from torch_geometric.loader import DataLoader

# Allow running as both script and module
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import TrafficGraphDataset
from graph_encoder import TrafficGraphModel
from fed.data_partition import load_jsonl


def parse_args():
    p = argparse.ArgumentParser(description="FedSTAR Model Evaluation")
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to checkpoint .pth file")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory containing train.jsonl and valid.jsonl")
    p.add_argument("--train_file", type=str, default="train.jsonl")
    p.add_argument("--test_file", type=str, default="valid.jsonl")
    p.add_argument("--dataset", type=str, default="traffic")
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()

    data_dir = os.path.abspath(args.data_dir)
    train_path = os.path.join(data_dir, args.train_file)
    test_path = os.path.join(data_dir, args.test_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Test data:  {test_path}")

    # 1. Label Encoder
    train_data = load_jsonl(train_path)
    test_data = load_jsonl(test_path)
    all_labels = [e["output"] for e in train_data] + [e["output"] for e in test_data]
    le = LabelEncoder().fit(all_labels)
    num_classes = len(le.classes_)
    print(f"Classes: {num_classes}")

    # 2. Test loader
    test_dataset = TrafficGraphDataset(test_path, label_encoder=le, fit_label=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Test samples: {len(test_dataset)}")

    # 3. Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model_state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    prototypes = ckpt.get("global_prototypes", None)
    ckpt_round = ckpt.get("round", "?")
    print(f"Checkpoint round: {ckpt_round}")

    # 4. Build model
    model = TrafficGraphModel(
        num_classes=num_classes, num_kernels=64,
        embed_dim=128, hidden_dim=256, gat_heads=4,
    ).to(device)
    model.load_state_dict(model_state, strict=False)
    model.eval()

    if prototypes is not None:
        prototypes = prototypes.to(device)
        print(f"Prototypes loaded: {prototypes.shape}, "
              f"norm mean={prototypes.norm(dim=1).mean().item():.4f}")
    else:
        print("Warning: No prototypes in checkpoint — using single-forward eval")

    # 5. Evaluate
    correct, total, total_loss = 0, 0, 0.0
    all_preds, all_labels_list = [], []
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            bs = batch.y.size(0)

            if prototypes is not None:
                # Two-stage inference (FedSTAR)
                mean_proto = prototypes.mean(dim=0, keepdim=True).expand(bs, -1)
                s1 = model(batch, per_graph_prototypes=mean_proto)
                if isinstance(s1, tuple):
                    s1 = s1[0]
                alpha = F.softmax(s1, dim=-1)
                weighted = torch.matmul(alpha, prototypes)
                logits = model(batch, per_graph_prototypes=weighted)
                if isinstance(logits, tuple):
                    logits = logits[0]
            else:
                logits = model(batch)
                if isinstance(logits, tuple):
                    logits = logits[0]

            loss = criterion(logits, batch.y)
            total_loss += loss.item() * bs
            preds = logits.argmax(dim=-1)
            correct += (preds == batch.y).sum().item()
            total += bs
            all_preds.extend(preds.cpu().tolist())
            all_labels_list.extend(batch.y.cpu().tolist())

    accuracy = correct / total
    avg_loss = total_loss / total
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels_list, all_preds, average="macro", zero_division=0
    )

    print("\n" + "=" * 55)
    print(f"  FedSTAR Evaluation  |  Round {ckpt_round}")
    print("-" * 55)
    print(f"  Accuracy:   {accuracy:.4f}")
    print(f"  Loss:       {avg_loss:.4f}")
    print(f"  Precision:  {precision:.4f}  (macro)")
    print(f"  Recall:     {recall:.4f}  (macro)")
    print(f"  F1:         {f1:.4f}  (macro)")
    print("=" * 55)


if __name__ == "__main__":
    main()
