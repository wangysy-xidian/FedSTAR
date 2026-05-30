"""
FedSTAR Ablation Study — Training & Evaluation Entry Point
===========================================================
Usage:
    # Train all 4 variants
    python -m ablation.main_ablation --mode train --data_dir /path/to/data --alpha 0.1 --rounds 100

    # Evaluate only (requires existing checkpoints)
    python -m ablation.main_ablation --mode eval --variant M0 --alpha 0.1

    # Gaussian noise robustness test
    python -m ablation.main_ablation --mode noise --data_dir /path/to/data --alpha 0.1

Output:
    ablation/results/{variant}_{dataset}_alpha{alpha}_round_{round}.pth
    ablation/results/ablation_results.json
"""

import os
import sys
import json
import copy
import argparse
import logging
import re
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Allow running as both script and module
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ablation.ablation_models import (
    M0_BaseSeq, M1_GraphOnly, M2_NodeOnly, M3_FedSTAR,
)
from dataset import TrafficGraphDataset
from fed.data_partition import load_jsonl, partition_label_skew
from fed.fed_config import FED_CONFIG
from fed.fed_server import FedServer
from fed.fed_client import FedClient, MODE_GNN
from fed.prototype import compute_client_prototypes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [Ablation] - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── Variant registry ──────────────────────────────────────────────
VARIANT_REGISTRY = {
    "M0": {
        "cls": M0_BaseSeq, "use_prototypes": False, "use_graph": False,
        "description": "Base Seq (1D-CNN + FedAvg)",
    },
    "M1": {
        "cls": M1_GraphOnly, "use_prototypes": False, "use_graph": True,
        "description": "Graph Only (GAT, no Soft-Quant, no Prototype)",
    },
    "M2": {
        "cls": M2_NodeOnly, "use_prototypes": True, "use_graph": False,
        "description": "Node Only (Soft-Quant + FiLM, no Graph)",
    },
    "M3": {
        "cls": M3_FedSTAR, "use_prototypes": True, "use_graph": True,
        "description": "FedSTAR Full",
    },
}

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ═══════════════════════════════════════════════════════════════════
# Model construction
# ═══════════════════════════════════════════════════════════════════

def build_model(variant: str, num_classes: int, device: torch.device):
    info = VARIANT_REGISTRY[variant]
    if variant == "M0":
        model = M0_BaseSeq(num_classes=num_classes, hidden_dim=128)
    elif variant == "M1":
        model = M1_GraphOnly(num_classes=num_classes, hidden_dim=256, gat_heads=4)
    elif variant == "M2":
        model = M2_NodeOnly(num_classes=num_classes, num_kernels=64,
                            embed_dim=128, hidden_dim=256, prototype_dim=76)
    elif variant == "M3":
        model = M3_FedSTAR(num_classes=num_classes, num_kernels=64,
                           embed_dim=128, hidden_dim=256, gat_heads=4)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return model.to(device), info


# ═══════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════

def evaluate(model, loader, device, prototypes=None, use_prototypes=False):
    """Evaluate on a given loader. Returns {accuracy, precision, recall, f1}."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if use_prototypes and prototypes is not None:
                bs = batch.y.size(0)
                mean_p = prototypes.mean(dim=0, keepdim=True).expand(bs, -1)
                s1 = model(batch, per_graph_prototypes=mean_p)
                if isinstance(s1, tuple):
                    s1 = s1[0]
                alpha = torch.softmax(s1, dim=-1)
                weighted = torch.matmul(alpha, prototypes)
                logits = model(batch, per_graph_prototypes=weighted)
            else:
                logits = model(batch)
            if isinstance(logits, tuple):
                logits = logits[0]

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())

    if not all_labels:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0)
    return {"accuracy": float(acc), "precision": float(p),
            "recall": float(r), "f1": float(f1)}


def evaluate_with_noise(model, loader, device, sigma,
                        prototypes=None, use_prototypes=False):
    """Evaluate after injecting Gaussian noise into x_seq."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            noisy_batch = batch.clone()
            noise = torch.normal(0, sigma, noisy_batch.x_seq.shape, device=device)
            noisy_batch.x_seq = noisy_batch.x_seq.float() + noise

            if use_prototypes and prototypes is not None:
                bs = noisy_batch.y.size(0)
                mean_p = prototypes.mean(dim=0, keepdim=True).expand(bs, -1)
                s1 = model(noisy_batch, per_graph_prototypes=mean_p)
                if isinstance(s1, tuple):
                    s1 = s1[0]
                alpha = torch.softmax(s1, dim=-1)
                weighted = torch.matmul(alpha, prototypes)
                logits = model(noisy_batch, per_graph_prototypes=weighted)
            else:
                logits = model(noisy_batch)
            if isinstance(logits, tuple):
                logits = logits[0]

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(noisy_batch.y.cpu().tolist())

    if not all_labels:
        return 0.0
    return float(precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0)[2])


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

def train_variant(variant: str, args):
    """Train a single ablation variant."""
    info = VARIANT_REGISTRY[variant]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        f"\n{'='*60}\n  Training {variant}: {info['description']}\n{'='*60}"
    )

    data_dir = os.path.abspath(args.data_dir)
    train_path = os.path.join(data_dir, args.train_file)
    test_path = os.path.join(data_dir, args.test_file)
    d2_path = args.d2_path or os.path.join(data_dir, "test_indistribution.jsonl")
    if not os.path.exists(d2_path):
        d2_path = test_path

    raw_train = load_jsonl(train_path)
    raw_test = load_jsonl(test_path)
    all_labels = sorted(set(e["output"] for e in raw_train + raw_test))
    label_encoder = LabelEncoder().fit(all_labels)
    num_classes = len(label_encoder.classes_)

    # Collect packet sizes for K-Means anchor initialisation
    all_pkt = []
    for e in raw_train:
        m = re.search(r"\[Seq:\s*(.*?)\]", e["input"])
        if m:
            for v in m.group(1).split(","):
                v = v.strip()
                if v.lstrip("-").isdigit():
                    all_pkt.append(float(v))

    client_raw = partition_label_skew(
        raw_train, num_clients=args.num_clients, alpha=args.alpha, seed=42)

    # ── Model ──
    model, _ = build_model(variant, num_classes, device)

    # K-Means anchor initialisation (M2 / M3)
    if variant in ("M2", "M3") and all_pkt:
        model.node_processor.init_anchors(all_pkt)
        logger.info(f"  Anchors initialised on {len(all_pkt)} values")

    # ── Config ──
    config = copy.deepcopy(FED_CONFIG)
    config.update({
        "NUM_CLIENTS": args.num_clients,
        "COMMUNICATION_ROUNDS": args.rounds,
        "LOCAL_EPOCHS": args.local_epochs,
        "USE_PROTOTYPES": info["use_prototypes"],
        "DATASET": args.dataset,
        "ALPHA": args.alpha,
        "SAVE_DIR": RESULTS_DIR,
        "SAVE_PREFIX": variant,
    })

    # ── Clients ──
    cache_base = os.path.join(
        RESULTS_DIR, "cache",
        f"{args.dataset}_alpha{args.alpha}_{args.num_clients}clients")
    os.makedirs(cache_base, exist_ok=True)

    clients = []
    for cid in range(args.num_clients):
        cdir = os.path.join(cache_base, f"client_{cid}")
        os.makedirs(cdir, exist_ok=True)
        cfp = os.path.join(cdir, "data.jsonl")
        if not os.path.exists(cfp):
            with open(cfp, "w") as f:
                for e in client_raw[cid]:
                    f.write(json.dumps(e) + "\n")
        ds = TrafficGraphDataset(cfp, label_encoder=label_encoder, fit_label=False)
        cm = copy.deepcopy(model)
        clients.append(FedClient(
            client_id=cid, model=cm, dataset=ds, config=config,
            mode=MODE_GNN, device=device,
            raw_entries=client_raw[cid], label_encoder=label_encoder,
        ))

    test_loader = DataLoader(
        TrafficGraphDataset(test_path, label_encoder=label_encoder, fit_label=False),
        batch_size=32, shuffle=False)
    d2_loader = DataLoader(
        TrafficGraphDataset(d2_path, label_encoder=label_encoder, fit_label=False),
        batch_size=32, shuffle=False)

    # ── Server ──
    server = FedServer(
        global_model=model, num_classes=num_classes, config=config, device=device)

    # ── Prototype initialisation (M2 / M3) ──
    if info["use_prototypes"]:
        logger.info("Initialising global prototypes ...")
        init_protos: Dict[int, Dict[int, np.ndarray]] = {}
        init_counts: Dict[int, Dict[int, int]] = {}
        for cid in range(args.num_clients):
            client = clients[cid]
            client.model.load_state_dict(model.state_dict(), strict=True)
            client.model.eval()
            with torch.no_grad():
                lp = compute_client_prototypes(
                    client.model, client.raw_entries, label_encoder, config, device)
            if lp:
                init_protos[cid] = lp
                cnt = {}
                for e in client.raw_entries:
                    c = int(label_encoder.transform([e["output"]])[0])
                    cnt[c] = cnt.get(c, 0) + 1
                init_counts[cid] = cnt
        server.init_prototypes(init_protos, init_counts)

    # ── Training loop ──
    num_rounds = args.rounds
    logger.info(
        f"Starting {variant} training: "
        f"{args.num_clients} clients × {num_rounds} rounds"
    )
    logger.info(
        f"{'Round':>6}  {'D1 Acc':>8}  {'D1 F1':>7}  {'D2 F1':>7}  {'Δ':>7}"
    )
    logger.info(f"{'-'*6}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}")

    for round_idx in range(num_rounds):
        server.round = round_idx + 1

        selected_ids = server.select_clients(len(clients))
        selected_clients_list = [clients[cid] for cid in selected_ids]

        distributed_models = server.distribute_model(selected_ids)
        global_protos = (
            server.distribute_prototypes() if info["use_prototypes"] else None
        )

        client_updates, client_weights, client_accuracies = {}, {}, []
        client_prototypes: Dict[int, Dict[int, np.ndarray]] = {}
        client_class_counts: Dict[int, Dict[int, int]] = {}

        for cidx, cid in enumerate(selected_ids):
            client = selected_clients_list[cidx]
            result = client.local_train(
                global_state_dict=distributed_models[cid],
                global_prototypes=global_protos,
                server_control_variate=server.server_control_variate,
            )
            client_updates[cid] = result["state_dict"]
            client_weights[cid] = result["num_samples"]
            client_accuracies.append(result.get("accuracy", 0.0))
            if info["use_prototypes"] and "local_prototypes" in result:
                client_prototypes[cid] = result["local_prototypes"]
                cnt = {}
                for e in client.raw_entries:
                    c = int(label_encoder.transform([e["output"]])[0])
                    cnt[c] = cnt.get(c, 0) + 1
                client_class_counts[cid] = cnt

        new_state = server.aggregate(client_updates, client_weights)
        server.update_global_model(new_state)

        if info["use_prototypes"] and client_prototypes:
            server.aggregate_prototypes(client_prototypes, client_class_counts)

        # Log D1 / D2 F1 every 10 rounds (including round 1)
        if server.round == 1 or server.round % 10 == 0:
            protos = server.global_prototypes if info["use_prototypes"] else None
            d1_m = evaluate(model, test_loader, device, protos,
                            info["use_prototypes"])
            d2_m = evaluate(model, d2_loader, device, protos,
                            info["use_prototypes"])
            gap = d1_m["f1"] - d2_m["f1"]
            logger.info(
                f"{server.round:>5d}  {d1_m['accuracy']*100:>7.2f}% "
                f"{d1_m['f1']*100:>6.2f}%  {d2_m['f1']*100:>6.2f}%  "
                f"{gap*100:>6.2f}%"
            )

        if server.round % 10 == 0:
            server.save_checkpoint(server.round)

    server.save_checkpoint(num_rounds, {"final": True})
    server.save_history()
    logger.info(f"{variant} training completed!")

    # ── Final D1 / D2 evaluation ──
    logger.info("Final evaluation on D1 and D2 ...")
    model.load_state_dict(server.global_state_dict, strict=False)
    model.eval()

    protos = server.global_prototypes if info["use_prototypes"] else None

    d1_loader_final = DataLoader(
        TrafficGraphDataset(train_path, label_encoder=label_encoder, fit_label=False),
        batch_size=32, shuffle=False)
    d1_metrics = evaluate(model, d1_loader_final, device, protos,
                          info["use_prototypes"])
    d2_metrics = evaluate(model, d2_loader, device, protos,
                          info["use_prototypes"])

    gap = d1_metrics["f1"] - d2_metrics["f1"]

    logger.info(
        f"  D1 F1: {d1_metrics['f1']*100:.2f}%  |  "
        f"D2 F1: {d2_metrics['f1']*100:.2f}%  |  "
        f"Δ: {gap*100:.2f}%"
    )

    # ── Save ──
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ckpt_path = os.path.join(
        RESULTS_DIR,
        f"{variant}_{args.dataset}_alpha{args.alpha}_round_{args.rounds:04d}.pth")
    torch.save({
        "variant": variant,
        "model_state_dict": model.state_dict(),
        "global_prototypes": protos.cpu() if protos is not None else None,
        "d1_metrics": d1_metrics,
        "d2_metrics": d2_metrics,
        "gap": gap,
        "config": {k: v for k, v in config.items()
                   if isinstance(v, (int, float, str, bool))},
    }, ckpt_path)
    logger.info(f"Checkpoint saved → {ckpt_path}")

    return {"variant": variant, "d1_f1": d1_metrics["f1"],
            "d2_f1": d2_metrics["f1"], "gap": gap}


# ═══════════════════════════════════════════════════════════════════
# Noise robustness test
# ═══════════════════════════════════════════════════════════════════

def noise_robustness_test(args):
    """Gaussian noise robustness test across all 4 variants."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sigmas = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    results = {}

    data_dir = os.path.abspath(args.data_dir)
    train_path = os.path.join(data_dir, args.train_file)
    test_path = os.path.join(data_dir, args.test_file)

    raw_train = load_jsonl(train_path)
    raw_test = load_jsonl(test_path)
    all_labels = sorted(set(e["output"] for e in raw_train + raw_test))
    label_encoder = LabelEncoder().fit(all_labels)
    num_classes = len(label_encoder.classes_)

    test_loader = DataLoader(
        TrafficGraphDataset(test_path, label_encoder=label_encoder, fit_label=False),
        batch_size=32, shuffle=False)

    for variant in ["M0", "M1", "M2", "M3"]:
        info = VARIANT_REGISTRY[variant]
        ckpt_path = os.path.join(
            RESULTS_DIR,
            f"{variant}_{args.dataset}_alpha{args.alpha}_round_{args.rounds:04d}.pth")
        if not os.path.exists(ckpt_path):
            logger.warning(f"  {variant}: checkpoint not found → {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model, _ = build_model(variant, num_classes, device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()

        protos = ckpt.get("global_prototypes")
        if protos is not None:
            protos = protos.to(device)

        f1s = []
        for s in sigmas:
            f1 = evaluate_with_noise(
                model, test_loader, device, s, protos, info["use_prototypes"])
            f1s.append(round(f1 * 100, 2))
        results[variant] = f1s
        logger.info(f"  {variant}: σ={sigmas} → F1={f1s}")

    out = {"sigmas": sigmas, "results": results}
    out_path = os.path.join(RESULTS_DIR, "noise_robustness.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Noise robustness results → {out_path}")

    # Print table
    print("\n" + "=" * 70)
    print("  Noise Robustness Results (Macro-F1 %)")
    print("-" * 70)
    header = "Variant".ljust(12) + "".join(f"σ={s:<5}" for s in sigmas)
    print(header)
    print("-" * 70)
    for v in ["M0", "M1", "M2", "M3"]:
        if v in results:
            row = v.ljust(12) + "".join(f"{x:<7.2f}" for x in results[v])
            print(row)
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="FedSTAR Ablation Study")
    p.add_argument("--mode", type=str, default="train",
                   choices=["train", "eval", "noise"])
    p.add_argument("--variant", type=str, default="all",
                   help="M0 | M1 | M2 | M3 | all")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory containing train.jsonl and valid.jsonl")
    p.add_argument("--train_file", type=str, default="train.jsonl")
    p.add_argument("--test_file", type=str, default="valid.jsonl")
    p.add_argument("--d2_path", type=str, default=None,
                   help="Target-domain test JSONL (default: test_indistribution.jsonl)")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--local_epochs", type=int, default=5)
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--dataset", type=str, default="traffic")
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "noise":
        noise_robustness_test(args)
        return

    variants = ["M0", "M1", "M2", "M3"] if args.variant == "all" else [args.variant]
    all_results = []

    for v in variants:
        r = train_variant(v, args)
        all_results.append(r)

    # Summary
    print("\n" + "=" * 70)
    print(f"  Ablation Results Summary ({args.dataset}, α={args.alpha})")
    print("-" * 70)
    print(f"{'Variant':<8} {'D1 F1':>8} {'D2 F1':>8} {'Δ↓':>8}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['variant']:<8} {r['d1_f1']*100:>7.2f}% "
              f"{r['d2_f1']*100:>7.2f}% {r['gap']*100:>7.2f}%")
    print("=" * 70)

    out_path = os.path.join(RESULTS_DIR, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
