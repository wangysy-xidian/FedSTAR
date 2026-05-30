"""
Non-IID Data Partitioning for Federated Learning.

Supports multiple partition strategies:
  - label_skew:   Dirichlet-based label distribution skew (simulates diverse app preferences)
  - quantity_skew: Dirichlet-based sample quantity skew (simulates varying data volumes)
  - time_skew:     temporal partitioning (simulates different time windows)
  - feature_skew:  packet-size perturbation (simulates network environment differences)
  - mixed_skew:    combination of label + quantity + feature skew

Reference: Li et al. (2020) "Federated Optimization in Heterogeneous Networks"
"""

import json
import random
import re
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split


def load_jsonl(file_path: str) -> List[Dict]:
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def partition_label_skew(
    data: List[Dict],
    num_clients: int,
    alpha: float = 0.5,
    min_samples_per_client: int = 100,
    seed: int = 42,
) -> Dict[int, List[Dict]]:
    """
    Label-skewed partition via Dirichlet distribution.

    Each client's label distribution follows Dir(α). Lower α → more skewed.
    Simulates different organisations using different subsets of applications.

    Args:
        data:                  full dataset (each dict has 'input' and 'output')
        num_clients:           number of clients
        alpha:                 Dirichlet concentration (lower = more skewed)
        min_samples_per_client: minimum samples guaranteed per client
        seed:                  random seed

    Returns:
        {client_id: [sample_list]}
    """
    np.random.seed(seed)
    random.seed(seed)

    labels = [entry["output"] for entry in data]
    unique_labels = sorted(set(labels))
    label_to_indices = defaultdict(list)

    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    num_labels = len(unique_labels)
    client_data = [[] for _ in range(num_clients)]

    for label in unique_labels:
        indices = label_to_indices[label]
        num_samples = len(indices)

        if num_samples < num_clients:
            client_id = random.randint(0, num_clients - 1)
            client_data[client_id].extend([data[i] for i in indices])
            continue

        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        assignments = (proportions * num_samples).astype(int)
        diff = num_samples - assignments.sum()
        if diff > 0:
            assignments[:diff] += 1
        elif diff < 0:
            assignments[0] -= diff

        start = 0
        for client_id in range(num_clients):
            if assignments[client_id] > 0:
                client_indices = indices[start:start + assignments[client_id]]
                client_data[client_id].extend([data[i] for i in client_indices])
                start += assignments[client_id]

    # Ensure minimum samples per client
    for client_id in range(num_clients):
        if len(client_data[client_id]) < min_samples_per_client:
            extra_needed = min_samples_per_client - len(client_data[client_id])
            source_clients = [c for c in range(num_clients) if c != client_id]
            random.shuffle(source_clients)

            for src_id in source_clients:
                if extra_needed <= 0:
                    break
                transfer = min(extra_needed, len(client_data[src_id]) // 2)
                client_data[client_id].extend(client_data[src_id][:transfer])
                client_data[src_id] = client_data[src_id][transfer:]
                extra_needed -= transfer

    return {i: data for i, data in enumerate(client_data)}


def partition_quantity_skew(
    data: List[Dict],
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> Dict[int, List[Dict]]:
    """
    Quantity-skewed partition via Dirichlet distribution.

    Simulates organisations with different amounts of traffic data.
    """
    np.random.seed(seed)
    random.seed(seed)

    total_samples = len(data)
    proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
    assignments = (proportions * total_samples).astype(int)

    diff = total_samples - assignments.sum()
    if diff > 0:
        assignments[:diff] += 1
    elif diff < 0:
        assignments[0] -= diff

    indices = list(range(total_samples))
    random.shuffle(indices)

    client_data = {}
    start = 0
    for client_id in range(num_clients):
        end = start + assignments[client_id]
        client_indices = indices[start:end]
        client_data[client_id] = [data[i] for i in client_indices]
        start = end

    return client_data


def partition_time_skew(
    data: List[Dict],
    num_clients: int,
    time_key: str = "timestamp",
    seed: int = 42,
) -> Dict[int, List[Dict]]:
    """
    Temporal-skewed partition.

    Each client receives a contiguous time window of data.
    Simulates monthly-updated federated learning with temporal drift.
    """
    random.seed(seed)

    try:
        if time_key in data[0]:
            data_sorted = sorted(data, key=lambda x: x[time_key])
        else:
            data_sorted = data
    except (KeyError, TypeError):
        data_sorted = data

    total = len(data_sorted)
    window_size = total // num_clients

    client_data = {}
    for client_id in range(num_clients):
        start = client_id * window_size
        end = start + window_size if client_id < num_clients - 1 else total
        client_data[client_id] = data_sorted[start:end]

    return client_data


def partition_feature_skew(
        data: List[Dict],
        num_clients: int,
        noise_std: float = 0.1,
        seed: int = 42,
) -> Dict[int, List[Dict]]:
    """
    Feature-skewed partition via packet-size perturbation.

    Each client's packet sizes are scaled by a Gaussian noise factor,
    simulating different network conditions (MTU, fragmentation, etc.).
    """
    np.random.seed(seed)
    random.seed(seed)

    client_shifts = np.random.normal(0, noise_std, size=num_clients)

    client_data = {}
    total = len(data)
    indices = list(range(total))
    random.shuffle(indices)

    chunk_size = total // num_clients
    for client_id in range(num_clients):
        start = client_id * chunk_size
        end = start + chunk_size if client_id < num_clients - 1 else total
        client_indices = indices[start:end]

        shifted_data = []

        def shift_seq_match(match):
            seq_str = match.group(1)
            nums = seq_str.split(",")
            shifted_nums = []
            for n in nums:
                n_str = n.strip()
                if n_str.lstrip('-').isdigit():
                    val = int(n_str)
                    sign = 1 if val > 0 else -1
                    new_val = max(1, int(abs(val) * (1 + client_shifts[client_id])))
                    shifted_nums.append(str(sign * new_val))
                else:
                    shifted_nums.append(n_str)
            return "[Seq: " + ", ".join(shifted_nums) + "]"

        for idx in client_indices:
            entry = dict(data[idx])
            entry["input"] = re.sub(
                r"\[Seq:\s*(.*?)\]", shift_seq_match, entry["input"]
            )
            shifted_data.append(entry)

        client_data[client_id] = shifted_data

    return client_data


def partition_mixed_skew(
    data: List[Dict],
    num_clients: int,
    label_alpha: float = 0.5,
    quantity_alpha: float = 1.0,
    noise_std: float = 0.05,
    seed: int = 42,
) -> Dict[int, List[Dict]]:
    """
    Mixed-skew partition (closest to real-world deployment).

    Combines label skew + quantity skew + feature perturbation.
    """
    np.random.seed(seed)
    random.seed(seed)

    label_partition = partition_label_skew(
        data, num_clients, alpha=label_alpha, seed=seed
    )

    total = len(data)
    proportions = np.random.dirichlet(np.repeat(quantity_alpha, num_clients))
    target_sizes = (proportions * total).astype(int)

    client_data = {i: list(v) for i, v in label_partition.items()}

    for client_id in range(num_clients):
        current = len(client_data[client_id])
        target = target_sizes[client_id]

        if current > target:
            excess = current - target
            client_data[client_id] = client_data[client_id][excess:]
        elif current < target:
            needed = target - current
            for src_id in range(num_clients):
                if src_id == client_id:
                    continue
                if len(client_data[src_id]) > target_sizes[src_id]:
                    transfer = min(
                        needed,
                        len(client_data[src_id]) - target_sizes[src_id],
                    )
                    client_data[client_id].extend(
                        client_data[src_id][-transfer:]
                    )
                    client_data[src_id] = client_data[src_id][:-transfer]
                    needed -= transfer
                    if needed <= 0:
                        break

    return client_data


def split_train_valid(
    client_data: Dict[int, List[Dict]],
    valid_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[Dict[int, List[Dict]], Dict[int, List[Dict]]]:
    """
    Split each client's data into training and validation sets.

    Returns:
        (train_data: {client_id: [samples]}, valid_data: {client_id: [samples]})
    """
    train_data = {}
    valid_data = {}

    for client_id, samples in client_data.items():
        train, valid = train_test_split(
            samples, test_size=valid_ratio, random_state=seed + client_id
        )
        train_data[client_id] = train
        valid_data[client_id] = valid

    return train_data, valid_data


def print_partition_stats(
    client_data: Dict[int, List[Dict]],
    title: str = "Partition Statistics",
) -> None:
    """Print summary statistics for a data partition."""
    print(f"\n{'=' * 50}")
    print(f" {title}")
    print(f"{'=' * 50}")

    total_samples = sum(len(v) for v in client_data.values())
    print(f"Total clients: {len(client_data)}")
    print(f"Total samples: {total_samples}")

    sizes = [len(v) for v in client_data.values()]
    print(f"Sample sizes: min={min(sizes)}, max={max(sizes)}, "
          f"mean={np.mean(sizes):.1f}, std={np.std(sizes):.1f}")

    all_labels = []
    for samples in client_data.values():
        all_labels.extend([s["output"] for s in samples])

    label_counts = Counter(all_labels)
    print(f"Total labels: {len(label_counts)}")
    print(f"Label distribution (top 5):")
    for label, count in label_counts.most_common(5):
        print(f"  {label}: {count} ({count / total_samples * 100:.1f}%)")

    print(f"\nPer-client label distribution:")
    for client_id, samples in sorted(client_data.items()):
        client_labels = Counter([s["output"] for s in samples])
        top_label = client_labels.most_common(1)[0]
        print(f"  Client {client_id}: {len(samples)} samples, "
              f"top label: {top_label[0]} ({top_label[1] / len(samples) * 100:.1f}%)")

    print(f"{'=' * 50}\n")
