"""
FedSTAR Statistical Prototype Computation
(S)tatistical (T)raffic-prototype guided (A)lignment and (R)efinement

Computes federated statistical prototypes comprising:
  1. Burst structure parsing — extracts burst groups from raw text
  2. Gaussian anchor activation distribution — per-class mean anchor weights
  3. Packet-size statistical moments — mean, variance, skewness, kurtosis
  4. Burst structure descriptors — count, duration quantiles, interval quantiles, avg_size

Final prototype dimension: 64 (anchor) + 4 (moments) + 8 (burst) = 76
"""

import re
import math
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def parse_nodes_and_edges(input_text: str) -> Tuple[Dict, List]:
    """
    Parse node and edge information from raw text representation.

    Returns:
        nodes: {node_id: {'t': float, 'stats': [...], 'seq': [...]}}
        edges: [(src, dst, edge_type_str)]
    """
    node_pattern = re.compile(
        r"Node_(\d+): \[T=(.*?)s, Dur=(.*?), Up=(.*?), "
        r"Dn=(.*?), Cnt=(.*?), IAT=(.*?)\] \[Seq: (.*?)\]"
    )
    nodes = {}
    for match in node_pattern.finditer(input_text):
        nid, t, dur, up, dn, cnt, iat, seq_str = match.groups()
        nid = int(nid)
        seq_vals = [float(x.strip()) for x in seq_str.split(',') if x.strip()]
        nodes[nid] = {
            't': float(t),
            'stats': [float(dur), float(up), float(dn), float(cnt), float(iat)],
            'seq': seq_vals,
        }

    edge_pattern = re.compile(r"\((\d+) -> (\d+), Type: (.*?)\)")
    edges = []
    for match in edge_pattern.finditer(input_text):
        u, v, e_type = match.groups()
        u, v = int(u), int(v)
        if u in nodes and v in nodes:
            edges.append((u, v, e_type.strip()))

    return nodes, edges


def extract_bursts(nodes: Dict, edges: List) -> List[Dict]:
    """
    Extract burst groups from nodes and edges.

    Burst edges (Type: Burst) connect consecutive flows within the same burst.
    A union-find on the Burst-edge subgraph yields connected components,
    each corresponding to one burst.

    Returns:
        bursts: [{'flow_ids': [...], 'start_t': float, 'end_t': float,
                   'duration': float, 'size': int, 'seq_values': [...]}]
    """
    if not nodes:
        return []

    nid_to_idx = {nid: i for i, nid in enumerate(sorted(nodes.keys()))}
    sorted_nids = sorted(nodes.keys())

    # Union-Find
    parent = list(range(len(sorted_nids)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for u, v, e_type in edges:
        if e_type == 'Burst':
            union(nid_to_idx[u], nid_to_idx[v])

    # Collect connected components
    comp_flows = defaultdict(list)
    for i, nid in enumerate(sorted_nids):
        root = find(i)
        comp_flows[root].append(nid)

    bursts = []
    for root, flow_ids in comp_flows.items():
        flow_ids_sorted = sorted(flow_ids, key=lambda nid: nodes[nid]['t'])
        times = [nodes[nid]['t'] for nid in flow_ids_sorted]
        all_seq = []
        for nid in flow_ids_sorted:
            all_seq.extend(nodes[nid]['seq'])

        bursts.append({
            'flow_ids': flow_ids_sorted,
            'start_t': min(times),
            'end_t': max(times),
            'duration': max(times) - min(times),
            'size': len(flow_ids_sorted),
            'seq_values': all_seq,
        })

    bursts.sort(key=lambda b: b['start_t'])
    return bursts


def compute_burst_descriptors(bursts: List[Dict],
                              quantiles: List[float]) -> np.ndarray:
    """
    Compute burst structure descriptors (8-dim):
      - burst_count (1)
      - duration quantiles (len(quantiles))
      - interval quantiles (len(quantiles))
      - avg_burst_size (1)
    """
    n_q = len(quantiles)

    if not bursts:
        return np.zeros(1 + n_q + n_q + 1, dtype=np.float32)

    burst_count = len(bursts)

    durations = [b['duration'] for b in bursts]
    dur_quantiles = np.quantile(durations, quantiles) if durations else np.zeros(n_q)

    if len(bursts) >= 2:
        intervals = [
            bursts[i + 1]['start_t'] - bursts[i]['start_t']
            for i in range(len(bursts) - 1)
        ]
        interval_quantiles = np.quantile(intervals, quantiles)
    else:
        interval_quantiles = np.zeros(n_q)

    avg_size = np.mean([b['size'] for b in bursts])

    descriptors = np.concatenate([
        [burst_count],
        dur_quantiles,
        interval_quantiles,
        [avg_size],
    ]).astype(np.float32)

    return descriptors


def compute_packet_moments(all_seq_values: List[float]) -> np.ndarray:
    """
    Compute the first four statistical moments of (absolute) packet sizes:
      - mean
      - variance
      - skewness
      - excess kurtosis (Fisher definition)
    """
    if not all_seq_values:
        return np.zeros(4, dtype=np.float32)

    abs_vals = np.abs(np.array(all_seq_values, dtype=np.float32))
    n = len(abs_vals)

    mean = np.mean(abs_vals)
    if n < 2:
        return np.array([mean, 0.0, 0.0, 0.0], dtype=np.float32)

    var = np.var(abs_vals)
    std = np.sqrt(var)

    if std < 1e-8:
        return np.array([mean, var, 0.0, 0.0], dtype=np.float32)

    skew = np.mean(((abs_vals - mean) / std) ** 3)
    kurt = np.mean(((abs_vals - mean) / std) ** 4) - 3.0

    return np.array([mean, var, skew, kurt], dtype=np.float32)


def compute_anchor_activations(
    raw_seqs: List[List[float]],
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    min_sigma: float = 1.0,
    device: torch.device = None,
) -> np.ndarray:
    """
    Compute the mean activation weight for each of K Gaussian anchors.

    For each packet size m:
      π_k(m) = exp(-(m-μ_k)²/(2σ_k²)) / Σ_j exp(-(m-μ_j)²/(2σ_j²))

    Then average π_k over all packets.

    Args:
        raw_seqs:   list of per-sample absolute packet-size sequences
        mu:         anchor centres [K]
        log_sigma:  anchor log-widths [K]
        min_sigma:  minimum sigma for numerical stability
        device:     torch device

    Returns:
        [K] numpy array of mean activation weights
    """
    if device is None:
        device = mu.device

    K = mu.shape[0]
    sigma = F.softplus(log_sigma) + min_sigma

    all_vals = []
    for seq in raw_seqs:
        all_vals.extend([abs(v) for v in seq if v != 0])

    if not all_vals:
        return np.zeros(K, dtype=np.float32)

    batch_size = 4096
    all_activations = []

    for i in range(0, len(all_vals), batch_size):
        batch = torch.tensor(
            all_vals[i:i + batch_size], dtype=torch.float32, device=device
        )
        x_expanded = batch.unsqueeze(-1)          # [B, 1]
        mu_expanded = mu.unsqueeze(0)              # [1, K]
        sigma_expanded = sigma.unsqueeze(0)        # [1, K]

        dist_sq = (x_expanded - mu_expanded) ** 2
        weights = torch.exp(-dist_sq / (2 * sigma_expanded ** 2 + 1e-8))
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        all_activations.append(weights.mean(dim=0).cpu().numpy())

    anchor_activation = np.mean(np.stack(all_activations, axis=0), axis=0)
    return anchor_activation.astype(np.float32)


def compute_client_prototypes(
    model: torch.nn.Module,
    raw_entries: List[Dict],
    label_encoder,
    config: Dict,
    device: torch.device,
) -> Dict[int, np.ndarray]:
    """
    Compute per-class local statistical prototypes for a client.

    For each class c:
      p_{k,c} = [anchor_activation(64) || moments(4) || burst_descriptors(8)]

    Args:
        model:         client's local model (must have node_processor with
                       embedder exposing mu, log_sigma, min_sigma)
        raw_entries:   client's raw data [{'input': ..., 'output': ...}]
        label_encoder: fitted sklearn LabelEncoder
        config:        FED_CONFIG dict
        device:        torch device

    Returns:
        {class_idx: np.ndarray of shape [76]}
    """
    # Resolve embedder (handles both direct and wrapped access patterns)
    if hasattr(model.node_processor, 'embedder'):
        embedder = model.node_processor.embedder
    else:
        embedder = model.node_processor
    mu = embedder.mu.detach()
    log_sigma = embedder.log_sigma.detach()
    min_sigma = embedder.min_sigma

    quantiles = config.get("BURST_QUANTILES", [0.25, 0.5, 0.75])
    anchor_dim = config.get("ANCHOR_DIM", 64)
    moment_dim = config.get("MOMENT_DIM", 4)
    burst_dim = config.get("BURST_DIM", 8)

    # Group entries by class
    class_entries = defaultdict(list)
    for entry in raw_entries:
        label_str = entry['output']
        class_idx = int(label_encoder.transform([label_str])[0])
        class_entries[class_idx].append(entry)

    prototypes = {}
    num_classes = len(label_encoder.classes_)

    for class_idx in range(num_classes):
        entries = class_entries.get(class_idx, [])
        if not entries:
            continue  # skip classes with no samples

        all_raw_seqs = []
        all_burst_descriptors = []

        for entry in entries:
            input_text = entry['input']
            nodes, edges = parse_nodes_and_edges(input_text)
            bursts = extract_bursts(nodes, edges)

            for nid, node in nodes.items():
                all_raw_seqs.append([abs(v) for v in node['seq'] if v != 0])

            burst_desc = compute_burst_descriptors(bursts, quantiles)
            all_burst_descriptors.append(burst_desc)

        # 1. Anchor activation distribution
        anchor_activation = compute_anchor_activations(
            all_raw_seqs, mu, log_sigma, min_sigma, device
        )

        # 2. Packet-size moments
        all_flat_seq = [v for seq in all_raw_seqs for v in seq]
        moments = compute_packet_moments(all_flat_seq)

        # 3. Burst descriptors (averaged over samples)
        avg_burst_desc = np.mean(
            np.stack(all_burst_descriptors, axis=0), axis=0
        )

        prototype = np.concatenate(
            [anchor_activation, moments, avg_burst_desc]
        ).astype(np.float32)
        prototypes[class_idx] = prototype

    return prototypes


def prototypes_to_tensor(
    prototypes: Dict[int, np.ndarray],
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert a prototypes dict to a tensor [num_classes, D_p].
    Missing classes are zero-filled.
    """
    proto_dim = None
    for p in prototypes.values():
        proto_dim = p.shape[0]
        break
    if proto_dim is None:
        proto_dim = 76

    tensor = torch.zeros(num_classes, proto_dim, device=device)
    for class_idx, proto in prototypes.items():
        if class_idx < num_classes:
            tensor[class_idx] = torch.from_numpy(proto).to(device)
    return tensor


def init_global_prototypes(num_classes: int, proto_dim: int,
                           device: torch.device) -> torch.Tensor:
    """Initialise global prototypes as a zero tensor [num_classes, D_p]."""
    return torch.zeros(num_classes, proto_dim, device=device)
