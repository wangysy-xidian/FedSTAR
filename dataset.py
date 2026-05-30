"""
Traffic Graph Dataset — parses JSONL traffic records into PyG Data objects.

Each JSONL line is expected to contain:
  {
    "input":  "<textual graph representation>",
    "output": "<traffic class label>"
  }

The textual format encodes:
  - Nodes with flow statistics and packet-size sequences
  - Edges with type (Burst / NextStart / NextEnd) and time differences

Numerical features are normalized via log1p to prevent gradient explosion.
"""

import json
import re
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from tqdm import tqdm


class TrafficGraphDataset(Dataset):
    """
    PyG Dataset that parses textual traffic-graph representations on the fly
    and caches them in memory.

    Args:
        jsonl_path:    path to a .jsonl file
        label_encoder: fitted sklearn LabelEncoder
        max_seq_len:   packet sequence truncation length (default 50)
        fit_label:     if True, fit the label_encoder on this dataset's labels
                       (set True for training set, False for validation/test)
    """

    def __init__(self, jsonl_path, label_encoder=None, max_seq_len=50, fit_label=False):
        super().__init__()
        self.data_list = []
        self.label_encoder = label_encoder
        self.max_seq_len = max_seq_len

        print(f"[*] Loading data from {jsonl_path}...")
        raw_entries = []
        labels = []

        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    raw_entries.append(entry)
                    labels.append(entry['output'])
        except FileNotFoundError:
            print(f"[!] Error: File not found at {jsonl_path}")
            raise

        if fit_label and self.label_encoder is not None:
            self.label_encoder.fit(labels)
            print(f"[*] Label Encoder fitted. Classes: {len(self.label_encoder.classes_)}")

        for entry in tqdm(raw_entries, desc="Parsing Graphs"):
            graph_data = self.parse_text_to_graph(entry['input'], entry['output'])
            if graph_data:
                self.data_list.append(graph_data)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

    def parse_text_to_graph(self, input_text, label_str):
        """
        Parse a textual traffic-graph record into a PyG Data object.

        Node format:
          Node_N: [T=X.XXs, Dur=..., Up=..., Dn=..., Cnt=..., IAT=...] [Seq: v1, v2, ...]

        Edge format:
          (src -> dst, Type: Burst|Adj_NextStart|Adj_NextEnd)
        """
        try:
            # ── 1. Parse node features ──
            node_pattern = re.compile(
                r"Node_(\d+): \[T=(.*?)s, Dur=(.*?), Up=(.*?), "
                r"Dn=(.*?), Cnt=(.*?), IAT=(.*?)\] \[Seq: (.*?)\]"
            )

            nodes = {}

            for match in node_pattern.finditer(input_text):
                nid, t, dur, up, dn, cnt, iat, seq_str = match.groups()
                nid = int(nid)

                # Parse sequence (truncate or pad to max_seq_len)
                seq_vals = [float(x.strip()) for x in seq_str.split(',') if x.strip()]
                if len(seq_vals) > self.max_seq_len:
                    seq_vals = seq_vals[:self.max_seq_len]
                else:
                    seq_vals += [0.0] * (self.max_seq_len - len(seq_vals))

                # Normalize statistics via log1p to compress large dynamic range
                raw_stats = [float(dur), float(up), float(dn), float(cnt), float(iat)]
                log_stats = np.log1p(raw_stats).tolist()

                nodes[nid] = {
                    't': float(t),
                    'stats': log_stats,
                    'seq': seq_vals
                }

            if not nodes:
                return None

            num_nodes = len(nodes)
            sorted_nids = sorted(nodes.keys())

            x_seq = torch.tensor([nodes[i]['seq'] for i in sorted_nids], dtype=torch.float)
            x_stats = torch.tensor([nodes[i]['stats'] for i in sorted_nids], dtype=torch.float)
            node_times = np.array([nodes[i]['t'] for i in sorted_nids])

            # ── 2. Parse edge structure ──
            edge_pattern = re.compile(r"\((\d+) -> (\d+), Type: (.*?)\)")

            src_list = []
            dst_list = []
            edge_attrs = []   # [Type, TimeDiff]

            type_map = {'Burst': 0, 'Adj_NextStart': 1, 'Adj_NextEnd': 2}

            for match in edge_pattern.finditer(input_text):
                u, v, e_type = match.groups()
                u, v = int(u), int(v)

                if u not in nodes or v not in nodes:
                    continue

                src_list.append(u)
                dst_list.append(v)

                type_val = type_map.get(e_type.strip(), 0)
                dt = abs(node_times[v] - node_times[u])

                edge_attrs.append([type_val, dt])

            if not src_list:
                # Isolated node: add self-loop
                edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                edge_attr = torch.tensor([[0, 0.0]], dtype=torch.float)
            else:
                edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

            # ── 3. Label encoding ──
            y = torch.tensor(
                [self.label_encoder.transform([label_str])[0]], dtype=torch.long
            )

            return Data(
                x_seq=x_seq,
                x_stats=x_stats,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                num_nodes=num_nodes
            )

        except Exception as e:
            print(f"[!] Parse Error in dataset.py: {e}")
            return None
