"""
Graph Dataset for Baseline Models (GCN / GAT)

Reuses the same TRG graph construction logic as the main model.
Each sample is a PyG Data object with node features, edge indices,
edge attributes, and a class label.
"""

import json
import re
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

from baseline.config import BASELINE_CONFIG


class TrafficGraphBaselineDataset(Dataset):
    """
    Baseline graph dataset — reuses TRG construction logic.

    Data sources:
      1. JSONL file path
      2. Federated partition (list of {input, output} dicts)

    Each sample → PyG Data with:
      - x:           [num_nodes, node_feature_dim]
      - edge_index:  [2, num_edges]
      - edge_attr:   [num_edges, 2]  (type, log1p(time_diff))
      - y:           class label
    """

    def __init__(
        self,
        data_source=None,
        label_map: Optional[Dict[str, int]] = None,
        max_seq_len: int = None,
        node_feature_dim: int = None,
        fit_label_map: bool = False,
        jsonl_path: Optional[str] = None,
        partition_data: Optional[List[Dict]] = None,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len or BASELINE_CONFIG["MAX_SEQ_LEN"]
        self.node_feature_dim = node_feature_dim or BASELINE_CONFIG["NODE_FEATURE_DIM"]

        self.label_map = label_map if label_map is not None else {}
        self._next_label_id = len(self.label_map)
        self.data_list: List[Data] = []

        if jsonl_path is not None:
            self._load_from_jsonl(jsonl_path, fit_label_map)
        elif partition_data is not None:
            self._load_from_list(partition_data)
        elif data_source is not None:
            if isinstance(data_source, str):
                self._load_from_jsonl(data_source, fit_label_map)
            elif isinstance(data_source, list):
                self._load_from_list(data_source)

    def _load_from_jsonl(self, jsonl_path: str, fit_label_map: bool = False):
        if not os.path.exists(jsonl_path):
            print(f"[GraphDataset] File not found: {jsonl_path}")
            return

        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        all_labels = []
        entries = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
                all_labels.append(entry["output"])
            except (json.JSONDecodeError, KeyError):
                continue

        if fit_label_map:
            for label in sorted(set(all_labels)):
                if label not in self.label_map:
                    self.label_map[label] = self._next_label_id
                    self._next_label_id += 1
            print(f"[GraphDataset] Label map fitted: {len(self.label_map)} classes")

        for entry in tqdm(entries, desc="Parsing TRG Graphs"):
            graph_data = self._parse_text_to_graph(entry["input"], entry["output"])
            if graph_data is not None:
                self.data_list.append(graph_data)

        print(f"[GraphDataset] Loaded {len(self.data_list)} graphs from {jsonl_path}")

    def _load_from_list(self, samples: List[Dict]):
        for entry in samples:
            if "input" not in entry or "output" not in entry:
                continue
            app_name = entry["output"]
            if app_name not in self.label_map:
                self.label_map[app_name] = self._next_label_id
                self._next_label_id += 1
            graph_data = self._parse_text_to_graph(entry["input"], app_name)
            if graph_data is not None:
                self.data_list.append(graph_data)

        print(f"[GraphDataset] Loaded {len(self.data_list)} graphs from partition")

    def _parse_text_to_graph(self, input_text: str, label_str: str) -> Optional[Data]:
        """
        Parse TRG textual representation into a PyG Data object.

        Node format:
          Node_N: [T=X.XXs, Dur=..., Up=..., Dn=..., Cnt=..., IAT=...] [Seq: ...]
        Edge format:
          (src -> dst, Type: Burst|Adj_NextStart|Adj_NextEnd)
        """
        try:
            node_pattern = re.compile(
                r"Node_(\d+): \[T=(.*?)s, Dur=(.*?), Up=(.*?), "
                r"Dn=(.*?), Cnt=(.*?), IAT=(.*?)\] \[Seq: (.*?)\]"
            )
            nodes = {}

            for match in node_pattern.finditer(input_text):
                nid, t, dur, up, dn, cnt, iat, seq_str = match.groups()
                nid = int(nid)

                seq_vals = [float(x.strip()) for x in seq_str.split(",") if x.strip()]
                if len(seq_vals) > self.max_seq_len:
                    seq_vals = seq_vals[:self.max_seq_len]
                else:
                    seq_vals += [0.0] * (self.max_seq_len - len(seq_vals))

                raw_stats = [float(dur), float(up), float(dn), float(cnt), float(iat)]
                log_stats = np.log1p(raw_stats).tolist()

                nodes[nid] = {"t": float(t), "stats": log_stats, "seq": seq_vals}

            if not nodes:
                return None

            sorted_nids = sorted(nodes.keys())
            num_nodes = len(sorted_nids)

            raw_features = []
            for nid in sorted_nids:
                nd = nodes[nid]
                stats = nd["stats"]
                seq_vals = nd["seq"]

                remain_dim = self.node_feature_dim - len(stats)
                if remain_dim > 0:
                    seq_feats = [x / 1500.0 for x in seq_vals[:remain_dim]]
                    if len(seq_feats) < remain_dim:
                        seq_feats += [0.0] * (remain_dim - len(seq_feats))
                else:
                    seq_feats = []

                node_feat = stats + seq_feats
                raw_features.append(node_feat)

            feat_array = np.array(raw_features, dtype=np.float32)
            if feat_array.shape[1] > self.node_feature_dim:
                feat_array = feat_array[:, :self.node_feature_dim]

            x = torch.tensor(feat_array, dtype=torch.float)

            # Edges
            edge_pattern = re.compile(r"\((\d+) -> (\d+), Type: (.*?)\)")
            type_map = {"Burst": 0, "Adj_NextStart": 1, "Adj_NextEnd": 2}

            src_list, dst_list, edge_attrs = [], [], []
            node_times = np.array([nodes[i]["t"] for i in sorted_nids])

            for match in edge_pattern.finditer(input_text):
                u, v, e_type = match.groups()
                u, v = int(u), int(v)
                if u not in nodes or v not in nodes:
                    continue

                src_list.append(u)
                dst_list.append(v)
                type_val = type_map.get(e_type.strip(), 0)
                dt = abs(node_times[v] - node_times[u])
                edge_attrs.append([float(type_val), float(np.log1p(dt))])

            if not src_list:
                edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                edge_attr = torch.tensor([[0, 0.0]], dtype=torch.float)
            else:
                edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

            label_id = self.label_map.get(label_str, 0)
            y = torch.tensor([label_id], dtype=torch.long)

            return Data(
                x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=y, num_nodes=num_nodes,
            )

        except Exception as e:
            print(f"[GraphDataset] Parse error: {e}")
            return None

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]

    def get_label_map(self) -> Dict[str, int]:
        return self.label_map

    @property
    def num_classes(self) -> int:
        return len(self.label_map)
