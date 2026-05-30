"""
Packet Sequence Dataset

Provides sequence datasets for CNN/BiLSTM baselines. Supports two data sources:
  1. JSONL files with pre-extracted packet_length arrays
  2. Federated partitions (list of dicts with 'input' and 'output' keys)

Each sample is a fixed-length tensor of normalised packet sizes (max_seq_len).
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from baseline.config import BASELINE_CONFIG


class PacketSequenceDataset(Dataset):
    """
    Fixed-length packet sequence dataset.

    Each sample = (tensor[max_seq_len], label_id).
    Packet sizes are normalised by dividing by 1500.0.
    """

    def __init__(self, max_seq_len: int = None, num_classes: int = None):
        self.max_seq_len = max_seq_len or BASELINE_CONFIG["MAX_SEQ_LEN"]
        self.num_classes = num_classes or BASELINE_CONFIG["NUM_CLASSES"]
        self.data: List[Tuple[torch.Tensor, int]] = []
        self.label_map: Dict[str, int] = {}
        self._label_idx = 0

    # ═══════════════════════════════════════════════════════════
    # Loading from federated partition
    # ═══════════════════════════════════════════════════════════

    def load_from_partition(self, partition: List[Dict],
                            label_map: Dict[str, int] = None):
        """Load from a federated partition (list of {input, output} dicts)."""
        if label_map is not None:
            self.label_map = dict(label_map)
            self._label_idx = len(label_map)

        for sample in partition:
            app_name = sample["output"]
            if label_map is not None:
                label_id = label_map.get(app_name, 0)
            else:
                if app_name not in self.label_map:
                    self.label_map[app_name] = self._label_idx
                    self._label_idx += 1
                label_id = self.label_map[app_name]

            if "packet_length" in sample:
                seq = self._build_sequence_tensor(sample["packet_length"])
            elif "file_path" in sample:
                seq = self._extract_sequence(Path(sample["file_path"]))
            else:
                seq = self._parse_instruction_input(sample.get("input", ""))

            if seq is not None:
                self.data.append((seq, label_id))
        return self.label_map

    # ═══════════════════════════════════════════════════════════
    # Loading from raw JSON flow files
    # ═══════════════════════════════════════════════════════════

    def load_from_raw_flows(self, dataset_root: str,
                            label_map: Dict[str, int] = None):
        """
        Load from a directory tree of per-app JSON flow files.

        Directory layout:
            dataset_root/
                AppName1/
                    flow1.json
                    flow2.json
                AppName2/
                    ...

        Each JSON file contains a list of flow objects with fields:
            start_timestamp, packet_length, arrive_time_delta
        """
        if label_map is not None:
            self.label_map = dict(label_map)
            self._label_idx = len(label_map)

        root_path = Path(dataset_root)
        if not root_path.exists():
            print(f"Directory not found: {dataset_root}")
            return

        valid_labels_lower = {
            lbl.lower(): lbl for lbl in self.label_map
        } if self.label_map else {}

        for app_dir in sorted(root_path.iterdir()):
            if not app_dir.is_dir():
                continue
            app_name = app_dir.name

            if valid_labels_lower:
                if app_name.lower() in valid_labels_lower:
                    target_label = valid_labels_lower[app_name.lower()]
                else:
                    continue
            else:
                if app_name not in self.label_map:
                    self.label_map[app_name] = self._label_idx
                    self._label_idx += 1
                target_label = app_name

            label_id = self.label_map.get(target_label, 0)

            for f in app_dir.rglob("*.json"):
                seq = self._extract_sequence(f)
                if seq is not None:
                    self.data.append((seq, label_id))

        print(f"Loaded {len(self.data)} samples from {dataset_root}")
        return self.label_map

    # ═══════════════════════════════════════════════════════════
    # Internal: sequence extraction
    # ═══════════════════════════════════════════════════════════

    def _extract_sequence(self, json_path: Path) -> Optional[torch.Tensor]:
        """Extract interleaved packet sequence from a JSON flow file."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                flows = json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

        if not flows:
            return None

        all_interleaved_packets = []
        for flow in flows:
            start_ts = flow.get("start_timestamp", 0.0)
            pkts = flow.get("packet_length", [])
            deltas = flow.get("arrive_time_delta", [])

            min_len = min(len(pkts), len(deltas))
            for i in range(min_len):
                abs_time = start_ts + deltas[i]
                all_interleaved_packets.append((abs_time, pkts[i]))

        if not all_interleaved_packets:
            return None

        all_interleaved_packets.sort(key=lambda x: x[0])
        ordered_packets = [pkt for _, pkt in all_interleaved_packets]

        return self._build_sequence_tensor(ordered_packets)

    def _parse_instruction_input(self, input_text: str) -> Optional[torch.Tensor]:
        """Parse packet sizes from [Seq: ...] spans in text input."""
        all_packets = []
        for line in input_text.split("\n"):
            if "[Seq:" in line:
                idx = line.find("[Seq:")
                seq_part = line[idx + len("[Seq:"):]
                end_idx = seq_part.rfind("]")
                if end_idx != -1:
                    seq_part = seq_part[:end_idx]
                for token in seq_part.split(","):
                    if token.strip():
                        try:
                            all_packets.append(int(token.strip()))
                        except ValueError:
                            continue
        if not all_packets:
            return torch.zeros(self.max_seq_len)
        return self._build_sequence_tensor(all_packets)

    def _build_sequence_tensor(self, packets: List[int]) -> torch.Tensor:
        """Truncate/pad to max_seq_len and normalise by 1500."""
        if len(packets) > self.max_seq_len:
            packets = packets[:self.max_seq_len]
        else:
            packets = packets + [0] * (self.max_seq_len - len(packets))

        tensor = torch.tensor(packets, dtype=torch.float32)
        tensor = tensor / 1500.0
        return tensor

    # ═══════════════════════════════════════════════════════════
    # Dataset interface
    # ═══════════════════════════════════════════════════════════

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        seq, label = self.data[idx]
        return seq, torch.tensor(label, dtype=torch.long)

    def get_label_map(self) -> Dict[str, int]:
        return self.label_map
