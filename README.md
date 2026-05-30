# FedSTAR: Federated Statistical Traffic-Prototype Guided Alignment and Refinement

Official implementation of FedSTAR — a federated learning framework that uses
**statistical traffic prototypes** to align client feature spaces and resist
temporal concept drift in encrypted traffic classification.

## Core Features

- **Statistical Prototype Aggregation** — Each client uploads compact, physically
  interpretable class prototypes (anchor activations + packet-size moments + burst
  descriptors, 76-dim) alongside model weights. The server aggregates them with
  temporal smoothing, creating a federated reference that guides local training.
- **Dual-Source FiLM Modulation** — Local CNN features are modulated by both
  session-level statistics and the global prototype. An explicit deviation vector
  ($\mathbf{h}_{local} - \mathbf{h}_{global}$) enables targeted corrections: aligned
  clients stay sharp, drifted clients get pulled back.
- **Gaussian Soft-Quantisation** — Raw packet sizes are projected onto learnable
  Gaussian kernels (K-Means initialised), providing Lipschitz-continuous numerical
  embeddings that are robust to jitter, protocol updates, and MTU changes.
- **Relation-Aware GAT with Temporal Edges** — Sessions are modelled as heterogeneous
  graphs with three edge types (Burst / NextStart / NextEnd). Edge features encode
  both relation semantics and sinusoidal time-difference encodings, letting the
  attention mechanism attenuate drift-distorted transitions.
- **Two-Stage Prototype-Guided Inference** — At test time, a lightweight two-pass
  scheme first estimates soft class probabilities from the mean prototype, then
  refines them with the weighted prototype — no ground-truth labels needed.

## Architecture & Tech Stack

FedSTAR addresses a two-dimensional distribution shift: non-IID heterogeneity across
clients *and* temporal drift within each client. The design philosophy is
**multi-level statistical calibration** — stabilise representations at the packet,
session, and federation levels so the model remains robust regardless of whether the
shift originates along the client axis or the time axis.

```
                    Server (Dual-Track Aggregation)
                    ┌─────────────────────────────────┐
                    │  Model Params → FedAvg           │
                    │  Prototypes   → Temporal Smooth  │
                    └──────────────┬──────────────────┘
                                   │ broadcast θ + P_c
            ┌──────────────────────┼──────────────────────┐
            ▼                      ▼                      ▼
       Client 1               Client 2    ...        Client K
   ┌─────────────────────────────────────────────────────────┐
   │  Local Pipeline (per client):                           │
   │                                                         │
   │  Raw Pkts → Gaussian Soft-Quant → Multi-Scale 1D-CNN   │
   │                                  + Dual-Source FiLM     │
   │                                    ↓                    │
   │                              Node Embeddings            │
   │                                    ↓                    │
   │  Session Graph → Edge Encoder → Relation-Aware GAT      │
   │                                    ↓                    │
   │                           GlobalAttention Pooling       │
   │                                    ↓                    │
   │                              MLP Classifier             │
   └─────────────────────────────────────────────────────────┘
```

| Component | Technology | Why |
|---|---|---|
| Packet embedding | Gaussian soft-quantisation (64 anchors) | Lipschitz-continuous → bounded response to numerical jitter |
| Intra-node fusion | Multi-scale 1D-CNN (k=3,5,7) + dual-source FiLM | Captures local burst patterns; prototype-conditioned modulation aligns feature spaces |
| Session graph | Heterogeneous digraph (3 edge types) + sinusoidal time encoding | Topology captures invariant app logic; timing is an edge attribute, not a positional index |
| Graph reasoning | 2-layer GATv2Conv with edge-feature attention | Edge-conditioned attention gates out drift-distorted transitions |
| Readout | GlobalAttention pooling (learned gate) | Learns which nodes matter rather than hand-crafted aggregation |
| Federated aggregation | FedAvg (params) + exponential smoothing (prototypes, μ=0.8) | Prototypes are low-dim (76) → negligible communication overhead |
| Framework | PyTorch + PyTorch Geometric | Mature ecosystem, GPU-accelerated graph ops, reproducible |

```
Project Structure
=================
opensource/
├── fed/                        # Core FedSTAR framework
│   ├── main_federated.py       #   Federated training entry point
│   ├── fed_config.py           #   Hyperparameter configuration
│   ├── fed_server.py           #   Server-side aggregation + prototype management
│   ├── fed_client.py           #   Client-side local training with dual-source FiLM
│   ├── prototype.py            #   Statistical prototype computation
│   ├── data_partition.py       #   Non-IID data partitioning strategies
│   └── eval_fedstar.py         #   Evaluation with two-stage inference
├── baseline/                   # Baseline methods (FedAvg with various backbones)
│   ├── baseline_trainer.py     #   Baseline training entry point
│   ├── baseline_evaluation.py  #   Baseline evaluation
│   ├── baseline_server.py      #   Standard FedAvg server
│   ├── baseline_client.py      #   FedAvg client (sequence + graph modes)
│   ├── sequence_models.py      #   CNN1D and BiLSTM models
│   ├── sequence_dataset.py     #   Packet sequence dataset
│   ├── graph_models.py         #   GCN and GAT models
│   └── graph_dataset.py        #   Graph dataset (TRG construction)
├── ablation/                   # Ablation study (M0–M3 variants)
│   ├── main_ablation.py        #   Ablation training + noise robustness test
│   └── ablation_models.py      #   M0/M1/M2/M3 model definitions
├── graph_encoder.py            # TrafficGraphModel (R-GAT + FiLM)
├── node_encoder.py             # TrafficNodeModel (soft-quantisation + intra-node fusion)
├── dataset.py                  # TrafficGraphDataset (JSONL → PyG)
├── requirements.txt
└── README.md
```

## Installation

```bash
# Create environment
conda create -n fedstar python=3.10
conda activate fedstar

# Install PyTorch (CUDA 11.8 example — adjust to your setup)
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install PyG
pip install torch_geometric

# Install remaining dependencies
pip install -r requirements.txt
```

## Data Format

FedSTAR expects JSONL files where each line is:

```json
{
  "input": "<textual graph representation>",
  "output": "<traffic class label>"
}
```

The textual graph format encodes:
- **Nodes** with flow statistics and packet-size sequences
- **Edges** with type (Burst / NextStart / NextEnd) and time differences

Example (abbreviated):
```
Node_0: [T=0.00s, Dur=0.115, Up=5000, Dn=3000, Cnt=12, IAT=0.023] [Seq: 120, -80, 512, ...]
Node_1: [T=1.55s, Dur=0.200, Up=2000, Dn=8000, Cnt=20, IAT=0.010] [Seq: 200, -100, ...]
(0 -> 1, Type: Burst)
```

### Directory Layout

```
/path/to/data/
├── train.jsonl
├── valid.jsonl
└── test_indistribution.jsonl   # optional, for domain shift evaluation
```

## Usage

### FedSTAR Training

```bash
cd opensource/

python -m fed.main_federated \
    --data_dir /path/to/data \
    --dataset mydataset \
    --mode gnn \
    --partition label \
    --num_clients 25 \
    --rounds 50 \
    --alpha 0.1 \
    --save_dir ./results
```

Key arguments:
| Argument | Description | Default |
|---|---|---|
| `--data_dir` | Path to directory with train.jsonl/valid.jsonl | **required** |
| `--dataset` | Dataset name for logging | `traffic` |
| `--mode` | `gnn` or `multimodal` | `gnn` |
| `--partition` | `label`, `time`, `quantity`, `mixed`, `feature` | `label` |
| `--num_clients` | Number of clients | `25` |
| `--rounds` | Communication rounds | `50` |
| `--alpha` | Dirichlet concentration (lower = more skewed) | `0.1` |
| `--dp` | Enable differential privacy | `False` |

### Baseline Training

```bash
python -m baseline.baseline_trainer \
    --model cnn1d \
    --data_dir /path/to/data \
    --alpha 0.5 \
    --rounds 100
```

Supported models: `cnn1d`, `bilstm`, `gcn`, `gat`

### Ablation Study

```bash
# Train all 4 variants (M0–M3)
python -m ablation.main_ablation \
    --mode train \
    --data_dir /path/to/data \
    --alpha 0.1 \
    --rounds 100

# Noise robustness test (requires trained checkpoints)
python -m ablation.main_ablation \
    --mode noise \
    --data_dir /path/to/data \
    --alpha 0.1
```

### Evaluation

```bash
# FedSTAR evaluation with two-stage inference
python -m fed.eval_fedstar \
    --ckpt ./results/fedstar_mydataset_alpha0.1_round_0050.pth \
    --data_dir /path/to/data

# Baseline evaluation
python -m baseline.baseline_evaluation \
    --models cnn1d,bilstm,gcn,gat \
    --data_dir /path/to/data \
    --alpha 0.1
```

## Ablation Variants

| Variant | Description | Prototypes | Graph |
|---|---|---|---|
| M0 | Base Seq (1D-CNN + FedAvg) | ✗ | ✗ |
| M1 | Graph Only (GAT, no soft-quant, no FiLM) | ✗ | ✓ |
| M2 | Node Only (soft-quant + FiLM, no graph) | ✓ | ✗ |
| M3 | **FedSTAR Full** | ✓ | ✓ |

## Baseline Models (included in this repo)

| Model | Type | Architecture |
|---|---|---|
| CNN1D | Sequence | 3-layer 1D-CNN + attention pooling |
| BiLSTM | Sequence | 2-layer BiLSTM + attention pooling |
| GCN | Graph | Multi-layer GCN + mean pooling |
| GAT | Graph | Multi-layer GAT + mean/max pooling |

These correspond to the **FedETC** (CNN1D) and **FedGCN** (GCN) baselines in our paper,
aggregated via standard FedAvg.

## Compared Methods (external — refer to official repositories)

Our paper also compares against the following state-of-the-art federated adaptation
methods. These are **not** included in this repository — please refer to their
respective papers and official implementations:

| Method | Paper                                                                                                                   | Official Repository                                 |
|---|-------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| **FedCCFA** | Chen et al. "Classifier Clustering and Feature Alignment for Federated Learning under Distributed Concept Drift" (2024) | https://github.com/Chen-Junbao/FedCCFA              |
| **FEAT** | Guo et al. "FEAT: A Federated Approach for Privacy-Preserving Network Traffic Classification in Heterogeneous Environments" (2022)                                      | Refer to the authors' repository                    |
| **FedAvg** | McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data" AISTATS 2017                 | Standard algorithm (implemented in our `baseline/`) |

In our experiments (see Table 2 in the paper), we evaluate composite baselines by
combining these federated adaptation layers with different traffic encoders:

| Composite Baseline | Encoder | FL Adaptation |
|---|---|---|
| FedETC-FedCCFA | CNN1D | FedCCFA |
| FedGCN-FedCCFA | GCN | FedCCFA |
| SM-GAT-FedCCFA | TrafficGraphModel (ours) | FedCCFA |
| FedETC-FEAT | CNN1D | FEAT |
| FedGCN-FEAT | GCN | FEAT |
| SM-GAT-FEAT | TrafficGraphModel (ours) | FEAT |

To reproduce these baselines, obtain the FedCCFA / FEAT implementation from their
official repositories, then replace the `FedServer.aggregate()` method in
`fed/fed_server.py` with the corresponding aggregation logic.

## Dependencies

- Python ≥ 3.9
- PyTorch ≥ 2.0
- PyTorch Geometric ≥ 2.3
- scikit-learn
- NumPy
- tqdm

## License

This project is released under the MIT License.
