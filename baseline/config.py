"""
Baseline Experiment Configuration

All baseline models (1D-CNN, BiLSTM, GCN, GAT) share these hyperparameters
so the model architecture is the only independent variable.
"""

# ═══════════════════════════════════════════════
# Data Configuration
# ═══════════════════════════════════════════════

# Sequence model settings (CNN / BiLSTM)
MAX_SEQ_LEN = 256
NUM_CLASSES = 53                # set at runtime from data

# Graph model settings (GCN / GAT)
NODE_FEATURE_DIM = 64
HIDDEN_DIM = 64
NUM_LAYERS = 2
GAT_HEADS = 8
DROPOUT = 0.3

# ═══════════════════════════════════════════════
# Federated Learning Settings
# ═══════════════════════════════════════════════

NUM_CLIENTS = 10
LOCAL_EPOCHS = 5
LOCAL_BATCH_SIZE = 32
LOCAL_LR = 0.001
LOCAL_WEIGHT_DECAY = 1e-4
COMMUNICATION_ROUNDS = 100

# Non-IID data partition
# α=1.0 : near IID
# α=0.5 : moderate Non-IID
# α=0.1 : strong Non-IID
# α=0.05: extreme Non-IID
DIRICHLET_ALPHA = 0.5

# ═══════════════════════════════════════════════
# Optimizer
# ═══════════════════════════════════════════════

OPTIMIZER = "Adam"
LOSS_FN = "CrossEntropyLoss"

# ═══════════════════════════════════════════════
# Misc
# ═══════════════════════════════════════════════

RANDOM_SEED = 42
PARTITION_STRATEGY = "label_skew"
OUTPUT_DIR = "experiments/baseline_results"
SAVE_CHECKPOINTS = True
LOG_INTERVAL = 10
DEVICE = "cuda"

# ── Runtime overrides (set by CLI) ──
BASELINE_CONFIG = {
    "MAX_SEQ_LEN": MAX_SEQ_LEN,
    "NUM_CLASSES": NUM_CLASSES,
    "NODE_FEATURE_DIM": NODE_FEATURE_DIM,
    "HIDDEN_DIM": HIDDEN_DIM,
    "NUM_LAYERS": NUM_LAYERS,
    "GAT_HEADS": GAT_HEADS,
    "DROPOUT": DROPOUT,
    "NUM_CLIENTS": NUM_CLIENTS,
    "LOCAL_EPOCHS": LOCAL_EPOCHS,
    "LOCAL_BATCH_SIZE": LOCAL_BATCH_SIZE,
    "LOCAL_LR": LOCAL_LR,
    "LOCAL_WEIGHT_DECAY": LOCAL_WEIGHT_DECAY,
    "COMMUNICATION_ROUNDS": COMMUNICATION_ROUNDS,
    "DIRICHLET_ALPHA": DIRICHLET_ALPHA,
    "OPTIMIZER": OPTIMIZER,
    "LOSS_FN": LOSS_FN,
    "RANDOM_SEED": RANDOM_SEED,
    "PARTITION_STRATEGY": PARTITION_STRATEGY,
    "OUTPUT_DIR": OUTPUT_DIR,
    "SAVE_CHECKPOINTS": SAVE_CHECKPOINTS,
    "LOG_INTERVAL": LOG_INTERVAL,
    "DEVICE": DEVICE,
}
