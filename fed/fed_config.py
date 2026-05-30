"""
FedSTAR Configuration
(S)tatistical (T)raffic-prototype guided (A)lignment and (R)efinement

All configurable parameters for the federated learning system.
Paths (save_dir, log_dir) are set at runtime via CLI args.
"""

FED_CONFIG = {
    # ═══════════════════════════════════════════════════════════════
    # Federated Learning Basics
    # ═══════════════════════════════════════════════════════════════
    "NUM_CLIENTS": 25,
    "FRACTION_SAMPLE": 0.6,          # fraction of clients selected per round
    "COMMUNICATION_ROUNDS": 50,
    "LOCAL_EPOCHS": 5,

    # ═══════════════════════════════════════════════════════════════
    # Aggregation Strategy
    # ═══════════════════════════════════════════════════════════════
    # Options: "fedavg", "fedprox", "scaffold"
    "AGGREGATION": "fedavg",
    "FEDPROX_MU": 0.01,              # only used when AGGREGATION="fedprox"

    # ═══════════════════════════════════════════════════════════════
    # Differential Privacy (optional)
    # ═══════════════════════════════════════════════════════════════
    "USE_DP": False,
    "DP_EPSILON": 8.0,
    "DP_DELTA": 1e-5,
    "DP_MAX_GRAD_NORM": 1.0,
    "DP_NOISE_MULTIPLIER": 1.1,

    # ═══════════════════════════════════════════════════════════════
    # Communication Optimisation
    # ═══════════════════════════════════════════════════════════════
    "USE_COMPRESSION": False,
    "COMPRESSION_BITS": 8,           # 8-bit or 16-bit quantisation
    "USE_GRADIENT_SPARSIFICATION": False,
    "SPARSIFICATION_RATE": 0.01,     # Top-K fraction to retain

    # ═══════════════════════════════════════════════════════════════
    # Client Selection
    # ═══════════════════════════════════════════════════════════════
    "CLIENT_SELECTION_STRATEGY": "random",  # "random" or "weighted"
    "CLIENT_WEIGHTS": None,                 # auto-computed for "weighted"

    # ═══════════════════════════════════════════════════════════════
    # Local Training
    # ═══════════════════════════════════════════════════════════════
    "LOCAL_BATCH_SIZE": 32,
    "LOCAL_LR": 0.001,
    "LOCAL_WEIGHT_DECAY": 1e-4,
    "LOCAL_OPTIMIZER": "adamw",

    # ═══════════════════════════════════════════════════════════════
    # Anchor Initialisation
    # ═══════════════════════════════════════════════════════════════
    # True:  each client initialises anchors from local data independently
    # False: all clients share global anchors from the server model
    "CLIENT_SPECIFIC_ANCHORS": True,
    "NUM_KERNELS": 64,

    # ═══════════════════════════════════════════════════════════════
    # FedSTAR: Federated Statistical Prototype Aggregation
    # ═══════════════════════════════════════════════════════════════
    "USE_PROTOTYPES": True,
    "PROTOTYPE_DIM": 76,             # 64 (anchor) + 4 (moments) + 8 (burst)
    "ANCHOR_DIM": 64,                # Gaussian anchor activation distribution
    "MOMENT_DIM": 4,                 # mean, var, skew, kurtosis
    "BURST_DIM": 8,                  # burst count + duration Q + interval Q + avg size
    "BURST_QUANTILES": [0.25, 0.5, 0.75],
    "PROTOTYPE_MOMENTUM": 0.8,       # temporal smoothing μ ∈ [0.7, 0.9]
    "PROTOTYPE_MLP_HIDDEN": 64,      # MLP_g hidden dimension

    # ═══════════════════════════════════════════════════════════════
    # Logging & Checkpointing (set at runtime)
    # ═══════════════════════════════════════════════════════════════
    "LOG_INTERVAL": 1,
    "SAVE_INTERVAL": 10,
    "SAVE_DIR": "./results",         # overridden by --save_dir
    "LOG_DIR": "./results",          # overridden by --save_dir
    "SAVE_PREFIX": "fedstar",
}
