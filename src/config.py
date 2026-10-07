# Configuration parameters for the Adaptive Load Balancing project

# Environment parameters
N_SERVERS = 3
MAX_QUEUE = 20
MAX_STEPS_PER_EPISODE = 200

# DQN Hyperparameters
HIDDEN_SIZE = 192          # Neurons per hidden layer (shared feature extractor)
REPLAY_BUFFER_SIZE = 8000  # Prioritized experience replay buffer capacity
BATCH_SIZE = 48            # Mini-batch size for training
GAMMA = 0.99               # Discount factor
EPSILON_START = 1.0        # Initial exploration rate
EPSILON_END = 0.05         # Minimum exploration rate
EPSILON_DECAY = 0.995      # Epsilon decay per episode
TARGET_UPDATE_FREQ = 10    # Episodes between hard target network syncs
LEARNING_RATE = 0.001      # Neural network learning rate
N_EPISODES = 20            # Training episodes

# Double DQN + Dueling + Prioritized Experience Replay
TAU = 0.01                 # Soft (Polyak) target update rate, applied every train step
WEIGHT_DECAY = 1e-5        # AdamW weight decay for generalization
PER_ALPHA = 0.6            # Prioritized replay: how much prioritization is used (0 = uniform)
PER_BETA_START = 0.4       # Prioritized replay: initial importance-sampling correction
PER_BETA_END = 1.0         # Importance-sampling correction annealed to this by end of training
GRAD_CLIP_NORM = 1.0       # Max gradient norm for gradient clipping

# Learning-rate scheduling
LR_STEP_SIZE = 200         # Episodes between LR decay steps
LR_GAMMA = 0.9             # Multiplicative LR decay factor

# Mixed precision (only engaged automatically when a CUDA device is available)
USE_AMP = None             # None = auto-detect based on device

# Checkpointing / early stopping
EARLY_STOPPING_PATIENCE = 30  # Episodes without improvement before stopping training

# Evaluation parameters
N_EVALUATION_EPISODES = 10

# Azure Dataset
AZURE_DATA_DIR = "data/extracted"
AZURE_DEFAULT_DAY = 1      # Load only Day 1 by default

# File paths
MODEL_PATH = "models/rl_load_balancer"
RESULTS_DIR = "results"
LOGS_DIR = "logs"

# Plot settings
PLOT_DPI = 150
FIGURE_SIZE = (10, 6)

# Random seeds for reproducibility
ENV_SEED = 42
MODEL_SEED = 42