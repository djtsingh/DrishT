"""
CRNN Configuration
====================
Hyperparameters and paths for CRNN text recognition training.
"""

from pathlib import Path


class CRNNConfig:
    """Configuration for CRNN training."""

    # --- Paths ---
    DATA_DIR = Path("data/final/recognition")
    TRAIN_CSV = DATA_DIR / "train" / "labels.csv"
    TRAIN_IMAGES = DATA_DIR / "train" / "images"
    VAL_CSV = DATA_DIR / "val" / "labels.csv"
    VAL_IMAGES = DATA_DIR / "val" / "images"
    TEST_CSV = DATA_DIR / "test" / "labels.csv"
    TEST_IMAGES = DATA_DIR / "test" / "images"
    CHARSET_FILE = DATA_DIR / "charset.txt"

    CHECKPOINT_DIR = Path("models/crnn")
    LOG_DIR = Path("runs/crnn")

    # --- Model ---
    IMG_HEIGHT = 32           # Fixed input height
    IMG_WIDTH = 128           # Max width (pad shorter, trim longer)
    NUM_CHANNELS = 1          # 1=grayscale, 3=RGB
    HIDDEN_SIZE = 256         # BiLSTM hidden size per direction
    NUM_LSTM_LAYERS = 2       # Number of BiLSTM layers
    DROPOUT = 0.3             # Dropout between LSTM layers

    # CNN backbone channel progression
    # Input → 64 → 128 → 256 → 256 → 512 → 512 → 512
    CNN_CHANNELS = [64, 128, 256, 256, 512, 512, 512]

    # --- Training ---
    BATCH_SIZE = 64           # Reduce to 16-32 for low-VRAM GPUs
    NUM_WORKERS = 4
    EPOCHS = 100
    WARMUP_EPOCHS = 3

    # Optimizer
    OPTIMIZER = "adam"        # "adam" or "adadelta"
    LR = 1e-3                # Adam default; use 1.0 for Adadelta
    WEIGHT_DECAY = 1e-5
    BETAS = (0.9, 0.999)     # Adam betas

    # LR schedule
    LR_SCHEDULER = "cosine"  # "cosine", "step", or "plateau"
    LR_STEP_SIZE = 20        # For step scheduler
    LR_GAMMA = 0.5           # For step scheduler
    LR_MIN = 1e-6            # For cosine scheduler

    # --- CTC ---
    CTC_BLANK = 0             # CTC blank token index
    DECODE_METHOD = "greedy"  # "greedy" or "beam"
    BEAM_WIDTH = 10           # For beam search

    # --- Augmentation ---
    AUGMENT_TRAIN = True
    RANDOM_ROTATION = 2       # Max rotation degrees
    RANDOM_SCALE = (0.9, 1.1)

    # --- Evaluation ---
    # Metrics: CER (Character Error Rate), WER (Word Error Rate), accuracy
    EVAL_EVERY = 1            # Evaluate every N epochs

    # --- Early stopping ---
    PATIENCE = 15
    MIN_DELTA = 0.001

    # --- Mixed precision ---
    USE_AMP = True

    # --- Checkpoint ---
    SAVE_EVERY = 5
    KEEP_LAST = 3
