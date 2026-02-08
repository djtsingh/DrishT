"""
CRNN-Light Configuration
===========================
Hyperparameters and paths for CRNN-Light text recognition training.

Model: Lightweight CNN (depthwise sep + SE) + BiLSTM + CTC.
Upgrade from basic 7-layer CNN: 2× smaller, faster inference,
better accuracy on complex Indic scripts thanks to SE attention.
"""

from pathlib import Path


class CRNNConfig:
    """Configuration for CRNN-Light training."""

    # --- Paths ---
    DATA_DIR = Path("data/final/recognition")
    TRAIN_CSV = DATA_DIR / "train" / "labels.csv"
    TRAIN_IMAGES = DATA_DIR / "train" / "images"
    VAL_CSV = DATA_DIR / "val" / "labels.csv"
    VAL_IMAGES = DATA_DIR / "val" / "images"
    TEST_CSV = DATA_DIR / "test" / "labels.csv"
    TEST_IMAGES = DATA_DIR / "test" / "images"
    CHARSET_FILE = DATA_DIR / "charset.txt"

    CHECKPOINT_DIR = Path("models/recognition")
    LOG_DIR = Path("runs/recognition")

    # --- Model ---
    IMG_HEIGHT = 32           # Fixed input height
    IMG_WIDTH = 128           # Max width (pad shorter, trim longer)
    NUM_CHANNELS = 1          # 1=grayscale, 3=RGB
    HIDDEN_SIZE = 256         # BiLSTM hidden size per direction
    NUM_LSTM_LAYERS = 2       # Number of BiLSTM layers
    DROPOUT = 0.3             # Dropout between LSTM layers + in CNN

    # CNN: LightCNNEncoder channel progression
    # Stage1: 32→64, Stage2: 128, Stage3: 256, Stage4: 384, Stage5: 512
    # Uses depthwise separable convolutions + SE attention

    # --- Training ---
    BATCH_SIZE = 64           # Reduce to 16-32 for low-VRAM GPUs
    NUM_WORKERS = 4
    EPOCHS = 80               # Fewer epochs with better backbone
    WARMUP_EPOCHS = 3

    # Optimizer
    OPTIMIZER = "adam"
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    BETAS = (0.9, 0.999)

    # LR schedule
    LR_SCHEDULER = "cosine"
    LR_STEP_SIZE = 20
    LR_GAMMA = 0.5
    LR_MIN = 1e-6

    # --- CTC ---
    CTC_BLANK = 0
    DECODE_METHOD = "greedy"
    BEAM_WIDTH = 10

    # --- Augmentation ---
    AUGMENT_TRAIN = True
    RANDOM_ROTATION = 2
    RANDOM_SCALE = (0.9, 1.1)

    # --- Evaluation ---
    EVAL_EVERY = 1

    # --- Early stopping ---
    PATIENCE = 12
    MIN_DELTA = 0.001

    # --- Mixed precision ---
    USE_AMP = True

    # --- Checkpoint ---
    SAVE_EVERY = 5
    KEEP_LAST = 3
