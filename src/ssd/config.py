"""
SSD-VGG16 Configuration
=========================
Hyperparameters and paths for SSD-VGG16 text detection training.
"""

from pathlib import Path


class SSDConfig:
    """Configuration for SSD-VGG16 training."""

    # --- Paths ---
    DATA_DIR = Path("data/final/detection")
    TRAIN_JSON = DATA_DIR / "train" / "annotations.json"
    TRAIN_IMAGES = DATA_DIR / "train" / "images"
    VAL_JSON = DATA_DIR / "val" / "annotations.json"
    VAL_IMAGES = DATA_DIR / "val" / "images"
    TEST_JSON = DATA_DIR / "test" / "annotations.json"
    TEST_IMAGES = DATA_DIR / "test" / "images"

    CHECKPOINT_DIR = Path("models/ssd")
    LOG_DIR = Path("runs/ssd")

    # --- Model ---
    NUM_CLASSES = 8          # 7 categories + background
    INPUT_SIZE = 300         # SSD300 standard input
    BACKBONE = "vgg16_bn"   # VGG16 with batch normalization

    # Anchor box aspect ratios per feature map
    ASPECT_RATIOS = [
        [2],                 # conv4_3:  38×38
        [2, 3],              # conv7:    19×19
        [2, 3],              # conv8_2:  10×10
        [2, 3],              # conv9_2:   5×5
        [2],                 # conv10_2:  3×3
        [2],                 # conv11_2:  1×1
    ]

    # Feature map sizes
    FEATURE_MAPS = [38, 19, 10, 5, 3, 1]

    # Default box scales (min_size, max_size) for each feature map
    STEPS = [8, 16, 32, 64, 100, 300]
    MIN_SIZES = [30, 60, 111, 162, 213, 264]
    MAX_SIZES = [60, 111, 162, 213, 264, 315]

    # --- Training ---
    BATCH_SIZE = 16          # Reduce to 4-8 for low-VRAM GPUs
    NUM_WORKERS = 4
    EPOCHS = 120
    WARMUP_EPOCHS = 5

    # Optimizer
    LR = 1e-3
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4

    # LR schedule: cosine annealing
    LR_MIN = 1e-6

    # Loss
    NEG_POS_RATIO = 3        # Hard negative mining ratio
    ALPHA = 1.0              # Localization loss weight

    # --- Augmentation ---
    AUGMENT_TRAIN = True
    COLOR_JITTER = 0.3
    RANDOM_CROP_MIN_IOU = 0.3

    # --- Evaluation ---
    NMS_THRESHOLD = 0.45
    CONFIDENCE_THRESHOLD = 0.5
    MAX_DETECTIONS = 200

    # --- Early stopping ---
    PATIENCE = 15
    MIN_DELTA = 0.001

    # --- Mixed precision ---
    USE_AMP = True

    # --- Checkpoint ---
    SAVE_EVERY = 5           # Save checkpoint every N epochs
    KEEP_LAST = 3            # Keep last N checkpoints
