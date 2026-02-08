"""
SSDLite-MobileNetV3 Configuration
====================================
Hyperparameters and paths for SSDLite text detection training.

Model: SSDLite320 with MobileNetV3-Large backbone (torchvision built-in).
Upgrade from SSD-VGG16: 3.4M params (down from 26M), mobile-optimized.
"""

from pathlib import Path


class SSDConfig:
    """Configuration for SSDLite-MobileNetV3 training."""

    # --- Paths ---
    DATA_DIR = Path("data/final/detection")
    TRAIN_JSON = DATA_DIR / "train" / "annotations.json"
    TRAIN_IMAGES = DATA_DIR / "train" / "images"
    VAL_JSON = DATA_DIR / "val" / "annotations.json"
    VAL_IMAGES = DATA_DIR / "val" / "images"
    TEST_JSON = DATA_DIR / "test" / "annotations.json"
    TEST_IMAGES = DATA_DIR / "test" / "images"

    CHECKPOINT_DIR = Path("models/detection")
    LOG_DIR = Path("runs/detection")

    # --- Model ---
    NUM_CLASSES = 8          # 7 categories + background
    INPUT_SIZE = 320         # SSDLite320 native resolution
    BACKBONE = "mobilenet_v3_large"

    # Category mapping (matches build_detection_dataset.py)
    CATEGORIES = {
        0: "background",
        1: "text",
        2: "license_plate",
        3: "traffic_sign",
        4: "autorickshaw",
        5: "tempo",
        6: "truck",
        7: "bus",
    }

    # --- Training ---
    BATCH_SIZE = 16          # MobileNetV3 is memory-efficient; 16 fits most GPUs
    NUM_WORKERS = 4
    EPOCHS = 80              # Fewer epochs needed with strong pretrained backbone
    WARMUP_EPOCHS = 3        # Warm up LR for the first few epochs

    # Backbone freeze strategy
    FREEZE_BACKBONE_EPOCHS = 5   # Freeze backbone for first N epochs

    # Optimizer (SGD with momentum — standard for detection fine-tuning)
    LR = 0.01
    MOMENTUM = 0.9
    WEIGHT_DECAY = 4e-5      # Lighter regularization (MobileNet-style)

    # LR schedule: cosine annealing with warm restarts
    LR_MIN = 1e-6

    # --- Augmentation ---
    AUGMENT_TRAIN = True
    COLOR_JITTER = 0.3
    RANDOM_HORIZONTAL_FLIP = 0.5

    # --- Evaluation ---
    NMS_THRESHOLD = 0.45
    CONFIDENCE_THRESHOLD = 0.5
    MAX_DETECTIONS = 200
    IOU_THRESHOLD_MAP = 0.5  # IoU threshold for mAP computation

    # --- Early stopping ---
    PATIENCE = 12
    MIN_DELTA = 0.001

    # --- Mixed precision ---
    USE_AMP = True

    # --- Checkpoint ---
    SAVE_EVERY = 5
    KEEP_LAST = 3
