# DrishT: Dual-System OCR for Constrained Edge Devices (Indian Context)

## Complete Project Documentation

**Version:** 2.0  
**Last Updated:** July 2025  
**Status:** Data Pipeline Complete | Training Phase Pending

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Project Structure](#3-project-structure)
4. [Environment Setup](#4-environment-setup)
5. [Datasets](#5-datasets)
6. [Data Pipeline — Step by Step](#6-data-pipeline)
7. [Synthetic Data Generation](#7-synthetic-data-generation)
8. [Quality Control & Final Export](#8-quality-control--final-export)
9. [Model Architectures](#9-model-architectures)
10. [Azure Training Strategy](#10-azure-training-strategy)
11. [Training Scripts](#11-training-scripts)
12. [Known Issues & Resolutions](#12-known-issues--resolutions)
13. [Pipeline Run Log](#13-pipeline-run-log)

---

## 1. Project Overview

**DrishT** (from Sanskrit: दृष्ट = "seen/observed") is an end-to-end OCR system designed to run on
constrained edge devices (Raspberry Pi, Jetson Nano, mobile phones) in the Indian context.

### Problem Statement
Indian scene text presents unique challenges:
- **Multilingual text:** 12+ scripts (Devanagari, Bengali, Tamil, Telugu, etc.) co-occurring on signs
- **Harsh conditions:** Dust, glare, monsoon rain, faded paint, cluttered backgrounds
- **Mixed layouts:** Hindi + English on the same signboard, number plates, traffic signs
- **Edge deployment:** Must run at ≥5 FPS on devices with ≤1 GB RAM and no GPU

### Solution: Dual-System Architecture
Instead of a single monolithic model, we use two specialized lightweight models:

| Component | Model | Task | Input | Output |
|-----------|-------|------|-------|--------|
| **Detector** | SSD-VGG16 | Locate text regions in scene images | Full image (300×300) | Bounding boxes + category |
| **Recognizer** | CRNN (CNN + BiLSTM + CTC) | Read text from cropped regions | Word crop (32×W) | Unicode text string |

### Why SSD-VGG16?
- Proven single-shot detector — no region proposal overhead
- VGG16 backbone provides strong feature extraction for text
- Easily quantizable to INT8 for edge deployment
- Pre-trained COCO weights provide excellent transfer learning base

### Why CRNN?
- CNN encoder extracts visual features from variable-width word crops
- BiLSTM captures sequential dependencies (character order matters)
- CTC loss handles variable-length text without explicit alignment
- Supports any Unicode script — same architecture for all 12 Indic scripts + Latin

---

## 2. Architecture

```
Scene Image (1920×1080)
        │
    ┌───▼────┐
    │  SSD   │  ← SSD-VGG16 Text Detector
    │ VGG16  │     Input: 300×300 resized image
    └───┬────┘     Output: N bounding boxes + confidence
        │              Categories: text, license_plate,
        │                          traffic_sign, vehicle
    ┌───▼────┐
    │  NMS   │  ← Non-Maximum Suppression
    │ Filter │     Removes overlapping/low-confidence boxes
    └───┬────┘
        │
    ┌───▼────┐
    │  Crop  │  ← Extract each detected text region
    │& Resize│     Resize to 32×W (maintain aspect ratio)
    └───┬────┘
        │
    ┌───▼────┐
    │  CRNN  │  ← CNN + BiLSTM + CTC Recognizer
    │ Reader │     Input: 32×W grayscale/RGB crop
    └───┬────┘     Output: Unicode text string
        │
    "भारत STOP 42"   ← Final OCR output
```

### Data Flow Summary

```
Raw Datasets (12 sources, ~15 GB)
    │
    ├──[1] build_detection_dataset.py
    │       → data/processed/detection/ (COCO JSON + images)
    │
    ├──[2] build_recognition_dataset.py
    │       → data/processed/recognition/ (word crops + labels.csv)
    │
    ├──[3] generate_synthetic.py
    │       → data/processed/synthetic/ (rendered text images + labels.csv)
    │
    └──[4] quality_and_export.py
            → data/final/ (train/val/test splits, ready for training)
```

---

## 3. Project Structure

```
DrishT/
├── .gitignore                    # Git ignore rules
├── README.md                     # Quick-start guide
├── DOCUMENTATION.md              # ← THIS FILE (comprehensive docs)
├── DATASET_AUDIT.md              # Detailed audit of all 12 raw datasets
├── DATASET_CHECKLIST.md          # Download priority & strategy
├── DATA_EXPORT_REPORT.md         # Final export statistics
├── requirements.txt              # Python dependencies
│
├── src/                          # Model source code
│   ├── ssd/
│   │   ├── __init__.py
│   │   ├── model.py              # SSD-VGG16 detector (PyTorch)
│   │   ├── train.py              # Detection training loop
│   │   └── config.py             # SSD hyperparameters
│   └── crnn/
│       ├── __init__.py
│       ├── model.py              # CRNN recognizer (PyTorch)
│       ├── train.py              # Recognition training loop
│       └── config.py             # CRNN hyperparameters
│
├── scripts/                      # Data pipeline scripts
│   ├── build_detection_dataset.py    # [Step 1] 8 datasets → COCO JSON
│   ├── build_recognition_dataset.py  # [Step 2] Word crops + labels
│   ├── generate_synthetic.py         # [Step 3] Synthetic Indic text
│   ├── quality_and_export.py         # [Step 4] QC + final splits
│   ├── run_pipeline.py               # Master orchestrator
│   ├── create_env.ps1                # PowerShell env setup
│   ├── create_env.sh                 # Bash env setup
│   ├── download_datasets.ps1         # Dataset download helper
│   └── extract_remaining.py          # Extract archives
│
├── fiftyone_scripts/             # FiftyOne dataset visualization
│   ├── load_all.py               # Load all datasets into FiftyOne
│   ├── load_icdar2015.py
│   ├── load_totaltext.py
│   ├── load_ctw1500.py
│   ├── load_iiit5k.py
│   ├── load_mjsynth.py
│   ├── load_indic_scene.py
│   └── sample_fiftyone_loader.py
│
├── misc/
│   └── azure_resources.md        # Azure free tier analysis
│
├── data/                         # ⚠ NOT tracked in Git
│   ├── raw/                      # Downloaded datasets (~15 GB)
│   │   ├── icdar2015/
│   │   ├── totaltext/
│   │   ├── ctw1500/
│   │   ├── iiit5k/
│   │   ├── mjsynth/
│   │   ├── indic_scene/
│   │   ├── ind_kaggle/
│   │   ├── idd/
│   │   └── synthtext/
│   ├── fonts/                    # Noto fonts for synthetic gen
│   │   ├── Noto_Sans_Bengali/
│   │   ├── Noto_Sans_Devanagari/
│   │   └── Noto_Sans_Tamil/
│   ├── processed/                # Pipeline intermediate outputs
│   │   ├── detection/
│   │   ├── recognition/
│   │   └── synthetic/
│   └── final/                    # Training-ready splits
│       ├── detection/
│       ├── recognition/
│       └── stats/
│
├── models/                       # Saved model weights (not in Git)
├── notebooks/                    # Jupyter notebooks (analysis)
└── .venv/                        # Python virtual environment
```

---

## 4. Environment Setup

### Prerequisites
- Python 3.10+ (tested on 3.13.5)
- Windows 10/11 (pipeline tested on Windows PowerShell)
- ~20 GB free disk space (raw datasets + processed outputs)
- Git for version control

### Setup Steps

```powershell
# 1. Clone the repository
git clone <repo-url> DrishT
cd DrishT

# 2. Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. (When ready for training) Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Set encoding for Unicode scripts
$env:PYTHONIOENCODING = "utf-8"
```

### Critical Windows Notes
1. **Always set PYTHONIOENCODING** before running any pipeline script — Indian scripts contain Unicode characters that fail on Windows cp1252 encoding
2. **Always activate the venv** before running scripts — `run_pipeline.py` subprocess calls don't inherit the venv
3. Each script should be run individually with venv activated rather than through `run_pipeline.py`

---

## 5. Datasets

### Overview: 12 Dataset Sources

| # | Dataset | Task | Size | Scripts/Classes | Format |
|---|---------|------|------|-----------------|--------|
| 1 | ICDAR 2015 | Detection + Recognition | 1,500 imgs | Latin text | Quad-point TXT |
| 2 | Total-Text | Curved text detection | 1,555 imgs | Latin text | MATLAB .mat polygon |
| 3 | CTW1500 | Curved text detection | 1,500 imgs | Latin + Chinese | XML polygon |
| 4 | Indic Scene Text | Detection + Recognition | ~4,000 imgs | 12 Indic scripts | Tab-sep TXT |
| 5 | MJSynth (Synth90k) | Recognition (pre-train) | ~8.9M imgs | Latin | Annotation TXT |
| 6 | IIIT 5K-Word | Recognition | 5,000 imgs | Latin | MATLAB .mat |
| 7 | Indian Number Plates | Detection | ~1,600 imgs | License plates | YOLO TXT |
| 8 | Indian Traffic Signs | Detection | ~130 imgs | Traffic signs | Pascal VOC XML |
| 9 | Autorickshaw Dataset | Detection | ~100 imgs | Autorickshaws | Pascal VOC XML |
| 10 | Indian Vehicle Dataset | Detection | ~100 imgs | Vehicles | Pascal VOC XML |
| 11 | IDD (Indian Driving) | Future use | Large | Driving scenes | Various |
| 12 | SynthText | Optional augmentation | 41 GB | Latin | HDF5 |

### Dataset Download Status

| Dataset | Status | Notes |
|---------|--------|-------|
| ICDAR 2015 | ✅ Complete | Required registration |
| Total-Text | ✅ Complete | Downloaded via gdown |
| CTW1500 | ✅ Complete | Annotations from repo + images from zip |
| Indic Scene | ✅ Complete | 12 script directories |
| MJSynth | ⚠ Partial | Dirs 2885-4000 extracted (of 1-4000+). ~340K images usable |
| IIIT 5K | ✅ Complete | Both splits |
| Indian Kaggle Sets | ✅ Complete | Number plates, traffic signs, vehicles |
| IDD | ⏳ Future | Registration approved |
| SynthText | ⏳ Optional | 41 GB, skipped for now |

### MJSynth Special Note
MJSynth (Synth90k) tar extraction was partial — only directories 2885-4000 were extracted
(out of 1-4000+). The `annotation_train.txt` references directories 1-2425 which don't exist
on disk. **Resolution:** The recognition pipeline now scans disk files directly using
`glob('**/*.jpg')` and extracts word labels from filenames (pattern: `NNN_WORD_NNNNN.jpg`).
This yields ~340,000 usable word crops from the partial extraction.

---

## 6. Data Pipeline

The pipeline runs in 4 sequential steps. Each step must complete before the next begins.

### Step 1: Detection Dataset Builder (`build_detection_dataset.py`)

**Purpose:** Unify 8 heterogeneous detection datasets into a single COCO JSON format.

**Input:** Raw datasets in `data/raw/` with different annotation formats  
**Output:** `data/processed/detection/`
- `coco_train.json` — COCO format annotations (85% of data)
- `coco_val.json` — COCO format annotations (15% of data)
- `images/` — All images copied and renamed with dataset prefix
- `stats.json` — Per-dataset statistics

**How It Works:**

1. **Eight specialized parsers** handle different annotation formats:
   - `parse_icdar2015()` — Reads UTF-8 BOM quad-point TXT files (x1,y1,...,x4,y4,text)
   - `parse_totaltext()` — Loads MATLAB .mat files, extracts polygon vertices
   - `parse_ctw1500()` — Parses XML with `<box>` elements containing polygon segments
   - `parse_indic_scene()` — Reads tab-separated GT files for 12 scripts (index x1 y1...x4 y4 text)
   - `parse_number_plates()` — Converts YOLO normalized coords to pixel bboxes
   - `parse_traffic_signs()` — Generic Pascal VOC XML parser
   - `parse_autorickshaw()` — Pascal VOC XML
   - `parse_auto_rickshaw_db()` — Pascal VOC XML for Indian vehicles

2. **Unified category mapping:**
   - text (id=1), license_plate (id=2), traffic_sign (id=3)
   - autorickshaw (id=4), tempo (id=5), truck (id=6), bus (id=7)

3. **Quality filters:** `MIN_BOX_SIDE=8` pixels, illegible ("###") text skipped

4. **Output format:** Standard COCO JSON with polygon segmentation where available

**Latest Run Results (after CTW1500 fix):**

```
Dataset                    Images   Annotations
────────────────────────────────────────────────
ICDAR 2015                   1475         6545
Total-Text                   1255        11155
CTW1500                       985         6978
Indic Scene Text             3622        23903
Number Plates                1534         1955
Traffic Signs                 130          194
Autorickshaw                  100          118
Auto Rickshaw DB               73          150
────────────────────────────────────────────────
TOTAL                        9174        50998

Train: 7798 images | Val: 1376 images
```

**Command to run:**
```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING = "utf-8"
python scripts/build_detection_dataset.py
```

---

### Step 2: Recognition Dataset Builder (`build_recognition_dataset.py`)

**Purpose:** Collect word-level crop images with text labels for CRNN training.

**Input:** Raw datasets + detection pipeline images  
**Output:** `data/processed/recognition/`
- `images/` — All word crop images (renamed uniquely)
- `labels_train.csv` — (image, label, script, source)
- `labels_val.csv`
- `charset.json` — Unique character set
- `stats.json`

**Sources:**

| Source | Method | Expected Crops |
|--------|--------|----------------|
| MJSynth | Disk scan + filename parse | ~300K+ (capped at 500K) |
| MJSynth-val | Disk scan (separate portion) | ~50K |
| IIIT 5K | MATLAB .mat + train/test dirs | 5,000 |
| Indic Scene (cropped) | Pre-cropped images in script dirs | ~24K |
| Indic Scene (GT crops) | Crop from scene images using GT coords | ~24K |
| ICDAR 2015 | Crop from scene images | ~6.5K |

**MJSynth Parser (Fixed):**
```python
# Scans disk files directly (not annotation files)
all_images = list(base.glob("**/*.jpg"))  # Found 339,952 on disk
# Extracts word from filename: 1_WORD_12345.jpg → "WORD"
match = re.match(r"\d+_(.+)_\d+$", fname)
```

**Command to run:**
```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING = "utf-8"
python scripts/build_recognition_dataset.py
# ⚠ Takes ~4 hours with 340K MJSynth images
```

---

### Step 3: Synthetic Data Generator (`generate_synthetic.py`)

**Purpose:** Generate augmented synthetic Indian scene text for underrepresented scripts.

**Input:** Indic vocabulary lists + Noto fonts  
**Output:** `data/processed/synthetic/`
- `images/` — Rendered text images with augmentations
- `labels.csv`
- `stats.json`

**How It Works:**

1. **Font discovery:** Searches `data/fonts/` recursively for Noto font families:
   - Bengali → `Noto_Sans_Bengali/NotoSansBengali-Regular.ttf`
   - Hindi/Marathi → `Noto_Sans_Devanagari/NotoSansDevanagari-Regular.ttf`
   - Tamil → `Noto_Sans_Tamil/NotoSansTamil-Regular.ttf`
   - Other scripts → Windows system fonts or Arial fallback

2. **Text rendering:** Random word(s) from vocabulary → `ImageFont.truetype()` → PIL image with:
   - Random sign-like background colors (white, yellow, green, blue, red, etc.)
   - Contrasting text colors
   - Optional border (30% probability)

3. **Augmentation pipeline** (Albumentations):
   - Blur: Gaussian (p=0.3), Motion (p=0.3), Median (p=0.1)
   - Noise: GaussNoise (p=0.3), ISONoise (p=0.2)
   - Color: RandomBrightnessContrast, CLAHE, ColorJitter
   - Weather: RandomSunFlare (p=0.1), RandomShadow (p=0.2)
   - Geometric: Perspective warp (scale 2-8%)
   - Compression: JPEG quality 50-95

4. **Indian-specific effects:**
   - Dust/haze simulation (15% probability) — sandy overlay
   - Rain streaks (10% probability) — vertical streaks

5. **Output:** 2,000 samples per Indic script + 3,000 Latin = 27,000 total
   (with proper Noto fonts, Hindi/Marathi share Devanagari, Bengali/Assamese share Bengali)

**Command to run:**
```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING = "utf-8"
python scripts/generate_synthetic.py
```

---

### Step 4: Quality Control & Final Export (`quality_and_export.py`)

**Purpose:** Validate, filter, merge, and create final train/val/test splits.

**Input:** All processed data from Steps 1-3  
**Output:** `data/final/`

```
data/final/
├── detection/
│   ├── train/   (annotations.json + images/)
│   ├── val/
│   └── test/
├── recognition/
│   ├── train/   (labels.csv + images/)
│   ├── val/
│   ├── test/
│   └── charset.txt
└── stats/
    └── final_stats.json
```

**Quality checks:**
- Image existence verification
- Image corruption detection (`PIL.verify()`)
- Bounding box sanity (positive width/height, clip negatives)
- Label validity (no empty/"###"/UNK labels)
- Minimum crop dimensions (16×8 pixels)

**Splitting strategy:**
- **Detection:** Stratified by dataset source (80/10/10)
- **Recognition:** Stratified by script (80/10/10)
- Synthetic data merged into recognition set before splitting

**Command to run:**
```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING = "utf-8"
python scripts/quality_and_export.py
```

---

## 7. Synthetic Data Generation

### Vocabulary Coverage

The synthetic generator uses curated vocabulary lists for each Indian script, focusing on
words commonly found on Indian signs, boards, and nameplates:

| Script | Sample Words | Font Family |
|--------|-------------|-------------|
| Hindi | भारत, दिल्ली, स्टॉप, खतरा, निकास | Noto Sans Devanagari |
| Bengali | ভারত, কলকাতা, স্টপ, বিপদ, প্রবেশ | Noto Sans Bengali |
| Tamil | இந்தியா, சென்னை, நிறுத்தம், ஆபத்து | Noto Sans Tamil |
| Telugu | భారతదేశం, హైదరాబాద్, ఆపు, ప్రమాదం | System font / fallback |
| Gujarati | ભારત, અમદાવાદ, સ્ટોપ, ખતરો | System font / fallback |
| Kannada | ಭಾರತ, ಬೆಂಗಳೂರು, ನಿಲ್ಲಿ, ಅಪಾಯ | System font / fallback |
| Malayalam | ഭാരതം, കൊച്ചി, നിർത്തുക, അപകടം | System font / fallback |
| Marathi | भारत, मुंबई, थांबा, धोका | Noto Sans Devanagari |
| Odia | ଭାରତ, ଭୁବନେଶ୍ୱର, ରହ, ବିପଦ | System font / fallback |
| Punjabi | ਭਾਰਤ, ਅੰਮ੍ਰਿਤਸਰ, ਰੁਕੋ, ਖ਼ਤਰਾ | System font / fallback |
| Assamese | ভাৰত, গুৱাহাটী, ৰওক, বিপদ | Noto Sans Bengali |
| Urdu | بھارت, دہلی, رکیں, خطرہ | System font / fallback |
| Latin | STOP, DANGER, EXIT, HOSPITAL | Arial / system |

### Number Plates
The generator also creates synthetic Indian number plate text:
- Pattern: `XX 00 XX 0000` (standard Indian format)
- Included in Latin samples (30% of Latin generation)

---

## 8. Quality Control & Final Export

### Quality Metrics

| Check | Detection | Recognition |
|-------|-----------|-------------|
| Missing images | Counted, skipped | Counted, skipped |
| Corrupt images | PIL.verify() | N/A (already cropped) |
| Invalid boxes | w≤0 or h≤0 filtered | N/A |
| Negative coords | Clipped to 0 | N/A |
| Invalid labels | N/A | "###", "#", "?", empty → skipped |
| Duplicate IDs | Checked | N/A |

### Stratified Splitting

To ensure each dataset/script is proportionally represented in train/val/test:

```
For each dataset_source (e.g., ICDAR2015, TotalText, ...):
    shuffle images
    test  = first 10%
    val   = next 10%
    train = remaining 80%
```

Same approach for recognition but stratified by script (hindi, bengali, tamil, ...).

---

## 9. Model Architectures

### SSD-VGG16 Text Detector

**Architecture:**
- **Backbone:** VGG16 (pre-trained on ImageNet) with batch normalization
- **Extra layers:** 6 additional conv layers for multi-scale feature extraction
- **Detection heads:** Conv layers producing class scores + box offsets at 6 scales
- **Input:** 300×300×3 RGB image
- **Output:** Per-scale: confidence scores (C classes) + box regression (4 coords + 1 obj)
- **Anchor boxes:** 8732 default boxes across 6 feature maps

**Categories:**
| ID | Name | Description |
|----|------|-------------|
| 0 | background | No object |
| 1 | text | Any text region |
| 2 | license_plate | Indian vehicle number plate |
| 3 | traffic_sign | Road/traffic sign |
| 4 | autorickshaw | Three-wheeler |
| 5 | tempo | Small commercial vehicle |
| 6 | truck | Truck |
| 7 | bus | Bus |

**Training Configuration:**
- Optimizer: SGD with momentum 0.9, weight decay 5e-4
- Learning rate: 1e-3 → cosine decay
- Batch size: 16 (GPU) / 4 (CPU)
- Data augmentation: Random crop, color jitter, horizontal flip
- Loss: MultiBox loss (smooth L1 for regression + cross-entropy for classification)
- Hard negative mining ratio: 3:1

### CRNN Text Recognizer

**Architecture:**
- **CNN Encoder:** 7 conv layers (BN + ReLU + MaxPool) → H=1 feature map
- **Sequence Mapper:** Reshape to (W', C) → bidirectional feature sequence
- **BiLSTM:** 2-layer bidirectional LSTM (256 hidden units each direction)
- **Output:** Dense layer → softmax over charset (700+ characters)
- **Loss:** CTC (Connectionist Temporal Classification)

**Input/Output:**
- Input: 32×W×1 (grayscale) or 32×W×3 (RGB) word crop image
- Output: Sequence of character probabilities → CTC decode → text string

**Training Configuration:**
- Optimizer: Adam (lr=1e-3) or Adadelta (lr=1.0)
- Batch size: 64 (GPU) / 16 (CPU)
- Image preprocessing: Resize height to 32, maintain aspect ratio, pad/trim width
- Label encoding: Character-to-index mapping from `charset.txt`
- CTC blank token: index 0

---

## 10. Azure Training Strategy

### Free Tier Limitations

| Resource | Free Allowance | Limitation for Training |
|----------|---------------|------------------------|
| B1s VM | 750 hrs/mo | 1 vCPU, 1 GB RAM — too weak for DL training |
| Managed Disks | 2×64 GB P6 SSD | Good for dataset storage |
| Blob Storage | 5 GB LRS | Dataset staging |
| Container Apps | 180K vCPU-s | Not suitable for training |
| Custom Vision | 10K predictions | Pre-built, limited customization |
| Bandwidth | 100 GB/mo | Sufficient for model/data transfer |

### Recommended Approach: Azure ML + Low-Priority Compute

Since the free tier VMs are insufficient for deep learning training, we use:

1. **Azure ML Workspace** (free to create)
2. **Low-Priority/Spot Compute Instances** (~70-90% cheaper than on-demand)
   - `Standard_NC6s_v3` (1× V100 16GB) — ~$0.27/hr spot
   - `Standard_NC4as_T4_v3` (1× T4 16GB) — ~$0.13/hr spot
3. **Azure Blob Storage** for dataset hosting
4. **Budget alerts** at $5, $10, $20 thresholds

### Training Plan

**Phase 1: SSD-VGG16 Detection Fine-tuning**
- Estimated time: 20-30 epochs × ~15 min/epoch = ~8 hours
- Estimated cost: ~$2-4 on T4 spot instance
- Data: 9,174 images, 50,998 annotations

**Phase 2: CRNN Recognition Training**
- Estimated time: 30-50 epochs × ~20 min/epoch = ~17 hours
- Estimated cost: ~$2-5 on T4 spot instance
- Data: ~400K+ word crops

**Phase 3: Optimization & Export**
- INT8 quantization for edge deployment
- ONNX export for cross-platform inference
- Estimated: 1-2 hours compute

**Total estimated Azure cost: $6-15**

### Setup Commands

```bash
# Install Azure ML SDK
pip install azure-ai-ml azure-identity

# Login to Azure
az login

# Create resource group
az group create --name drisht-rg --location eastus

# Create Azure ML workspace
az ml workspace create --name drisht-ml --resource-group drisht-rg

# Create compute cluster (spot/low-priority)
az ml compute create --name gpu-cluster --type AmlCompute \
    --size Standard_NC4as_T4_v3 --min-instances 0 --max-instances 1 \
    --tier low_priority --resource-group drisht-rg --workspace-name drisht-ml
```

---

## 11. Training Scripts

### Detection Training (`src/ssd/train.py`)

See the implementation in `src/ssd/train.py`. Key features:
- COCO JSON dataset loader with augmentation
- SSD-VGG16 with pre-trained backbone
- MultiBox loss with hard negative mining
- Cosine learning rate schedule
- Checkpoint saving + early stopping
- TensorBoard logging

### Recognition Training (`src/crnn/train.py`)

See the implementation in `src/crnn/train.py`. Key features:
- CSV-based dataset loader with dynamic image resizing
- CRNN with configurable CNN backbone depth
- CTC loss with greedy/beam search decoding
- Character accuracy + word accuracy metrics
- Mixed precision training support
- Checkpoint saving with best model tracking

---

## 12. Known Issues & Resolutions

### Issue 1: MJSynth Path Mismatch
**Problem:** `annotation_train.txt` references directories 1-2425 but disk only has dirs 2885-4000 (partial tar extraction).  
**Resolution:** Rewrote MJSynth parser to scan disk files with `glob('**/*.jpg')` and extract labels from filenames (`NNN_WORD_NNNNN.jpg → WORD`).

### Issue 2: Windows Unicode Encoding
**Problem:** Pipeline scripts crash with `UnicodeEncodeError` when printing Indian script characters on Windows (cp1252 encoding).  
**Resolution:** Set `$env:PYTHONIOENCODING="utf-8"` before every Python invocation.

### Issue 3: Venv Not Inherited by Subprocesses
**Problem:** `run_pipeline.py` uses `subprocess.run()` which doesn't inherit the activated venv.  
**Resolution:** Run each pipeline script individually with venv activated, not through `run_pipeline.py`.

### Issue 4: CTW1500 Images Missing
**Problem:** CTW1500 had annotation XML files but no images initially.  
**Resolution:** Extracted `train_images.zip` (1000 images) and `test_images.zip` (500 images) that were already downloaded in the ctw1500 folder.

### Issue 5: Synthetic Font Discovery
**Problem:** Noto fonts inside subdirectories (`Noto_Sans_Bengali/NotoSansBengali-Regular.ttf`) not found by non-recursive glob.  
**Resolution:** Updated `generate_synthetic.py` font discovery to use recursive glob (`**/*FontName*.ttf`).

### Issue 6: Indian Vehicle Unknown Classes  
**Problem:** Indian Vehicle Dataset contains classes (car, two_wheelers, bicycle, tractor, concrete_mixer) not in the category map.  
**Resolution:** These are logged as warnings and skipped. Only tempo/truck/bus/autorickshaw categories are extracted.

### Issue 7: Number Plate Training Labels Missing
**Problem:** Indian Number Plates dataset only has test/valid splits with labels. The train split has images but no label files.  
**Resolution:** Only test/valid splits are used (1,534 images). Additional training data comes from synthetic number plate generation.

---

## 13. Pipeline Run Log

### Run 1 (Initial — before fixes)
```
Detection:  8,189 images, 44,020 annotations (CTW1500: 0)
Recognition: 59,081 crops (MJSynth: 0, IIIT5K: 5K, Indic: 47.5K, ICDAR: 6.5K)
Synthetic: 27,000 samples (all Arial fallback)
Export: train 68,881 / val 8,600 / test 8,600
```

### Run 2 (After CTW1500 fix, MJSynth fix pending completion)
```
Detection:  9,174 images, 50,998 annotations
  ICDAR2015:     1,475 images,  6,545 annotations
  TotalText:     1,255 images, 11,155 annotations
  CTW1500:         985 images,  6,978 annotations  ← NEW
  IndicScene:    3,622 images, 23,903 annotations
  NumPlates:     1,534 images,  1,955 annotations
  TrafficSign:     130 images,    194 annotations
  Autorickshaw:    100 images,    118 annotations
  IndVehicle:       73 images,    150 annotations
  Train: 7,798 | Val: 1,376

Recognition: IN PROGRESS (MJSynth processing ~340K images)
Synthetic: PENDING (fonts fixed, ready to re-run)
Export: PENDING
```

---

*Document generated as part of the DrishT OCR project. For questions, see the README or raise an issue.*
