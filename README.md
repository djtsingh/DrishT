# DrishT: Dual-System OCR for Constrained Edge Devices (Indian Context)

> **दृष्ट** (DrishT) = "seen/observed" in Sanskrit

An end-to-end OCR system designed for Indian scene text across 12+ scripts.

**Status**: v2.0 Excellence Roadmap in progress 🚀

## Architecture

### Current (v1) - Edge-Optimized
| Component | Model | Task | Performance |
|-----------|-------|------|-------------|
| **Detector** | SSDLite-MobileNetV3 | Text localization | mAP 0.23 |
| **Recognizer** | CRNN-Light | Unicode text recognition | CER 7.8% |

### Target (v2) - Excellence Grade
| Component | Model | Task | Target |
|-----------|-------|------|--------|
| **Detector** | DBNet++ (ResNet-50) | Scene text detection | mAP 0.85+ |
| **Recognizer** | ViT-Transformer | Multi-script recognition | CER <2% |

📖 See [docs/EXCELLENCE_ROADMAP.md](docs/EXCELLENCE_ROADMAP.md) for the full improvement plan.

## Quick Start

```powershell
# Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch torchvision

# Run inference
python -m inference.ocr_pipeline path/to/image.jpg

# Or use programmatically
from inference.ocr_pipeline import DrishTOCR
ocr = DrishTOCR()
results = ocr.process('image.jpg')
for r in results:
    print(f"{r.text} ({r.confidence:.2f})")
```

## Project Structure

```
drisht/              — v2 models (DBNet++, ViT-Transformer)
├── models/          — Detection & recognition architectures
└── data/            — Indian text synthesizer
inference/           — End-to-end pipeline
notebooks/           — Kaggle training notebooks
models/              — Trained weights (not in git)
docs/                — Documentation & roadmaps
```

## Documentation

| Document | Description |
|----------|-------------|
| [MODEL_ASSESSMENT.md](docs/MODEL_ASSESSMENT.md) | Honest evaluation of current models |
| [EXCELLENCE_ROADMAP.md](docs/EXCELLENCE_ROADMAP.md) | Path to 0.85+ mAP, <2% CER |
| [DOCUMENTATION.md](DOCUMENTATION.md) | Architecture & pipeline details |
