# DrishT: Dual-System OCR for Constrained Edge Devices (Indian Context)

> **दृष्ट** (DrishT) = "seen/observed" in Sanskrit

An end-to-end OCR system designed for Indian scene text across 12+ scripts, 
optimized for edge deployment on devices with limited compute.

## Architecture

| Component | Model | Task |
|-----------|-------|------|
| **Detector** | SSD-VGG16 | Locate text, plates, signs, vehicles in scene images |
| **Recognizer** | CRNN (CNN + BiLSTM + CTC) | Read Unicode text from cropped regions |

## Quick Start

```powershell
# Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Set encoding for Indic scripts (Windows)
$env:PYTHONIOENCODING = "utf-8"

# Run data pipeline (steps 1-4)
python scripts/build_detection_dataset.py
python scripts/build_recognition_dataset.py
python scripts/generate_synthetic.py
python scripts/quality_and_export.py

# Train models (requires PyTorch + GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python src/ssd/train.py
python src/crnn/train.py
```

## Dataset Summary

- **Detection:** 9,174 images, 50,998 annotations (8 datasets)
- **Recognition:** 400K+ word crops (MJSynth + IIIT5K + Indic Scene + synthetic)
- **Scripts:** Hindi, Bengali, Tamil, Telugu, Gujarati, Kannada, Malayalam, 
  Marathi, Odia, Punjabi, Assamese, Urdu, Latin

## Documentation

See [DOCUMENTATION.md](DOCUMENTATION.md) for comprehensive project docs including:
- Architecture details & data flow
- Step-by-step pipeline explanation
- Model configurations
- Azure training strategy
- Known issues & resolutions

## Cloud Training (Azure)

Training uses Azure ML with low-priority GPU instances. Estimated cost: $6-15.
See [misc/azure_resources.md](misc/azure_resources.md) for free tier analysis.

## Project Structure

```
src/ssd/         — SSD-VGG16 model, config, training
src/crnn/        — CRNN model, config, training
scripts/         — Data pipeline (4 steps)
fiftyone_scripts/ — Dataset visualization
data/            — Raw, processed, and final datasets (not in Git)
```
