# Kaggle GPU Training Guide — DrishT OCR

## Overview

Azure GPU is blocked (subscription region-locked to centralindia, no compatible GPU VMs available there). Using **Kaggle free T4 GPU** instead (30h/week, ~5-8h total training needed).

## Datasets

| Dataset | Files | Size | Zip |
|---------|-------|------|-----|
| Detection | 9,177 (images + COCO JSON) | 3.3 GB | `data/kaggle_upload/drisht-detection.zip` |
| Recognition | 236,085 (word crops + CSV + charset) | 776 MB | `data/kaggle_upload/drisht-recognition.zip` |

## Step 1: Datasets (Already Uploaded)

Datasets are already uploaded to Kaggle:
- **Detection**: [kaggle.com/datasets/djt5ingh/drisht-detection](https://www.kaggle.com/datasets/djt5ingh/drisht-detection)
- **Recognition**: [kaggle.com/datasets/djt5ingh/drisht-recognition-data](https://www.kaggle.com/datasets/djt5ingh/drisht-recognition-data)

## Step 2: Create Kaggle Notebooks

### Detection Training (~2-3 hours on T4)
1. Go to [kaggle.com](https://www.kaggle.com) → **New Notebook**
2. Copy contents from `notebooks/kaggle_train_detection.ipynb`
3. **Add Data** → search your `drisht-detection` dataset → Add
4. **Settings** → Accelerator: **GPU T4 x2**
5. **Run All**

### Recognition Training (~3-5 hours on T4)
1. Same process with `notebooks/kaggle_train_recognition.ipynb`
2. Add your `drisht-recognition-data` dataset
3. Enable GPU T4
4. Run All

## Step 3: Download Results

After training completes, download from the output panel:
- `best.pth` — best model by validation loss
- `best_map.pth` / `best_acc.pth` — best by accuracy metric
- `*.onnx` — export for mobile deployment
- `history.json` — training metrics
- `training_curves.png` — loss/accuracy plots

Place downloaded files in:
```
models/detection/best.pth
models/detection/ssdlite_detection.onnx
models/recognition/best.pth
models/recognition/crnn_recognition.onnx
```

## Training Details

### Detection: SSDLite320-MobileNetV3
- **Params**: ~3.4M
- **Input**: 320×320 RGB
- **Strategy**: Freeze backbone 5 epochs → unfreeze, cosine LR, AMP
- **Epochs**: 80 (early stopping patience=12)
- **Expected**: mAP@0.5 ~0.55-0.70

### Recognition: CRNN-Light
- **Params**: ~3M
- **Input**: 32×128 grayscale
- **Strategy**: Adam + cosine LR, CTC loss, AMP
- **Epochs**: 80 (early stopping patience=12)
- **Expected**: Word accuracy ~60-75%, char accuracy ~85-92%
- **Charset**: 722 characters (12 Indic scripts + Latin)

## Cost

**$0** — Kaggle provides 30 hours/week of free T4 GPU.
Both models should complete training within a single week's quota (~5-8h total).
