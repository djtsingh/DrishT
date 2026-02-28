# DrishT Model Assessment & Improvement Roadmap

### Detection Model (SSDLite-MobileNetV3-Large)
| Metric | Value | Production Target | Gap |
|--------|-------|-------------------|-----|
| **mAP@0.5** | **0.23** | 0.60+ | **-37%** |
| Best per-class mAP (license_plate) | 0.80 | 0.85+ | -5% |
| Best per-class mAP (autorickshaw) | 0.55 | 0.70+ | -15% |
| Text class mAP | **0.18** | 0.60+ | **-42%** |
| Training epochs | 46 (early stop) | - | - |
| Model size | 15 MB | 20 MB max | OK |

### Recognition Model (CRNN-Light)
| Metric | Value | Production Target | Gap |
|--------|-------|-------------------|-----|
| **CER** | **7.8%** | <5% | **-2.8%** |
| **Word Accuracy** | **81.9%** | 90%+ | **-8.1%** |
| Training epochs | 52 | - | - |
| Model size | 36 MB | 50 MB max | OK |

---

## Brutally Honest Assessment

### What Works
1. **Models run**: End-to-end pipeline successfully loads and produces output
2. **License plates**: Detection performs adequately (0.80 mAP) - the domain is well-defined
3. **Size constraints**: Both models are edge-deployable size-wise
4. **Infrastructure**: Training notebooks, data pipelines, inference code all work

### What Doesn't Work
1. **Detection misses most text**: mAP 0.23 means ~77% of text is missed or incorrectly localized
2. **Recognition errors compound**: Even if detection is good, 8% CER means ~1 in 12 characters wrong
3. **Indian text performance**: The 724-char charset covering 12 scripts means each script gets limited training
4. **Domain mismatch**: Trained on synthetic + Western datasets, tested on Indian scene text

### Why These Results Are Poor for Production
- **Detection mAP 0.23**: In ICDAR/TotalText competitions, baseline methods achieve 0.70-0.85. You're at research-quality, not production-quality.
- **CER 7.8%**: A 10-character word has ~55% chance of having at least one error. For license plates, this means many misreads.
- **Combined error rate**: If detection has 77% miss rate and recognition has 18% word error → effective accuracy ~19% on end-to-end text reading.

---

## Root Cause Analysis

### Detection Weaknesses
1. **Class imbalance**: autorickshaw/truck images dominate, text regions are sparse
2. **Scale variance**: SSD320 struggles with very small text (< 20px)
3. **Limited scene text data**: Only TotalText (1.5K) and ICDAR (1K) vs 10K+ vehicle images
4. **IOU threshold mismatch**: Used 0.5 for matching, but small text needs 0.7+ localization

### Recognition Weaknesses
1. **Script diversity**: 724 chars across 12 scripts = ~60 chars/script average training
2. **Synthetic-real gap**: MJSynth/SynthText are unrealistic for Indian shop signs
3. **Image quality**: Real crops from detection will have blur/distortion not in training
4. **Sequence length**: Fixed 128px width truncates long words

---

## Concrete Improvement Steps (Ranked by Impact)

### Phase 1: Quick Wins (1-2 weeks, +10-15% improvement)

| Action | Expected Gain |
|--------|---------------|
| **Lower detection threshold to 0.3** | +5% recall |
| **Add test-time augmentation** (multi-scale) | +3-5% mAP |
| **Beam search decoding** (width=5) | +1-2% word acc |
| **Post-process with language model** | +2-3% word acc |

### Phase 2: Data Improvements (2-4 weeks, +15-25% improvement)

| Action | Expected Gain |
|--------|---------------|
| **Add ICDAR 2017/2019 MLT** (multi-lingual text) | +8-10% text mAP |
| **Generate synthetic Indian text** (IndicCorp fonts) | +5-7% CER |
| **Balance detection dataset** (undersample vehicles) | +5-8% text mAP |
| **Augment recognition with blur/noise** | +2-3% CER on real |

### Phase 3: Architecture Improvements (1-2 months, +20-30% improvement)

| Action | Expected Gain |
|--------|---------------|
| **Switch to FCOS/RetinaNet** for detection | +10-15% mAP (better for small objects) |
| **Add attention to CRNN** (Transformer decoder) | +5-8% word acc |
| **Script-specific recognition heads** | +3-5% per-script accuracy |
| **End-to-end fine-tuning** (detection + recognition joint loss) | +5-10% combined |

### Phase 4: Production Hardening (1 month)

| Action | Purpose |
|--------|---------|
| **Quantization (INT8)** | 4x faster inference |
| **Mobile deployment** (NNAPI/CoreML) | On-device feasibility |
| **Confidence calibration** | Reliable uncertainty estimates |
| **Active learning pipeline** | Continuous improvement |

---

## Recommended Next Steps (Pick One)

### Option A: Double Down on License Plates (Fastest ROI)
- Keep current detection (0.80 mAP on plates is usable)
- Add 5K more Indian license plate images
- Train plate-specific recognizer (just alphanumeric + Hindi digits)
- Expected result: 90%+ accuracy on plates in 2 weeks

### Option B: Fix General Text Detection First
- Add ICDAR MLT 2017/2019 (10K+ multi-lingual images)
- Retrain detection with balanced sampling
- Keep current recognition
- Expected result: 0.45+ mAP in 3 weeks

### Option C: Start Fresh with Pretrained Models
- Use DBNet/PaddleOCR as detection base (pretrained 0.7+ mAP)
- Use PaddleOCR/CRNN variants pretrained on Hindi
- Fine-tune on your dataset
- Expected result: 0.60+ mAP, 5% CER in 2 weeks

---

## What NOT to Do (Goofup Prevention)

| Bad Idea | Why It's Fatal |
|----------|----------------|
| Train longer on same data | Overfitting, won't generalize |
| Increase model size | Edge deployment constraint violated |
| Add more scripts | Dilutes per-script performance |
| Skip evaluation metrics | You won't know if changes help |
| Use high confidence threshold | Miss most detections |
| Deploy as-is | 19% effective accuracy is unusable |

---

## Final Verdict

**Current state**: Research prototype, not production-ready.

**Salvageable?**: Yes — the infrastructure is solid. The models are trainable. The main issue is insufficient and imbalanced training data.

**My recommendation**: Go with **Option A** (license plates) for a quick win, then **Option B** (fix general text) with more data. Option C is faster but loses the educational value of building from scratch.

---

## Files Reference
- Detection checkpoint: `models/detection_output/best_map.pth`
- Recognition checkpoint: `models/recognition_output/best_acc.pth`
- Inference pipeline: `inference/ocr_pipeline.py`
- Training notebooks: `notebooks/kaggle_train_*.ipynb`
