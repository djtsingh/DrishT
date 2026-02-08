# Dataset Checklist — DrishT OCR Project

## Strategy Summary

| Model           | Pretrained On       | Skip Download? | Fine-tune On                                        |
|-----------------|---------------------|----------------|-----------------------------------------------------|
| SSD-MobileNet   | COCO (object det.)  | YES — use checkpoint | ICDAR 2015, Total-Text, CTW1500, (SynthText opt.)  |
| CRNN (CNN enc.) | ImageNet            | YES — use checkpoint | —                                                   |
| CRNN (BiLSTM+CTC) | MJSynth (Synth90k) | NO — download   | IIIT 5K-Word, Indian text crops                     |

---

## Detection Datasets (fine-tune SSD for TEXT detection)

### 1. ICDAR 2015  ⚠ Registration Required
- **Purpose**: Standard scene-text detection benchmark (1000 train / 500 test).
- **URL**: https://rrc.cvc.uab.es/ → Challenge 4 (Incidental Scene Text)
- **Format**: ZIP → images/ + gt/ (txt: x1,y1,...,x4,y4,transcription)
- **Size**: ~150 MB
- **Action**: Register → download → place in `data/raw/icdar2015/`
- **Target dir**:
  ```
  data/raw/icdar2015/
    train_images/
    train_gt/
    test_images/
    test_gt/
  ```

### 2. Total-Text  ✅ Direct / Google Drive
- **Purpose**: Curved & multi-oriented text (1255 train / 300 test).
- **Repo**: https://github.com/cs-chan/Total-Text-Dataset
- **Images**: Google Drive links in repo README.
- **Size**: ~440 MB
- **Target dir**:
  ```
  data/raw/totaltext/
    Images/Train/
    Images/Test/
    Groundtruth/Polygon/Train/
    Groundtruth/Polygon/Test/
  ```

### 3. SCUT-CTW1500  ✅ Direct / Google Drive
- **Purpose**: Curved text detection (1000 train / 500 test).
- **Repo**: https://github.com/Yuliang-Liu/Curve-Text-Detector
- **Images**: Google Drive links in repo README.
- **Size**: ~200 MB
- **Target dir**:
  ```
  data/raw/ctw1500/
    train/text_image/
    train/text_label_curve/
    test/text_image/
    test/text_label_curve/
  ```

### 4. SynthText (OPTIONAL — 41 GB)
- **Purpose**: 800 K synthetic scene-text images with word/char boxes.
- **URL**: https://www.robots.ox.ac.uk/~vgg/data/scenetext/SynthText.zip
- **Size**: ~41 GB ← download only if disk allows
- **Target dir**: `data/raw/synthtext/`

---

## Recognition Datasets (pretrain + fine-tune CRNN)

### 5. MJSynth (Synth90k)  ✅ Direct HTTP
- **Purpose**: 9 M synthetic word images — pretrain CRNN sequence model.
- **URL**: https://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz
- **Size**: ~10 GB
- **Format**: tar.gz → word images + annotation_train.txt / annotation_val.txt / annotation_test.txt
- **Target dir**: `data/raw/mjsynth/`

### 6. IIIT 5K-Word  ✅ Direct HTTP
- **Purpose**: 5000 cropped word images from Google Street View — recognition fine-tuning with diverse fonts.
- **URL**: https://cvit.iiit.ac.in/images/Projects/SceneTextUnderstanding/IIIT5K-Word_V3.0.tar.gz
- **Size**: ~200 MB
- **Format**: tar.gz → images + trainCharBound.mat / testCharBound.mat
- **Target dir**: `data/raw/iiit5k/`

---

## Indian-Specific Datasets

### 7. IIIT-H Indic Scene Text  ⚠ Request Required
- **Purpose**: Indian script scene text (Devanagari, Telugu, etc.).
- **URL**: https://cvit.iiit.ac.in/ → request from CVIT lab
- **Action**: Email / request form → place in `data/raw/indic_scene/`

### 8. IDD — Indian Driving Dataset  ⚠ Registration Required
- **Purpose**: Indian road scenes (context images for augmentation / annotation).
- **URL**: https://idd.insaan.iiit.ac.in/
- **Action**: Register → download → place in `data/raw/idd/`

---

## Download Priority

| Priority | Dataset       | Size    | Method               |
|----------|---------------|---------|----------------------|
| 1        | IIIT 5K-Word  | ~200 MB | aria2c (direct HTTP) |
| 2        | Total-Text    | ~440 MB | gdown (Google Drive) |
| 3        | CTW1500       | ~200 MB | gdown (Google Drive) |
| 4        | MJSynth       | ~10 GB  | aria2c (direct HTTP) |
| 5        | ICDAR 2015    | ~150 MB | manual (registration)|
| 6        | SynthText     | ~41 GB  | aria2c (optional)    |
