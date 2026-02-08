# Comprehensive Dataset Audit Report

**Location:** `G:\2025\DrishT\data\raw\`  
**Date:** Auto-generated audit  
**Total Datasets:** 12 sub-datasets across 7 top-level directories

---

## 1. ICDAR 2015

| Property | Value |
|----------|-------|
| **Path** | `data/raw/icdar2015/` |
| **Task** | Text Detection + Recognition |
| **Annotation Format** | ICDAR quad-point TXT (UTF-8 BOM) |
| **Train Images** | 1,000 (`ch4_training_images/`) |
| **Test Images** | 500 (`ch4_test_images/`) |
| **Train GT Files** | 1,000 (`ch4_training_localization_transcription_gt/gt_img_N.txt`) |
| **Test GT Files** | 500 (`Challenge4_Test_Task1_GT/`) |
| **Image Format** | JPG |

**Directory Structure:**
```
icdar2015/
├── ch4_training_images/          # 1,000 JPGs (img_1.jpg .. img_1000.jpg)
├── ch4_test_images/              # 500 JPGs
├── ch4_training_localization_transcription_gt/  # per-image GT
└── Challenge4_Test_Task1_GT/     # test GT
```

**Annotation Sample** (`gt_img_1.txt`):
```
377,117,463,117,463,130,377,130,Genrehütte
493,115,519,115,519,131,493,131,Der
374,155,409,155,409,170,374,170, DISTRIBUTION
```
Format: `x1,y1,x2,y2,x3,y3,x4,y4,transcription` (4-point quadrilateral + word)

---

## 2. Total-Text

| Property | Value |
|----------|-------|
| **Path** | `data/raw/totaltext/` |
| **Task** | Curved Text Detection + Recognition |
| **Annotation Format** | MATLAB `.mat` files (polygon vertices) |
| **Train Images** | ~1,555 (`Images/Train/`) |
| **Test Images** | ~300 (`Images/Test/`) |
| **Image Format** | JPG |

**Directory Structure:**
```
totaltext/
├── Images/
│   ├── Train/                    # ~1,555 JPGs (img1.jpg .. imgNNNN.jpg)
│   └── Test/                     # ~300 JPGs
├── gt_latest/
│   └── Train/                    # per-image .mat GT files
├── Groundtruth/
│   └── Pixel/
│       ├── Train/                # character-level + text-region pixel masks
│       └── Test/
├── __MACOSX/
└── repo/
```

**Annotation Format:**  
MATLAB `.mat` files containing polygon vertex coordinates for curved text regions with transcriptions. Each `.mat` file corresponds to one image. Polygon annotations support arbitrary-shaped (curved) text, unlike axis-aligned bounding boxes.

---

## 3. CTW1500

| Property | Value |
|----------|-------|
| **Path** | `data/raw/ctw1500/` |
| **Task** | Curved Text Detection + Recognition |
| **Annotation Format** | Custom XML (polygon segments + text labels) |
| **Annotation Files** | 1,000 (`annotations_v2/xml_output/`) |
| **Image Format** | JPG (in separate train/test archives) |

**Directory Structure:**
```
ctw1500/
├── annotations_v2/
│   └── xml_output/               # 1,000 XML files (0001.xml .. 1000.xml)
└── repo/                         # original CTW1500 repository
    ├── data/
    ├── images/
    ├── models/
    └── ...
```

**Annotation Sample** (`0001.xml`):
```xml
<image file="0001.jpg">
  <box>
    <label>SALE</label>
    <segs>167,252,176,249,185,247,193,245,200,244,207,244,214,244...</segs>
    <pts>167,238,214,230,214,258,167,266</pts>
  </box>
</image>
```
Format: `<label>` = text transcription, `<segs>` = dense polygon coordinates, `<pts>` = simplified bounding polygon.

---

## 4. MJSynth (Synthetic Words)

| Property | Value |
|----------|-------|
| **Path** | `data/raw/mjsynth/` |
| **Task** | Text Recognition (synthetic word crops) |
| **Annotation Format** | Custom TXT (path + lexicon index) |
| **Total Annotations** | 7,224,612 lines (`annotation_train.txt`) |
| **Image Format** | JPG (word crops) |
| **Status** | Download incomplete (`.aria2` file present) |

**Directory Structure:**
```
mjsynth/
├── mjsynth.tar.gz.aria2          # incomplete download indicator
└── mnt/
    └── ramdisk/
        └── max/
            └── 90kDICT32px/
                ├── annotation_train.txt
                ├── annotation_val.txt
                ├── annotation_test.txt
                ├── lexicon.txt
                └── 1/ .. 2425/   # image subdirectories
```

**Annotation Sample** (`annotation_train.txt`):
```
./2425/1/115_Lube_45484.jpg 45484
./2425/1/116_clotheshorse_15765.jpg 15765
./2425/1/117_SALE_57393.jpg 57393
```
Format: `./path/to/imgfile.jpg lexicon_index`  
Word label is embedded in the filename (e.g., `115_Lube_45484.jpg` → word is "Lube"). Lexicon index maps to `lexicon.txt`.

---

## 5. IIIT5K

| Property | Value |
|----------|-------|
| **Path** | `data/raw/iiit5k/IIIT5K/` |
| **Task** | Text Recognition (cropped words) |
| **Annotation Format** | MATLAB `.mat` files |
| **Train Images** | 2,000 (`train/`) |
| **Test Images** | 3,000 (`test/`) |
| **Image Format** | PNG (word crops) |

**Directory Structure:**
```
iiit5k/
└── IIIT5K/
    ├── train/                    # 2,000 PNG word crops
    ├── test/                     # 3,000 PNG word crops
    ├── traindata.mat             # train annotations
    ├── testdata.mat              # test annotations
    ├── trainCharBound.mat        # character-level bounding boxes (train)
    ├── testCharBound.mat         # character-level bounding boxes (test)
    └── lexicon.txt               # word lexicon
```

**Annotation Format:**  
MATLAB `.mat` files with word labels and per-character bounding box coordinates. Images are pre-cropped word regions (e.g., `1009_2.png`). Each `.mat` entry maps an image filename to its ground-truth text and optional lexicon.

---

## 6. Indic Scene Text (Multi-script)

| Property | Value |
|----------|-------|
| **Path** | `data/raw/indic_scene/verified_twice/` |
| **Task** | Text Detection + Recognition (Indic scripts) |
| **Annotation Format** | Tab-separated TXT (quad-point + transcription) |
| **Languages** | 12: Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu, Urdu |
| **Image Format** | JPEG |

**Directory Structure:**
```
indic_scene/
└── verified_twice/
    ├── assamese/
    │   ├── 1.jpeg, 1_gt.txt
    │   ├── 2.jpeg, 2_gt.txt
    │   └── cropped_images/
    ├── bengali/
    ├── gujarati/
    ├── hindi/
    ├── kannada/
    ├── malayalam/
    ├── marathi/
    ├── odia/
    ├── punjabi/
    ├── tamil/
    ├── telugu/
    └── urdu/
```

**Annotation Sample** (Hindi `4_gt.txt`):
```
1	39	39	248	39	248	108	39	108	फॉर्ब्स
2	95	100	376	100	376	285	95	285	फॉर्ब्स
3	18	266	250	266	250	334	18	334	मार्शल
```
Format: `index\tx1\ty1\tx2\ty2\tx3\ty3\tx4\ty4\ttext` (tab-separated, quad-point bbox + native script transcription)

---

## 7. Indian Traffic Signs

| Property | Value |
|----------|-------|
| **Path** | `data/raw/ind_kaggle/ind_traffic_sign/` |
| **Task** | Object Detection (traffic signs) |
| **Annotation Format** | Pascal VOC XML |
| **Images** | 150 (`Batch-1/batch/`) |
| **Annotations** | 150 (`Annotations/Annotations/`) |
| **Image Format** | JPG |

**Directory Structure:**
```
ind_traffic_sign/
├── Annotations/
│   └── Annotations/             # 150 Pascal VOC XML files
└── Batch-1/
    └── batch/                   # 150 JPG images
```

**Annotation Sample** (`Datacluster Traffic Sign (1).xml`):
```xml
<annotation>
  <filename>Datacluster Traffic Sign (1).jpg</filename>
  <size><width>3120</width><height>4160</height><depth></depth></size>
  <object>
    <name>traffic_sign</name>
    <bndbox>
      <xmin>1222.19</xmin><ymin>428.99</ymin>
      <xmax>2043.36</xmax><ymax>1290.07</ymax>
    </bndbox>
  </object>
</annotation>
```

---

## 8. Indian Number Plates

| Property | Value |
|----------|-------|
| **Path** | `data/raw/ind_kaggle/indian_number_plate/` |
| **Task** | Object Detection (license plates) |
| **Annotation Format** | YOLO TXT (normalized xywh) |
| **Train Images** | 3,051 (images only — labels/ dir missing in train) |
| **Test Images** | 1,534 |
| **Test Labels** | 1,534 |
| **Classes** | 1: `LicensePlate` |
| **Image Format** | JPG |

**Directory Structure:**
```
indian_number_plate/
└── License Plate Detection/
    ├── data.yaml                 # YOLO config (nc: 1, names: ['LicensePlate'])
    ├── train/
    │   └── images/               # 3,051 JPGs (augmented, .rf. hashes)
    └── test/
        ├── images/               # 1,534 JPGs
        └── labels/               # 1,534 YOLO TXT files
```

**Annotation Sample** (YOLO label):
```
0 0.50390625 0.6875 0.6921875 0.6015625
0 0.50390625 0.18359375 0.90078125 0.33203125
```
Format: `class_id center_x center_y width height` (normalized 0-1)

> **Note:** Train split has images but NO labels directory. Only test split has paired labels. This may indicate the train labels were lost or stored elsewhere.

---

## 9. Autorickshaw Image Dataset

| Property | Value |
|----------|-------|
| **Path** | `data/raw/ind_kaggle/Autorickshaw_Image_Dataset/` |
| **Task** | Object Detection (autorickshaws) |
| **Annotation Format** | Pascal VOC XML |
| **Annotated Images** | 100 (in `Annotations/Annotations/`) |
| **Additional Images** | 100 (in `auto/auto/`, JPG only) |
| **Image Format** | JPG |

**Directory Structure:**
```
Autorickshaw_Image_Dataset/
├── Annotations/
│   └── Annotations/             # 100 Pascal VOC XML files
└── auto/
    └── auto/                    # 100 JPG images (no paired XML here)
```

**Annotation Sample** (`Datacluster Labs Auto (1).xml`):
```xml
<annotation>
  <filename>Datacluster Labs Auto (1).jpg</filename>
  <size><width>4160</width><height>1952</height></size>
  <object>
    <name>autorickshaw</name>
    <bndbox>
      <xmin>2232.12</xmin><ymin>362.98</ymin>
      <xmax>3802.59</xmax><ymax>1682.01</ymax>
    </bndbox>
  </object>
</annotation>
```

---

## 10. Auto Rickshaw DB (Indian Vehicle Dataset)

| Property | Value |
|----------|-------|
| **Path** | `data/raw/ind_kaggle/auto_rickshaw_db/Indian_vehicle_dataset/` |
| **Task** | Object Detection (Indian vehicles) |
| **Annotation Format** | Pascal VOC XML |
| **Total Files** | 200 (100 JPG + 100 XML, paired) |
| **Classes Observed** | `tempo` (plus likely `autorickshaw`, `truck`, `bus`) |
| **Image Format** | JPG |

**Directory Structure:**
```
auto_rickshaw_db/
└── Indian_vehicle_dataset/      # 200 files (100 JPG + 100 XML)
    ├── 20210427_..._4160_3120.jpg
    ├── 20210427_..._4160_3120.xml
    ├── Datacluster Auto (N).jpg/xml
    ├── Datacluster Truck (N).jpg/xml
    ├── dc_auto_image_NNNNNN.jpg/xml
    └── dc_bus_image_NNNNNN.jpg/xml
```

**Annotation Sample:**
```xml
<annotation>
  <filename>20210427_..._4160_3120.jpg</filename>
  <size><width>3120</width><height>4160</height></size>
  <object>
    <name>tempo</name>
    <bndbox>
      <xmin>174.87</xmin><ymin>247.64</ymin>
      <xmax>2861.28</xmax><ymax>2782.4</ymax>
    </bndbox>
  </object>
</annotation>
```

> **Note:** Contains mixed vehicle types: auto-rickshaws, trucks, and buses. Filenames with "Datacluster Auto/Truck" and "dc_bus_image" indicate multiclass data.

---

## 11. Indian Trucks

| Property | Value |
|----------|-------|
| **Path** | `data/raw/ind_kaggle/indian_trucks/` |
| **Task** | Scene/Context only (NO annotations) |
| **Total Images** | 466 |
| **Image Format** | Mixed (JPG, JPEG, PNG, JPG) |

**Directory Structure:**
```
indian_trucks/
└── (flat directory of ~466 images, no annotations subfolder)
```

> **Status:** Images only. No bounding boxes, labels, or annotation files of any kind. Usable only as unlabeled scene/context data or for self-supervised learning.

---

## 12. INDRA (Indian Driving RetinaNet Analysis)

| Property | Value |
|----------|-------|
| **Path** | `data/raw/ind_kaggle/INDRA/` |
| **Task** | Traffic analysis (pre-processed RetinaNet outputs) |
| **Data Format** | NumPy `.npy` arrays + CSV/PKL labels |
| **Array Count** | 104 each for Arrays, Directions, Speeds |
| **Label Files** | `labels_framewise_csv.csv`, `labels_secondwise.csv`, + PKL variants |

**Directory Structure:**
```
INDRA/
├── Arrays_RetinaNet/
│   └── Arrays_RetinaNet/        # 104 .npy files (detection arrays)
├── Directions_RetinaNet/        # 104 .npy files (direction predictions)
├── Speeds_RetinaNet/            # 104 .npy files (speed predictions)
├── Videos/                      # (appears empty)
├── saved_models_ML/             # pre-trained models
├── labels_framewise_csv.csv     # 104 lines, frame-level labels
├── labels_framewise_list.pkl
├── labels_secondwise.csv        # second-level aggregated labels
└── labels_secondwise.pkl
```

> **Status:** This is NOT a raw image/annotation dataset. It contains pre-processed RetinaNet detection outputs (numpy arrays) and aggregated traffic flow labels. Not directly usable for training detection/recognition models, but potentially useful for traffic analysis downstream tasks.

---

## Summary by Task

### Text Detection (bounding box annotations)
| Dataset | Format | Scripts | Geometry | Count |
|---------|--------|---------|----------|-------|
| ICDAR 2015 | ICDAR TXT | Latin | Quad-point | 1,000 train + 500 test |
| Total-Text | MATLAB .mat | Latin | Polygon (curved) | ~1,555 train + ~300 test |
| CTW1500 | Custom XML | Latin + Chinese | Polygon (curved) | 1,000 |
| Indic Scene | Tab-sep TXT | 12 Indic scripts | Quad-point | varies per language |

### Text Recognition (word-level transcription)
| Dataset | Format | Scripts | Count |
|---------|--------|---------|-------|
| MJSynth | Path+index TXT | Latin (synthetic) | 7,224,612 word crops |
| IIIT5K | MATLAB .mat | Latin | 2,000 train + 3,000 test |
| ICDAR 2015 | (same as detection) | Latin | (included in detection GT) |
| CTW1500 | (same as detection) | Latin + Chinese | (included in detection GT) |
| Indic Scene | (same as detection) | 12 Indic scripts | (included in detection GT) |

### Object Detection (non-text)
| Dataset | Format | Classes | Count |
|---------|--------|---------|-------|
| Traffic Signs | Pascal VOC XML | `traffic_sign` | 150 |
| Number Plates | YOLO TXT | `LicensePlate` | 3,051 train + 1,534 test |
| Autorickshaw | Pascal VOC XML | `autorickshaw` | 100 annotated |
| Auto Rickshaw DB | Pascal VOC XML | `tempo`, `truck`, `bus` | 100 pairs |

### Unannotated / Pre-processed
| Dataset | Type | Count |
|---------|------|-------|
| Indian Trucks | Images only (no annotations) | 466 images |
| INDRA | Pre-processed RetinaNet arrays | 104 numpy arrays × 3 |

---

## Key Issues & Notes

1. **MJSynth incomplete download:** `mjsynth.tar.gz.aria2` indicates the archive download was interrupted. Verify data completeness.
2. **Number plate train labels missing:** The `train/` split only has `images/` but no `labels/` directory. Only `test/` has paired labels.
3. **Indian trucks has no annotations:** 466 images with zero annotation files.
4. **INDRA is not trainable:** Contains only pre-processed numpy arrays, not raw images suitable for model training.
5. **Autorickshaw datasets overlap:** `Autorickshaw_Image_Dataset` and `auto_rickshaw_db` share Datacluster naming conventions and INDRA-timestamped images — likely from the same source, split differently.
6. **Total-Text `.mat` format:** Requires MATLAB or `scipy.io.loadmat()` to parse. Not as easily scriptable as TXT/XML/JSON formats.
7. **Synthtext directory exists but was not explored:** `data/raw/synthtext/` and `data/raw/idd/` appear in the workspace but were not included in this audit scope.
