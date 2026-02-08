"""
Unified Recognition Dataset Builder
=====================================
Collects all word-level cropped images + transcription labels into a single
dataset ready for CRNN (BiLSTM + CTC) fine-tuning.

Sources:
  1. MJSynth    – 7M+ synthetic word crops (Latin)
  2. IIIT5K     – 5K real word crops (Latin)
  3. Indic Scene – cropped word images per script (12 Indic scripts)
  4. Crops from detection datasets (ICDAR2015, Total-Text, CTW1500, Indic Scene)

Output: data/processed/recognition/
        ├── images/          (all word crop images, renamed uniquely)
        ├── labels.csv       (image_name, label, script, source)
        └── stats.json
"""
import os
import sys
import csv
import json
import shutil
import random
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW = Path("data/raw")
OUT = Path("data/processed/recognition")
OUT_IMAGES = OUT / "images"
VAL_RATIO = 0.10
SEED = 42
MAX_MJSYNTH = 100_000  # Cap MJSynth — 100K covers the full 90K vocabulary with
                       # redundancy for font/style diversity. More than enough for
                       # Latin coverage without overwhelming the Indic data.

# Recognition-focused: min word crop size
MIN_CROP_W = 16
MIN_CROP_H = 8
MAX_LABEL_LEN = 50  # skip absurdly long labels

random.seed(SEED)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
rec_counter = 0

def next_rec_id():
    global rec_counter
    rec_counter += 1
    return rec_counter


def is_valid_label(text):
    """Check if label is valid for recognition training."""
    if not text or text.strip() == "":
        return False
    if text in ("###", "#", "?", ""):
        return False
    if len(text) > MAX_LABEL_LEN:
        return False
    return True


def copy_crop(src_path, dest_name):
    """Copy a word crop image, return (width, height) or None."""
    dst = OUT_IMAGES / dest_name
    if dst.exists():
        return True
    try:
        shutil.copy2(src_path, dst)
        return True
    except Exception:
        return None


def crop_from_image(img_path, pts, dest_name):
    """Crop a text region from a scene image using bounding rect of polygon."""
    try:
        with Image.open(img_path) as im:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1 = max(0, int(min(xs)))
            y1 = max(0, int(min(ys)))
            x2 = min(im.width, int(max(xs)))
            y2 = min(im.height, int(max(ys)))
            if x2 - x1 < MIN_CROP_W or y2 - y1 < MIN_CROP_H:
                return None
            crop = im.crop((x1, y1, x2, y2))
            dst = OUT_IMAGES / dest_name
            crop.save(dst, quality=95)
            return True
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Dataset Parsers — each returns list of (dest_filename, label, script, source)
# ---------------------------------------------------------------------------

def parse_mjsynth():
    """Parse MJSynth synthetic word recognition dataset.
    
    Scans disk files directly instead of using annotation files,
    since partial tar extraction means annotation paths don't match
    the directories actually present on disk.
    Filename pattern: NNN_WORD_NNNNN.jpg -> extract WORD as label.
    """
    records = []
    base = RAW / "mjsynth" / "mnt" / "ramdisk" / "max" / "90kDICT32px"
    if not base.exists():
        print("  [MJSynth] base directory not found, skipping")
        return records

    # Scan all jpg files on disk directly
    print("  [MJSynth] Scanning disk for image files (this may take a while)...")
    all_images = list(base.glob("**/*.jpg"))
    print(f"  [MJSynth] Found {len(all_images):,} images on disk")

    if not all_images:
        print("  [MJSynth] No images found on disk, skipping")
        return records

    # Shuffle and cap
    random.shuffle(all_images)
    all_images = all_images[:MAX_MJSYNTH]

    skipped_no_match = 0
    skipped_invalid = 0
    for img_path in tqdm(all_images, desc="MJSynth"):
        # Extract word from filename: 1_WORD_12345.jpg -> WORD
        fname = img_path.stem
        match = re.match(r"\d+_(.+)_\d+$", fname)
        if match:
            word = match.group(1)
        else:
            skipped_no_match += 1
            continue

        if not is_valid_label(word):
            skipped_invalid += 1
            continue

        rid = next_rec_id()
        dest_name = f"mjsynth_{rid}.jpg"
        if copy_crop(img_path, dest_name):
            records.append((dest_name, word, "latin", "mjsynth"))

    print(f"  [MJSynth] Processed: {len(records):,} valid, "
          f"{skipped_no_match:,} no-match filenames, "
          f"{skipped_invalid:,} invalid labels")
    return records


def parse_mjsynth_val():
    """Parse MJSynth validation set (smaller, for balanced eval).
    
    Uses annotation_val.txt if available AND matching disk files exist,
    otherwise falls back to sampling from disk files.
    """
    records = []
    base = RAW / "mjsynth" / "mnt" / "ramdisk" / "max" / "90kDICT32px"
    if not base.exists():
        return records

    VAL_CAP = 50_000

    # Try annotation file first
    ann_file = base / "annotation_val.txt"
    ann_found = 0
    if ann_file.exists():
        lines = ann_file.read_text(encoding="utf-8", errors="ignore").strip().split("\n")
        random.shuffle(lines)
        for line in tqdm(lines[:VAL_CAP * 2], desc="MJSynth-val(ann)"):
            parts = line.strip().split(" ", 1)
            if not parts:
                continue
            rel_path = parts[0].lstrip("./")
            img_path = base / rel_path
            if not img_path.exists():
                continue
            fname = img_path.stem
            match = re.match(r"\d+_(.+)_\d+$", fname)
            if match:
                word = match.group(1)
            else:
                continue
            if not is_valid_label(word):
                continue
            rid = next_rec_id()
            dest_name = f"mjsynth_val_{rid}.jpg"
            if copy_crop(img_path, dest_name):
                records.append((dest_name, word, "latin", "mjsynth_val"))
                ann_found += 1
            if ann_found >= VAL_CAP:
                break

    # If annotation-based approach found too few, scan disk directly
    if ann_found < 1000:
        print(f"  [MJSynth-val] Annotation found {ann_found}, falling back to disk scan")
        records = []  # Reset
        all_images = list(base.glob("**/*.jpg"))
        random.shuffle(all_images)
        # Use a separate portion from what train might use
        # Take from the end of the shuffled list
        val_images = all_images[-min(VAL_CAP * 2, len(all_images)):][:VAL_CAP]
        for img_path in tqdm(val_images, desc="MJSynth-val(disk)"):
            fname = img_path.stem
            match = re.match(r"\d+_(.+)_\d+$", fname)
            if match:
                word = match.group(1)
            else:
                continue
            if not is_valid_label(word):
                continue
            rid = next_rec_id()
            dest_name = f"mjsynth_val_{rid}.jpg"
            if copy_crop(img_path, dest_name):
                records.append((dest_name, word, "latin", "mjsynth_val"))

    print(f"  [MJSynth-val] Total: {len(records):,} crops")
    return records


def parse_iiit5k():
    """Parse IIIT 5K-Word recognition dataset."""
    records = []
    try:
        from scipy.io import loadmat
    except ImportError:
        print("  [IIIT5K] scipy not installed, skipping")
        return records

    base = RAW / "iiit5k" / "IIIT5K"
    if not base.exists():
        base = RAW / "iiit5k"

    for split, mat_name, img_subdir in [
        ("train", "traindata.mat", "train"),
        ("test",  "testdata.mat",  "test"),
    ]:
        mat_path = base / mat_name
        img_dir = base / img_subdir
        if not mat_path.exists():
            # Try charbound mat
            mat_path = base / f"{split}CharBound.mat"
        if not mat_path.exists() or not img_dir.exists():
            print(f"  [IIIT5K] skip {split}: {mat_path.exists()}, {img_dir.exists()}")
            continue

        try:
            mat = loadmat(str(mat_path))
        except Exception as e:
            print(f"  [IIIT5K] Error loading {mat_path}: {e}")
            continue

        # Try different keys
        data_key = None
        for key in [f"{split}data", f"{split}CharBound", "data"]:
            if key in mat:
                data_key = key
                break
        if data_key is None:
            # Try first non-underscore key
            for k in mat:
                if not k.startswith("_"):
                    data_key = k
                    break
        if data_key is None:
            print(f"  [IIIT5K] no data key found in {mat_path.name}: {list(mat.keys())}")
            continue

        entries = mat[data_key].flatten()
        for entry in tqdm(entries, desc=f"IIIT5K-{split}"):
            try:
                # IIIT5K mat format: entry is struct with ImgName, GroundTruth
                if hasattr(entry, 'dtype') and entry.dtype.names:
                    img_name = str(entry['ImgName'].flat[0]).strip()
                    word = str(entry['GroundTruth'].flat[0]).strip()
                else:
                    # Array access
                    img_name = str(entry[0][0]).strip()
                    word = str(entry[1][0]).strip()
            except (IndexError, KeyError, TypeError):
                continue

            if not is_valid_label(word):
                continue

            img_path = img_dir / img_name
            if not img_path.exists():
                # Try just the filename
                img_path = img_dir / Path(img_name).name
            if not img_path.exists():
                continue

            rid = next_rec_id()
            dest_name = f"iiit5k_{split}_{rid}{img_path.suffix}"
            if copy_crop(img_path, dest_name):
                records.append((dest_name, word, "latin", f"iiit5k_{split}"))

    # Fallback: glob all images and try to use filenames
    if not records:
        print("  [IIIT5K] Mat parsing produced 0 records, trying glob fallback")
        for split in ["train", "test"]:
            img_dir = base / split
            if not img_dir.exists():
                continue
            for img_path in tqdm(sorted(img_dir.glob("*.*")), desc=f"IIIT5K-{split}-glob"):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                    continue
                rid = next_rec_id()
                dest_name = f"iiit5k_{split}_{rid}{img_path.suffix}"
                # Label unknown from glob - mark as "UNK"
                if copy_crop(img_path, dest_name):
                    records.append((dest_name, "UNK", "latin", f"iiit5k_{split}"))

    return records


def parse_indic_scene_crops():
    """Parse Indic Scene Text cropped word images."""
    records = []
    base = RAW / "indic_scene" / "verified_twice"
    if not base.exists():
        base = RAW / "indic_scene"

    scripts = [d for d in base.iterdir() if d.is_dir()]
    for script_dir in sorted(scripts):
        script_name = script_dir.name
        crops_dir = script_dir / "cropped_images"
        if not crops_dir.exists():
            # Maybe cropped images are directly in script dir
            continue

        # Parse GT files to get transcriptions for crops
        gt_map = {}  # img_stem -> list of (index, transcription)
        for gt_file in script_dir.glob("*_gt.txt"):
            stem = gt_file.stem.replace("_gt", "")
            try:
                lines = gt_file.read_text(encoding="utf-8").strip().split("\n")
                for line in lines:
                    parts = line.split("\t")
                    if len(parts) >= 10:
                        idx = parts[0].strip()
                        text = parts[9].strip()
                        gt_map[f"{stem}_{idx}"] = text
            except Exception:
                continue

        # Process cropped images
        crop_files = sorted(crops_dir.glob("*.*"))
        for crop_path in tqdm(crop_files, desc=f"Indic-{script_name}-crops"):
            if crop_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue

            # Try to match to GT
            crop_stem = crop_path.stem
            transcription = gt_map.get(crop_stem, "")

            # Try alternative matching patterns
            if not transcription:
                # Pattern: 1_1.jpg -> image 1, text region 1
                match = re.match(r"(\d+)_(\d+)", crop_stem)
                if match:
                    key = f"{match.group(1)}_{match.group(2)}"
                    transcription = gt_map.get(key, "")

            if not is_valid_label(transcription):
                # Still include with UNK for potential self-supervised use
                transcription = "UNK"

            rid = next_rec_id()
            dest_name = f"indic_{script_name}_{rid}{crop_path.suffix}"
            if copy_crop(crop_path, dest_name):
                records.append((dest_name, transcription, script_name, "indic_scene"))

    return records


def parse_icdar2015_crops():
    """Crop text regions from ICDAR 2015 scene images."""
    records = []
    for split, img_dir, gt_dir in [
        ("train", "ch4_training_images", "ch4_training_localization_transcription_gt"),
        ("test",  "ch4_test_images",     "Challenge4_Test_Task1_GT"),
    ]:
        img_root = RAW / "icdar2015" / img_dir
        gt_root = RAW / "icdar2015" / gt_dir
        if not img_root.exists() or not gt_root.exists():
            continue
        gt_files = sorted(gt_root.glob("*.txt"))
        for gt_file in tqdm(gt_files, desc=f"ICDAR2015-crops-{split}"):
            stem = gt_file.stem.replace("gt_", "")
            img_path = None
            for ext in (".jpg", ".JPG", ".jpeg", ".png"):
                candidate = img_root / (stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                continue

            try:
                lines = gt_file.read_text(encoding="utf-8-sig").strip().split("\n")
            except Exception:
                lines = gt_file.read_text(encoding="latin-1").strip().split("\n")

            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",", 8)
                if len(parts) < 9:
                    continue
                try:
                    coords = [int(p) for p in parts[:8]]
                except ValueError:
                    continue
                transcription = parts[8].strip()
                if not is_valid_label(transcription):
                    continue

                pts = [(coords[j], coords[j+1]) for j in range(0, 8, 2)]
                rid = next_rec_id()
                dest_name = f"icdar15_crop_{split}_{rid}.jpg"
                if crop_from_image(img_path, pts, dest_name):
                    records.append((dest_name, transcription, "latin", "icdar2015_crop"))

    return records


def parse_indic_scene_crops_from_gt():
    """Crop text regions from Indic Scene full images using GT coordinates."""
    records = []
    base = RAW / "indic_scene" / "verified_twice"
    if not base.exists():
        base = RAW / "indic_scene"

    scripts = [d for d in base.iterdir() if d.is_dir() and d.name != "cropped_images"]
    for script_dir in sorted(scripts):
        script_name = script_dir.name
        gt_files = sorted(script_dir.glob("*_gt.txt"))
        for gt_file in tqdm(gt_files, desc=f"IndicCrop-{script_name}"):
            stem = gt_file.stem.replace("_gt", "")
            img_path = None
            for ext in (".jpeg", ".jpg", ".png", ".JPEG"):
                candidate = script_dir / (stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                continue

            try:
                lines = gt_file.read_text(encoding="utf-8").strip().split("\n")
            except Exception:
                continue

            for line in lines:
                parts = line.split("\t")
                if len(parts) < 10:
                    continue
                try:
                    coords = [float(parts[i]) for i in range(1, 9)]
                    transcription = parts[9].strip()
                except (ValueError, IndexError):
                    continue
                if not is_valid_label(transcription):
                    continue

                pts = [(coords[j], coords[j+1]) for j in range(0, 8, 2)]
                rid = next_rec_id()
                dest_name = f"indic_crop_{script_name}_{rid}.jpg"
                if crop_from_image(img_path, pts, dest_name):
                    records.append((dest_name, transcription, script_name, "indic_scene_crop"))

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Unified Recognition Dataset Builder")
    print("=" * 60)

    OUT_IMAGES.mkdir(parents=True, exist_ok=True)

    all_records = []
    stats = defaultdict(lambda: defaultdict(int))

    parsers = [
        ("MJSynth (train)",       parse_mjsynth),
        ("MJSynth (val)",         parse_mjsynth_val),
        ("IIIT5K",                parse_iiit5k),
        ("Indic Scene Crops",     parse_indic_scene_crops),
        ("Indic Scene GT Crops",  parse_indic_scene_crops_from_gt),
        ("ICDAR2015 Crops",       parse_icdar2015_crops),
    ]

    for name, parser_fn in parsers:
        print(f"\n{'─'*40}")
        print(f"  Parsing: {name}")
        print(f"{'─'*40}")
        try:
            records = parser_fn()
            all_records.extend(records)
            n = len(records)
            # Count by script
            script_counts = defaultdict(int)
            for _, _, script, _ in records:
                script_counts[script] += 1
            stats[name] = dict(script_counts)
            print(f"  ✓ {name}: {n} word crops")
            for sc, cnt in sorted(script_counts.items()):
                print(f"      {sc}: {cnt}")
        except Exception as e:
            print(f"  ✗ {name}: ERROR - {e}")
            import traceback
            traceback.print_exc()

    # --- Shuffle and split ---
    print(f"\n{'='*60}")
    print(f"  Total: {len(all_records)} word crops")
    print(f"{'='*60}")

    random.shuffle(all_records)
    n_val = max(1, int(len(all_records) * VAL_RATIO))
    val_records = all_records[:n_val]
    train_records = all_records[n_val:]

    print(f"  Train: {len(train_records)} word crops")
    print(f"  Val:   {len(val_records)} word crops")

    # --- Write labels CSV ---
    for split_name, records in [("train", train_records), ("val", val_records)]:
        csv_path = OUT / f"labels_{split_name}.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "label", "script", "source"])
            for dest_name, label, script, source in records:
                writer.writerow([dest_name, label, script, source])
        print(f"  Saved: {csv_path}")

    # Also write combined labels
    csv_path = OUT / "labels_all.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label", "script", "source"])
        for dest_name, label, script, source in all_records:
            writer.writerow([dest_name, label, script, source])
    print(f"  Saved: {csv_path}")

    # --- Build charset ---
    all_chars = set()
    for _, label, _, _ in all_records:
        if label != "UNK":
            all_chars.update(label)
    charset = sorted(all_chars)
    charset_path = OUT / "charset.txt"
    with open(charset_path, "w", encoding="utf-8") as f:
        for ch in charset:
            f.write(ch + "\n")
    print(f"  Charset: {len(charset)} unique characters -> {charset_path}")

    # --- Script distribution ---
    script_dist = defaultdict(int)
    source_dist = defaultdict(int)
    for _, label, script, source in all_records:
        script_dist[script] += 1
        source_dist[source] += 1

    # --- Save stats ---
    stats_out = {
        "total_crops": len(all_records),
        "train_crops": len(train_records),
        "val_crops": len(val_records),
        "charset_size": len(charset),
        "script_distribution": dict(sorted(script_dist.items(), key=lambda x: -x[1])),
        "source_distribution": dict(sorted(source_dist.items(), key=lambda x: -x[1])),
        "per_parser": {k: dict(v) for k, v in stats.items()},
        "unk_labels": sum(1 for _, l, _, _ in all_records if l == "UNK"),
    }
    stats_path = OUT / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_out, f, indent=2, ensure_ascii=False)
    print(f"  Stats: {stats_path}")

    # --- Print summary ---
    print(f"\n{'='*60}")
    print("  Recognition Dataset Summary")
    print(f"{'='*60}")
    print(f"  {'Script':<20} {'Count':>10}")
    print(f"  {'─'*35}")
    for sc, cnt in sorted(script_dist.items(), key=lambda x: -x[1]):
        print(f"  {sc:<20} {cnt:>10}")
    print(f"  {'─'*35}")
    print(f"  {'TOTAL':<20} {len(all_records):>10}")
    print(f"  UNK labels: {stats_out['unk_labels']}")
    print(f"\n  Output: {OUT}")
    print("\n  Done!\n")


if __name__ == "__main__":
    main()
