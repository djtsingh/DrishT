"""
Data Quality Pipeline & Final Export
======================================
1. Validates detection COCO JSON  – checks image existence, bbox sanity
2. Validates recognition labels   – checks image existence, label format
3. Merges synthetic data into recognition set
4. Creates final train/val/test splits with stratified sampling
5. Generates comprehensive statistics + visualizations

Output: data/final/detection/    (train/val/test COCO JSONs + images)
        data/final/recognition/  (train/val/test labels + images)
        data/final/stats/        (reports + charts)
"""
import os
import sys
import csv
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict, Counter

from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DET_DIR = Path("data/processed/detection")
REC_DIR = Path("data/processed/recognition")
SYN_DIR = Path("data/processed/synthetic")
FINAL = Path("data/final")
FINAL_DET = FINAL / "detection"
FINAL_REC = FINAL / "recognition"
FINAL_STATS = FINAL / "stats"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

SEED = 42
random.seed(SEED)


# ---------------------------------------------------------------------------
# Detection QC
# ---------------------------------------------------------------------------
def validate_detection():
    """Validate detection COCO JSONs and filter bad entries."""
    print("\n" + "=" * 60)
    print("  Detection Dataset Quality Check")
    print("=" * 60)

    results = {}
    for split in ["train", "val"]:
        json_path = DET_DIR / f"coco_{split}.json"
        if not json_path.exists():
            print(f"  SKIP: {json_path} not found")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        images = coco["images"]
        annotations = coco["annotations"]
        categories = coco["categories"]

        print(f"\n  [{split}] Images: {len(images)}, Annotations: {len(annotations)}")

        # Check image existence
        img_dir = DET_DIR / "images"
        valid_img_ids = set()
        missing_images = 0
        corrupt_images = 0

        for img_rec in tqdm(images, desc=f"  Check images ({split})"):
            img_path = img_dir / img_rec["file_name"]
            if not img_path.exists():
                missing_images += 1
                continue
            try:
                with Image.open(img_path) as im:
                    im.verify()
                valid_img_ids.add(img_rec["id"])
            except Exception:
                corrupt_images += 1

        # Filter annotations to valid images
        valid_anns = []
        bbox_issues = 0
        for ann in annotations:
            if ann["image_id"] not in valid_img_ids:
                continue
            bbox = ann["bbox"]
            # Sanity check bbox
            if bbox[2] <= 0 or bbox[3] <= 0:
                bbox_issues += 1
                continue
            if bbox[0] < 0:
                ann["bbox"][0] = 0
            if bbox[1] < 0:
                ann["bbox"][1] = 0
            valid_anns.append(ann)

        # Filter images to those with at least one annotation
        ann_img_ids = set(a["image_id"] for a in valid_anns)
        valid_images = [img for img in images if img["id"] in ann_img_ids]

        print(f"  Missing images: {missing_images}")
        print(f"  Corrupt images: {corrupt_images}")
        print(f"  Bbox issues: {bbox_issues}")
        print(f"  Valid: {len(valid_images)} images, {len(valid_anns)} annotations")

        # Category distribution
        cat_id_to_name = {c["id"]: c["name"] for c in categories}
        cat_dist = Counter(cat_id_to_name.get(a["category_id"], "unknown") for a in valid_anns)
        print(f"  Category distribution:")
        for cat, cnt in cat_dist.most_common():
            print(f"    {cat}: {cnt}")

        results[split] = {
            "images": valid_images,
            "annotations": valid_anns,
            "categories": categories,
        }

    return results


def validate_recognition():
    """Validate recognition dataset and merge with synthetic data."""
    print("\n" + "=" * 60)
    print("  Recognition Dataset Quality Check")
    print("=" * 60)

    all_records = []

    # Load real data
    for split in ["train", "val"]:
        csv_path = REC_DIR / f"labels_{split}.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_records.append(row)
    print(f"  Real data: {len(all_records)} records")

    # Load synthetic data
    syn_csv = SYN_DIR / "labels.csv"
    syn_count = 0
    if syn_csv.exists():
        with open(syn_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["image_dir"] = "synthetic"  # mark source dir
                all_records.append(row)
                syn_count += 1
        print(f"  Synthetic data: {syn_count} records")

    # Validate images exist
    valid_records = []
    missing = 0
    for rec in tqdm(all_records, desc="  Validate recognition images"):
        img_name = rec["image"]
        # Check both directories
        img_path = REC_DIR / "images" / img_name
        if not img_path.exists():
            img_path = SYN_DIR / "images" / img_name
        if not img_path.exists():
            missing += 1
            continue
        rec["_full_path"] = str(img_path)
        valid_records.append(rec)

    print(f"  Missing images: {missing}")
    print(f"  Valid records: {len(valid_records)}")

    # Label quality
    unk_count = sum(1 for r in valid_records if r["label"] == "UNK")
    empty_count = sum(1 for r in valid_records if not r["label"].strip())
    print(f"  UNK labels: {unk_count}")
    print(f"  Empty labels: {empty_count}")

    # Script distribution
    script_dist = Counter(r.get("script", "unknown") for r in valid_records)
    print(f"  Script distribution:")
    for sc, cnt in script_dist.most_common():
        print(f"    {sc}: {cnt}")

    return valid_records


# ---------------------------------------------------------------------------
# Final Export
# ---------------------------------------------------------------------------
def export_detection(det_data):
    """Create final train/val/test detection splits."""
    print("\n" + "=" * 60)
    print("  Exporting Final Detection Dataset")
    print("=" * 60)

    if not det_data:
        print("  No detection data to export")
        return {}

    # Combine all splits
    all_images = []
    all_anns = []
    categories = None
    for split, data in det_data.items():
        all_images.extend(data["images"])
        all_anns.extend(data["annotations"])
        categories = data["categories"]

    if not all_images:
        print("  No valid images")
        return {}

    # Create image_id -> annotations mapping
    img_id_to_anns = defaultdict(list)
    for ann in all_anns:
        img_id_to_anns[ann["image_id"]].append(ann)

    # Stratified split by dataset source
    dataset_groups = defaultdict(list)
    for img in all_images:
        ds = img.get("dataset", "unknown")
        dataset_groups[ds].append(img)

    train_images, val_images, test_images = [], [], []
    for ds, imgs in dataset_groups.items():
        random.shuffle(imgs)
        n = len(imgs)
        n_test = max(1, int(n * TEST_RATIO))
        n_val = max(1, int(n * VAL_RATIO))
        test_images.extend(imgs[:n_test])
        val_images.extend(imgs[n_test:n_test + n_val])
        train_images.extend(imgs[n_test + n_val:])

    print(f"  Train: {len(train_images)} images")
    print(f"  Val:   {len(val_images)} images")
    print(f"  Test:  {len(test_images)} images")

    # Export each split
    det_img_dir = DET_DIR / "images"
    stats = {}
    for split_name, split_imgs in [("train", train_images), ("val", val_images), ("test", test_images)]:
        out_dir = FINAL_DET / split_name / "images"
        out_dir.mkdir(parents=True, exist_ok=True)

        split_img_ids = set(img["id"] for img in split_imgs)
        split_anns = [a for a in all_anns if a["image_id"] in split_img_ids]

        # Copy images
        for img in tqdm(split_imgs, desc=f"  Copy {split_name} images"):
            src = det_img_dir / img["file_name"]
            dst = out_dir / img["file_name"]
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

        coco = {
            "images": split_imgs,
            "annotations": split_anns,
            "categories": categories,
            "info": {"description": f"DrishT Detection {split_name}", "version": "1.0"},
        }
        json_path = FINAL_DET / split_name / f"annotations.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False, indent=2)

        stats[split_name] = {"images": len(split_imgs), "annotations": len(split_anns)}
        print(f"  Saved: {json_path}")

    return stats


def export_recognition(rec_records):
    """Create final train/val/test recognition splits."""
    print("\n" + "=" * 60)
    print("  Exporting Final Recognition Dataset")
    print("=" * 60)

    if not rec_records:
        print("  No recognition data to export")
        return {}

    # Stratified split by script
    script_groups = defaultdict(list)
    for rec in rec_records:
        sc = rec.get("script", "unknown")
        script_groups[sc].append(rec)

    train_recs, val_recs, test_recs = [], [], []
    for sc, recs in script_groups.items():
        random.shuffle(recs)
        n = len(recs)
        n_test = max(1, int(n * TEST_RATIO))
        n_val = max(1, int(n * VAL_RATIO))
        test_recs.extend(recs[:n_test])
        val_recs.extend(recs[n_test:n_test + n_val])
        train_recs.extend(recs[n_test + n_val:])

    print(f"  Train: {len(train_recs)} word crops")
    print(f"  Val:   {len(val_recs)} word crops")
    print(f"  Test:  {len(test_recs)} word crops")

    stats = {}
    for split_name, split_recs in [("train", train_recs), ("val", val_recs), ("test", test_recs)]:
        out_dir = FINAL_REC / split_name / "images"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Copy images and write labels
        csv_path = FINAL_REC / split_name / "labels.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "label", "script", "source"])
            for rec in tqdm(split_recs, desc=f"  Export {split_name}"):
                img_name = rec["image"]
                full_path = rec.get("_full_path", "")
                if full_path and os.path.exists(full_path):
                    dst = out_dir / img_name
                    if not dst.exists():
                        try:
                            shutil.copy2(full_path, dst)
                        except Exception:
                            continue
                writer.writerow([img_name, rec["label"], rec.get("script", ""), rec.get("source", "")])

        stats[split_name] = {
            "total": len(split_recs),
            "scripts": dict(Counter(r.get("script", "unknown") for r in split_recs)),
        }
        print(f"  Saved: {csv_path}")

    # Build combined charset
    all_chars = set()
    for rec in rec_records:
        if rec["label"] not in ("UNK", ""):
            all_chars.update(rec["label"])
    charset = sorted(all_chars)
    charset_path = FINAL_REC / "charset.txt"
    with open(charset_path, "w", encoding="utf-8") as f:
        for ch in charset:
            f.write(ch + "\n")
    print(f"  Charset: {len(charset)} characters -> {charset_path}")

    return stats


# ---------------------------------------------------------------------------
# Summary Report
# ---------------------------------------------------------------------------
def generate_report(det_stats, rec_stats):
    """Generate final summary report."""
    print("\n" + "=" * 60)
    print("  Generating Summary Report")
    print("=" * 60)

    FINAL_STATS.mkdir(parents=True, exist_ok=True)

    report = {
        "detection": det_stats,
        "recognition": rec_stats,
    }

    with open(FINAL_STATS / "final_stats.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Markdown report
    md_lines = [
        "# DrishT Dataset - Final Export Report\n",
        "## Detection Dataset\n",
    ]
    if det_stats:
        md_lines.append("| Split | Images | Annotations |\n")
        md_lines.append("|-------|--------|-------------|\n")
        for split, s in det_stats.items():
            md_lines.append(f"| {split} | {s.get('images', 0)} | {s.get('annotations', 0)} |\n")
    else:
        md_lines.append("No detection data exported.\n")

    md_lines.append("\n## Recognition Dataset\n")
    if rec_stats:
        md_lines.append("| Split | Total |\n")
        md_lines.append("|-------|-------|\n")
        for split, s in rec_stats.items():
            md_lines.append(f"| {split} | {s.get('total', 0)} |\n")
        md_lines.append("\n### Script Distribution (Train)\n")
        if "train" in rec_stats and "scripts" in rec_stats["train"]:
            md_lines.append("| Script | Count |\n")
            md_lines.append("|--------|-------|\n")
            for sc, cnt in sorted(rec_stats["train"]["scripts"].items(), key=lambda x: -x[1]):
                md_lines.append(f"| {sc} | {cnt} |\n")
    else:
        md_lines.append("No recognition data exported.\n")

    md_lines.append("\n## Output Structure\n")
    md_lines.append("```\n")
    md_lines.append("data/final/\n")
    md_lines.append("├── detection/\n")
    md_lines.append("│   ├── train/ (annotations.json + images/)\n")
    md_lines.append("│   ├── val/\n")
    md_lines.append("│   └── test/\n")
    md_lines.append("├── recognition/\n")
    md_lines.append("│   ├── train/ (labels.csv + images/)\n")
    md_lines.append("│   ├── val/\n")
    md_lines.append("│   ├── test/\n")
    md_lines.append("│   └── charset.txt\n")
    md_lines.append("└── stats/\n")
    md_lines.append("    └── final_stats.json\n")
    md_lines.append("```\n")

    md_path = FINAL_STATS / "REPORT.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.writelines(md_lines)
    print(f"  Report: {md_path}")

    # Also copy to project root
    root_report = Path("DATA_EXPORT_REPORT.md")
    shutil.copy2(md_path, root_report)
    print(f"  Also: {root_report}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  DrishT Data Quality & Final Export Pipeline")
    print("=" * 60)

    # 1. Validate detection
    det_data = validate_detection()

    # 2. Validate recognition
    rec_records = validate_recognition()

    # 3. Export detection
    det_stats = export_detection(det_data)

    # 4. Export recognition
    rec_stats = export_recognition(rec_records)

    # 5. Generate report
    generate_report(det_stats, rec_stats)

    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)
    print(f"\n  Final datasets: {FINAL}")
    print("  Ready for fine-tuning on Google Cloud AI Platform.\n")


if __name__ == "__main__":
    main()
