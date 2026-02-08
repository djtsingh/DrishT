"""
Load ICDAR 2015 into FiftyOne.
Expects:
  data/raw/icdar2015/train_images/*.jpg
  data/raw/icdar2015/train_gt/gt_img_*.txt
  (same for test)

ICDAR 2015 GT format per line:
  x1,y1,x2,y2,x3,y3,x4,y4,transcription
"""

import os, csv, glob
from pathlib import Path
import fiftyone as fo

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "icdar2015"

def parse_icdar_gt(gt_path):
    """Return list of (polygon_points, transcription) from an ICDAR gt file."""
    detections = []
    with open(gt_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 9:
                continue
            coords = list(map(int, row[:8]))
            text = ",".join(row[8:]).strip()
            # Normalise polygon to relative coords later (need image size)
            detections.append((coords, text))
    return detections


def _poly_to_rel(coords, img_w, img_h):
    """Convert 8 absolute coords to FiftyOne polyline points [[x,y], ...]."""
    pts = []
    for i in range(0, 8, 2):
        pts.append([coords[i] / img_w, coords[i + 1] / img_h])
    pts.append(pts[0])  # close polygon
    return pts


def load_split(split="train"):
    img_dir = DATA_ROOT / f"{split}_images"
    gt_dir = DATA_ROOT / f"{split}_gt"
    if not img_dir.exists():
        raise FileNotFoundError(f"{img_dir} not found. Download ICDAR 2015 first.")

    samples = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        gt_name = f"gt_{img_path.stem}.txt"
        gt_path = gt_dir / gt_name
        sample = fo.Sample(filepath=str(img_path))

        if gt_path.exists():
            from PIL import Image
            im = Image.open(img_path)
            w, h = im.size
            polylines = []
            for coords, text in parse_icdar_gt(gt_path):
                is_illegible = text == "###"
                pts = _poly_to_rel(coords, w, h)
                polylines.append(
                    fo.Polyline(
                        points=[pts],
                        closed=True,
                        filled=True,
                        label="illegible" if is_illegible else "text",
                    )
                )
                sample["transcription"] = text if not is_illegible else None
            sample["gt_polylines"] = fo.Polylines(polylines=polylines)
        samples.append(sample)
    return samples


def create_dataset(name="icdar2015"):
    dataset = fo.Dataset(name, overwrite=True)
    for split in ("train", "test"):
        try:
            samples = load_split(split)
            dataset.add_samples(samples, tags=[split])
            print(f"  Added {len(samples)} {split} samples")
        except FileNotFoundError as e:
            print(f"  Skipping {split}: {e}")
    dataset.persistent = True
    print(f"Dataset '{name}' created with {len(dataset)} samples.")
    return dataset


if __name__ == "__main__":
    ds = create_dataset()
    session = fo.launch_app(ds)
    session.wait()
