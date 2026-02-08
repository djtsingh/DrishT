"""
Load Total-Text into FiftyOne.
Expects:
  data/raw/totaltext/Images/Train/*.jpg
  data/raw/totaltext/Images/Test/*.jpg
  data/raw/totaltext/gt_latest/  — .mat files (gt_img*.mat) with polygon annotations

Each .mat file contains a 'polygt' array with rows:
  [x1,y1, ..., xN,yN, transcription]
"""

import re
from pathlib import Path
import numpy as np
import scipy.io as sio
import fiftyone as fo

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "totaltext"
GT_DIR = DATA_ROOT / "gt_latest"


def parse_mat_gt(mat_path):
    """Parse a Total-Text .mat groundtruth file.
    Returns list of (polygon_coords_abs, transcription).
    """
    mat = sio.loadmat(str(mat_path))
    detections = []
    # Key is usually 'polygt'
    key = None
    for k in mat:
        if not k.startswith("__"):
            key = k
            break
    if key is None:
        return detections

    data = mat[key]
    for row in data:
        # Each row: array of objects — coords then text
        try:
            # Coordinates: first elements are x,y pairs
            x_coords = np.array(row[1], dtype=float).flatten()
            y_coords = np.array(row[3], dtype=float).flatten()
            text = str(row[4].flat[0]) if len(row) > 4 else ""
            coords = list(zip(x_coords.tolist(), y_coords.tolist()))
            if coords:
                detections.append((coords, text))
        except (IndexError, ValueError, TypeError):
            # Fallback: try treating whole row as flat array
            try:
                flat = np.array(row, dtype=float).flatten()
                n = len(flat)
                if n >= 4:
                    coords = [(flat[i], flat[i + 1]) for i in range(0, n - 1, 2)]
                    detections.append((coords, ""))
            except (ValueError, TypeError):
                continue
    return detections


def _poly_to_rel(coords, img_w, img_h):
    pts = [[x / img_w, y / img_h] for x, y in coords]
    pts.append(pts[0])
    return pts


def load_split(split="Train"):
    img_dir = DATA_ROOT / "Images" / split
    if not img_dir.exists():
        raise FileNotFoundError(f"{img_dir} not found.")

    from PIL import Image
    samples = []
    for img_path in sorted(img_dir.glob("*.*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        sample = fo.Sample(filepath=str(img_path))
        # Match gt_img<num>.mat to img<num>.jpg
        gt_name = f"gt_{img_path.stem}.mat"
        gt_path = GT_DIR / gt_name
        if gt_path.exists():
            im = Image.open(img_path)
            w, h = im.size
            polylines = []
            for coords, text in parse_mat_gt(gt_path):
                pts = _poly_to_rel(coords, w, h)
                polylines.append(
                    fo.Polyline(points=[pts], closed=True, filled=True, label="text")
                )
            sample["gt_polylines"] = fo.Polylines(polylines=polylines)
        samples.append(sample)
    return samples


def create_dataset(name="totaltext"):
    dataset = fo.Dataset(name, overwrite=True)
    for split in ("Train", "Test"):
        try:
            tag = split.lower()
            samples = load_split(split)
            dataset.add_samples(samples, tags=[tag])
            print(f"  Added {len(samples)} {tag} samples")
        except FileNotFoundError as e:
            print(f"  Skipping {split}: {e}")
    dataset.persistent = True
    print(f"Dataset '{name}' created with {len(dataset)} samples.")
    return dataset


if __name__ == "__main__":
    ds = create_dataset()
    session = fo.launch_app(ds)
    session.wait()
