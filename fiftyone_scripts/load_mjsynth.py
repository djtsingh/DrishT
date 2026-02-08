"""
Load MJSynth (Synth90k) into FiftyOne.
Expects:
  data/raw/mjsynth/mnt/ramdisk/max/90kDICT32px/   (standard extraction path)
    annotation_train.txt
    annotation_val.txt
    annotation_test.txt
  Each line: ./path/to/img.jpg label_index

Word label is encoded in the filename: e.g. 1_HOUSE_45678.jpg → "HOUSE"
"""

from pathlib import Path
import fiftyone as fo

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "mjsynth"

# MJSynth extracts into a deep folder structure; locate the annotation files
POSSIBLE_ROOTS = [
    DATA_ROOT,
    DATA_ROOT / "mnt" / "ramdisk" / "max" / "90kDICT32px",
]


def _find_root():
    for r in POSSIBLE_ROOTS:
        if (r / "annotation_train.txt").exists():
            return r
    raise FileNotFoundError(
        "Cannot find annotation_train.txt. Check extraction path under data/raw/mjsynth/"
    )


def _word_from_filename(fname):
    """Extract the word from filename like '1_HOUSE_45678.jpg' → 'HOUSE'."""
    stem = Path(fname).stem
    parts = stem.split("_")
    if len(parts) >= 3:
        return "_".join(parts[1:-1])
    return stem


def load_split(root, split="train", max_samples=None):
    ann_file = root / f"annotation_{split}.txt"
    if not ann_file.exists():
        raise FileNotFoundError(f"{ann_file} not found.")

    samples = []
    with open(ann_file, "r") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            parts = line.strip().split()
            if not parts:
                continue
            rel_path = parts[0].lstrip("./")
            img_path = root / rel_path
            if not img_path.exists():
                continue
            word = _word_from_filename(rel_path)
            sample = fo.Sample(filepath=str(img_path))
            sample["word"] = word
            sample["ground_truth"] = fo.Classification(label=word)
            samples.append(sample)
    return samples


def create_dataset(name="mjsynth", max_per_split=None):
    """Load MJSynth. Pass max_per_split to limit samples (useful for quick tests)."""
    root = _find_root()
    dataset = fo.Dataset(name, overwrite=True)
    for split in ("train", "val", "test"):
        try:
            samples = load_split(root, split, max_samples=max_per_split)
            dataset.add_samples(samples, tags=[split])
            print(f"  Added {len(samples)} {split} samples")
        except FileNotFoundError as e:
            print(f"  Skipping {split}: {e}")
    dataset.persistent = True
    print(f"Dataset '{name}' created with {len(dataset)} samples.")
    return dataset


if __name__ == "__main__":
    # Load a small subset first to verify (remove max_per_split for full load)
    ds = create_dataset(max_per_split=5000)
    session = fo.launch_app(ds)
    session.wait()
