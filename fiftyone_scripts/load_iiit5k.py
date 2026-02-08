"""
Load IIIT 5K-Word into FiftyOne.
Expects:
  data/raw/iiit5k/IIIT5K/   (after extraction)
    train/
    test/
    trainCharBound.mat   or  traindata.mat
    testCharBound.mat    or  testdata.mat

The .mat files contain word labels and bounding info.
Fallback: extract word from folder/filename structure.
"""

from pathlib import Path
import fiftyone as fo

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "iiit5k"

POSSIBLE_ROOTS = [
    DATA_ROOT,
    DATA_ROOT / "IIIT5K",
    DATA_ROOT / "IIIT5K-Word_V3.0",
]


def _find_root():
    for r in POSSIBLE_ROOTS:
        # Look for mat files or train/ folder
        if (r / "trainCharBound.mat").exists() or (r / "traindata.mat").exists():
            return r
        if (r / "train").exists():
            return r
    # Fallback: just use DATA_ROOT if images exist somewhere
    return DATA_ROOT


def load_via_mat(root, split="train"):
    """Load using scipy to read .mat label files."""
    import scipy.io as sio

    mat_candidates = [
        root / f"{split}CharBound.mat",
        root / f"{split}data.mat",
    ]
    mat_path = None
    for c in mat_candidates:
        if c.exists():
            mat_path = c
            break

    if mat_path is None:
        return load_via_glob(root, split)

    mat = sio.loadmat(str(mat_path))
    # The key varies: 'trainCharBound' or 'testCharBound' or 'traindata'
    key = None
    for k in mat:
        if split in k.lower() and not k.startswith("__"):
            key = k
            break
    if key is None:
        return load_via_glob(root, split)

    data = mat[key][0]
    samples = []
    for entry in data:
        img_rel = str(entry[0][0])  # relative image path
        word = str(entry[1][0])  # word label
        img_path = root / img_rel
        if not img_path.exists():
            continue
        sample = fo.Sample(filepath=str(img_path))
        sample["word"] = word
        sample["ground_truth"] = fo.Classification(label=word)
        samples.append(sample)
    return samples


def load_via_glob(root, split="train"):
    """Fallback: glob images and try to extract word from path."""
    img_dir = root / split
    if not img_dir.exists():
        img_dir = root  # images might be in root
    samples = []
    for img_path in sorted(img_dir.rglob("*.*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        sample = fo.Sample(filepath=str(img_path))
        # Try to extract word from filename
        word = img_path.stem.split("_")[0] if "_" in img_path.stem else img_path.stem
        sample["word"] = word
        sample["ground_truth"] = fo.Classification(label=word)
        samples.append(sample)
    return samples


def create_dataset(name="iiit5k"):
    root = _find_root()
    dataset = fo.Dataset(name, overwrite=True)
    for split in ("train", "test"):
        try:
            samples = load_via_mat(root, split)
            if samples:
                dataset.add_samples(samples, tags=[split])
                print(f"  Added {len(samples)} {split} samples")
            else:
                print(f"  No samples found for {split}")
        except Exception as e:
            print(f"  Error loading {split}: {e}")
    dataset.persistent = True
    print(f"Dataset '{name}' created with {len(dataset)} samples.")
    return dataset


if __name__ == "__main__":
    ds = create_dataset()
    session = fo.launch_app(ds)
    session.wait()
