"""
Load IIIT-H Indic Scene Text into FiftyOne.
Expects:
  data/raw/indic_scene/  (user places downloaded files here after requesting from CVIT)

This is a placeholder — actual parsing depends on the format delivered by CVIT lab.
Typical format: images + XML or JSON annotations with script labels.
"""

from pathlib import Path
import fiftyone as fo

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "indic_scene"


def create_dataset(name="indic_scene"):
    if not DATA_ROOT.exists() or not any(DATA_ROOT.iterdir()):
        print(
            "No data found in data/raw/indic_scene/.\n"
            "Request the dataset from CVIT IIIT-H: https://cvit.iiit.ac.in/\n"
            "Place the downloaded files in data/raw/indic_scene/ and re-run."
        )
        return None

    dataset = fo.Dataset(name, overwrite=True)
    # Glob all images
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    samples = []
    for img in sorted(DATA_ROOT.rglob("*")):
        if img.suffix.lower() in exts:
            sample = fo.Sample(filepath=str(img))
            samples.append(sample)

    if samples:
        dataset.add_samples(samples)
        print(f"  Added {len(samples)} images (annotations TBD based on format).")
    dataset.persistent = True
    print(f"Dataset '{name}' created with {len(dataset)} samples.")
    return dataset


if __name__ == "__main__":
    ds = create_dataset()
    if ds:
        session = fo.launch_app(ds)
        session.wait()
