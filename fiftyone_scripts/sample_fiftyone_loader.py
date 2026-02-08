import fiftyone as fo
from pathlib import Path

"""
Sample template to create a FiftyOne dataset from a folder of images and optional detection labels.
Fill in dataset paths after you download datasets.
"""

def create_image_dataset(name: str, images_dir: str):
    images_dir = Path(images_dir)
    dataset = fo.Dataset(name)

    # Add images from directory (simple example)
    samples = [fo.Sample(filepath=str(p)) for p in images_dir.glob("**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    dataset.add_samples(samples)
    print(f"Created dataset '{name}' with {len(samples)} samples")
    return dataset


if __name__ == "__main__":
    print("This is a template. Replace 'path/to/images' with your dataset path and run to load into FiftyOne.")
    # ds = create_image_dataset("ocr_demo", "data/raw/path/to/images")
    # fo.launch_app(ds)
