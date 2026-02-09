"""
Package DrishT datasets for Kaggle upload.
Creates two zip files:
  - drisht-detection.zip  (detection train/val/test + annotations)
  - drisht-recognition.zip (recognition train/val/test + labels + charset)
"""
import zipfile, os, sys
from pathlib import Path

def zip_dir(base_dir, zip_path, prefix=""):
    """Zip a directory recursively."""
    base = Path(base_dir)
    count = 0
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(base.rglob('*')):
            if f.is_file():
                arcname = prefix + str(f.relative_to(base))
                zf.write(f, arcname)
                count += 1
                if count % 5000 == 0:
                    print(f"  {count} files...")
    return count

out = Path("data/kaggle_upload")
out.mkdir(parents=True, exist_ok=True)

# Detection
print("=== Packaging Detection Dataset ===")
n = zip_dir("data/final/detection", out / "drisht-detection.zip")
sz = (out / "drisht-detection.zip").stat().st_size / (1024*1024)
print(f"  {n} files -> {sz:.1f} MB")

# Recognition
print("\n=== Packaging Recognition Dataset ===")
n = zip_dir("data/final/recognition", out / "drisht-recognition.zip")
sz = (out / "drisht-recognition.zip").stat().st_size / (1024*1024)
print(f"  {n} files -> {sz:.1f} MB")

# Also zip the charset separately (small, useful)
import shutil
shutil.copy2("data/final/recognition/charset.txt", out / "charset.txt")

print(f"\nDone! Files in: {out}")
print("Upload these as Kaggle Datasets:")
print("  1. drisht-detection.zip  -> kaggle datasets create")
print("  2. drisht-recognition.zip -> kaggle datasets create")
