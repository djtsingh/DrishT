"""Extract remaining ind_kaggle zip files."""
import zipfile
import os
import sys

base = "data/raw/ind_kaggle"

extractions = [
    ("Autorickshaw Image Dataset.zip", "Autorickshaw_Image_Dataset"),
    ("ind_traffic_sign_img.zip", "ind_traffic_sign"),
    ("INDRA (INdian Dataset for RoAd crossing.zip", "INDRA"),
]

for zf, dest in extractions:
    src = os.path.join(base, zf)
    dst = os.path.join(base, dest)
    if not os.path.exists(src):
        print(f"SKIP (not found): {zf}")
        continue
    os.makedirs(dst, exist_ok=True)
    existing = sum(len(fs) for _, _, fs in os.walk(dst))
    if existing > 50:
        print(f"SKIP ({existing} files already): {dest}")
        continue
    print(f"Extracting {zf} -> {dest}...", flush=True)
    try:
        with zipfile.ZipFile(src, "r") as z:
            z.extractall(dst)
        count = sum(len(fs) for _, _, fs in os.walk(dst))
        print(f"  Done: {count} files", flush=True)
    except zipfile.BadZipFile:
        print(f"  ERROR: Bad/corrupt zip file: {zf}", flush=True)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)

print("All extractions complete.", flush=True)
