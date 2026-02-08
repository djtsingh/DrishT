"""
Master loader — run all dataset loaders and print a summary.
Usage:  python fiftyone_scripts/load_all.py
"""

import importlib, sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

LOADERS = [
    ("load_icdar2015", "icdar2015"),
    ("load_totaltext", "totaltext"),
    ("load_ctw1500", "ctw1500"),
    ("load_mjsynth", "mjsynth"),
    ("load_iiit5k", "iiit5k"),
    ("load_indic_scene", "indic_scene"),
]


def main():
    results = {}
    for module_name, dataset_name in LOADERS:
        print(f"\n{'='*60}")
        print(f"  Loading: {dataset_name}")
        print(f"{'='*60}")
        try:
            mod = importlib.import_module(module_name)
            ds = mod.create_dataset(name=dataset_name)
            results[dataset_name] = len(ds) if ds else 0
        except Exception as e:
            print(f"  FAILED: {e}")
            results[dataset_name] = "ERROR"

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name, count in results.items():
        status = f"{count} samples" if isinstance(count, int) and count > 0 else "NOT LOADED"
        print(f"  {name:20s} → {status}")
    print()


if __name__ == "__main__":
    main()
