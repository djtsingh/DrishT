"""
DrishT Master Pipeline Runner
================================
Runs the complete data processing pipeline in order:
  1. Build unified detection dataset (COCO JSON)
  2. Build unified recognition dataset (word crops + labels)
  3. Generate synthetic Indian scene-text data
  4. Quality check + final train/val/test export

Usage:
  cd G:\2025\DrishT
  .\.venv\Scripts\Activate.ps1
  python scripts/run_pipeline.py          # run all steps
  python scripts/run_pipeline.py --step 1 # run only step 1
  python scripts/run_pipeline.py --step 3 # run only synthetic gen
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path


SCRIPTS = [
    ("Step 1: Detection Dataset",   "scripts/build_detection_dataset.py"),
    ("Step 2: Recognition Dataset",  "scripts/build_recognition_dataset.py"),
    ("Step 3: Synthetic Data",       "scripts/generate_synthetic.py"),
    ("Step 4: Quality & Export",     "scripts/quality_and_export.py"),
]


def run_step(name, script_path):
    print(f"\n{'#' * 60}")
    print(f"  {name}")
    print(f"  Script: {script_path}")
    print(f"{'#' * 60}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=str(Path(__file__).parent.parent),
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  ✗ {name} FAILED (exit code {result.returncode})")
        print(f"    Time: {elapsed:.1f}s")
        return False
    else:
        print(f"\n  ✓ {name} completed in {elapsed:.1f}s")
        return True


def main():
    parser = argparse.ArgumentParser(description="DrishT Master Pipeline")
    parser.add_argument("--step", type=int, default=0,
                        help="Run only this step (1-4). 0 = all steps.")
    args = parser.parse_args()

    print("=" * 60)
    print("  DrishT Master Data Pipeline")
    print("=" * 60)

    if args.step > 0:
        if args.step > len(SCRIPTS):
            print(f"Invalid step {args.step}. Choose 1-{len(SCRIPTS)}.")
            sys.exit(1)
        name, path = SCRIPTS[args.step - 1]
        success = run_step(name, path)
        sys.exit(0 if success else 1)

    # Run all steps
    results = []
    for name, path in SCRIPTS:
        success = run_step(name, path)
        results.append((name, success))
        if not success:
            print(f"\n  Pipeline stopped at: {name}")
            break

    print(f"\n{'='*60}")
    print("  Pipeline Summary")
    print(f"{'='*60}")
    for name, ok in results:
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")
    print()

    if all(ok for _, ok in results):
        print("  All steps completed successfully!")
        print("  Output: data/final/")
    else:
        print("  Some steps failed. Check output above for errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
