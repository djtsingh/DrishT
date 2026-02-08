"""
Load SCUT-CTW1500 into FiftyOne.

Annotations-v2 (XML format) are in:
  data/raw/ctw1500/annotations_v2/   (0001.xml ... 1000.xml  for train)

Images need manual download (Box.com links require browser):
  Open: https://github.com/Yuliang-Liu/Curve-Text-Detector README
  Download train_images.zip and test_images.zip from Box.com
  Extract into:
    data/raw/ctw1500/train/text_image/
    data/raw/ctw1500/test/text_image/

Alternative: download from Baidu Pan
  https://pan.baidu.com/s/1eSvpq7o  PASSWORD: fatf
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import fiftyone as fo

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "ctw1500"
ANNO_DIR = DATA_ROOT / "annotations_v2"

IMG_DIRS = [
    DATA_ROOT / "train" / "text_image",
    DATA_ROOT / "ctw1500_train",
    DATA_ROOT / "images",
]


def _find_image(filename):
    for d in IMG_DIRS:
        p = d / filename
        if p.exists():
            return p
    for p in DATA_ROOT.rglob(filename):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            return p
    return None


def parse_xml_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    detections = []
    image_file = None
    for img_elem in root.iter("image"):
        image_file = img_elem.get("file", "")
        for box_elem in img_elem.iter("box"):
            label = ""
            coords = []
            label_elem = box_elem.find("label")
            if label_elem is not None and label_elem.text:
                label = label_elem.text.strip()
            segs_elem = box_elem.find("segs")
            if segs_elem is not None and segs_elem.text:
                nums = list(map(float, segs_elem.text.strip().split(",")))
                coords = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
            if coords:
                detections.append((coords, label))
    return image_file, detections


def _poly_to_rel(coords, w, h):
    pts = [[x / w, y / h] for x, y in coords]
    pts.append(pts[0])
    return pts


def create_dataset(name="ctw1500"):
    if not ANNO_DIR.exists():
        print(f"Annotation dir not found: {ANNO_DIR}")
        return None

    dataset = fo.Dataset(name, overwrite=True)
    xml_files = sorted(ANNO_DIR.glob("*.xml"))
    print(f"  Found {len(xml_files)} XML annotation files.")

    loaded, skipped = 0, 0
    for xml_path in xml_files:
        image_file, detections = parse_xml_annotation(xml_path)
        if not image_file:
            skipped += 1
            continue
        img_path = _find_image(image_file)
        if img_path is None:
            skipped += 1
            continue
        from PIL import Image
        im = Image.open(img_path)
        w, h = im.size
        sample = fo.Sample(filepath=str(img_path))
        polylines = []
        for coords, text in detections:
            pts = _poly_to_rel(coords, w, h)
            polylines.append(
                fo.Polyline(points=[pts], closed=True, filled=True, label="text")
            )
        sample["gt_polylines"] = fo.Polylines(polylines=polylines)
        sample["num_text_instances"] = len(detections)
        dataset.add_sample(sample)
        loaded += 1

    dataset.persistent = True
    print(f"Dataset '{name}': {loaded} loaded, {skipped} skipped (images not found).")
    return dataset


if __name__ == "__main__":
    ds = create_dataset()
    if ds and len(ds) > 0:
        session = fo.launch_app(ds)
        session.wait()
