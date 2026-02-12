"""Quick test of DrishT OCR pipeline."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.ocr_pipeline import DrishTOCR

# Find test images with text (TotalText scene images)
test_images_dir = Path(__file__).parent.parent / 'data' / 'final' / 'detection' / 'test' / 'images'
images = list(test_images_dir.glob('totaltext*.jpg'))[:3]
if not images:
    images = list(test_images_dir.glob('*.jpg'))[:3]

if not images:
    print('No test images found!')
    exit(1)

print(f'Testing on {len(images)} images...\n')

# Initialize OCR (this also loads models)
try:
    ocr = DrishTOCR(recognition_use_onnx=True)
except Exception as e:
    print(f'Failed to initialize OCR: {e}')
    # Try without ONNX
    print('Trying with PyTorch instead of ONNX...')
    ocr = DrishTOCR(recognition_use_onnx=False)

# Test each image
for img_path in images:
    print(f'\n{"="*60}')
    print(f'Image: {img_path.name}')
    print('='*60)
    
    results = ocr.process(str(img_path), recognize_all_classes=False)
    
    if not results:
        print('  No detections.')
        continue
    
    for r in results:
        text_preview = r.text[:50] + '...' if len(r.text) > 50 else r.text
        print(f'  [{r.class_name}] "{text_preview}" (conf: {r.confidence:.3f})')

print('\n\nDone!')
