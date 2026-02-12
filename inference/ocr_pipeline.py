"""
DrishT: End-to-End OCR Pipeline for Indian Context
===================================================
Detection (SSDLite-MobileNetV3) → Crop → Recognition (CRNN-Light) → Text

Usage:
    from inference.ocr_pipeline import DrishTOCR
    
    ocr = DrishTOCR()
    results = ocr.process('image.jpg')
    
    for r in results:
        print(f"{r['text']} (conf: {r['confidence']:.2f}, class: {r['class']})")
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ============================================================================
# RECOGNITION MODEL ARCHITECTURE (must match training)
# ============================================================================

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False), nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        return x * self.excitation(y).view(b, c, 1, 1)


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=4, use_se=True):
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.use_residual = (stride == 1 and in_ch == out_ch)
        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_ch, mid_ch, 1, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU6(inplace=True)]
        layers += [nn.Conv2d(mid_ch, mid_ch, 3, stride, 1, groups=mid_ch, bias=False),
                   nn.BatchNorm2d(mid_ch), nn.ReLU6(inplace=True)]
        self.conv = nn.Sequential(*layers)
        self.se = SEBlock(mid_ch) if use_se else nn.Identity()
        self.project = nn.Sequential(nn.Conv2d(mid_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = self.project(self.se(self.conv(x)))
        return out + x if self.use_residual else out


class LightCNNEncoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU6(inplace=True),
            DepthwiseSeparableConv(32, 64), nn.MaxPool2d((2, 1)))
        self.stage2 = nn.Sequential(
            InvertedResidual(64, 128), InvertedResidual(128, 128), nn.MaxPool2d((2, 1)))
        self.stage3 = nn.Sequential(
            InvertedResidual(128, 256), InvertedResidual(256, 256), nn.MaxPool2d((2, 2)))
        self.stage4 = nn.Sequential(
            InvertedResidual(256, 384), InvertedResidual(384, 384), nn.MaxPool2d((2, 1)), nn.Dropout2d(0.1))
        self.stage5 = nn.Sequential(
            nn.Conv2d(384, 512, (2, 1), bias=False), nn.BatchNorm2d(512), nn.ReLU6(inplace=True),
            SEBlock(512), nn.Dropout2d(0.1))

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return x.squeeze(2).permute(0, 2, 1)


class BiLSTMSequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        return self.linear(self.lstm(x)[0])


class CRNN(nn.Module):
    def __init__(self, num_classes, img_h=32, num_channels=1, hidden_size=256, num_lstm_layers=2, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.img_h = img_h
        self.num_channels = num_channels
        self.cnn = LightCNNEncoder(in_channels=num_channels)
        self.rnn = BiLSTMSequenceModel(512, hidden_size, num_lstm_layers, dropout)
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.output(self.rnn(self.cnn(x)))


# ============================================================================
# CHARACTER CODEC
# ============================================================================

class CharCodec:
    """Encode/decode characters. Index 0 is CTC blank."""
    
    def __init__(self, charset_path: Union[str, Path]):
        with open(charset_path, 'r', encoding='utf-8') as f:
            chars = [line.rstrip('\n') for line in f]
        # Index 0 = CTC blank, chars start at index 1
        self.char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
        self.idx_to_char = {i + 1: c for i, c in enumerate(chars)}
        self.num_classes = len(chars) + 1  # +1 for blank

    def encode(self, text: str) -> List[int]:
        return [self.char_to_idx.get(c, 0) for c in text]

    def decode(self, indices: List[int]) -> str:
        return ''.join(self.idx_to_char.get(i, '') for i in indices if i > 0)


# ============================================================================
# OCR RESULT
# ============================================================================

@dataclass
class OCRResult:
    """Single OCR detection result."""
    text: str
    confidence: float
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_name: str
    class_id: int


# ============================================================================
# MAIN OCR PIPELINE
# ============================================================================

class DrishTOCR:
    """End-to-end OCR pipeline: Detection → Crop → Recognition."""
    
    DETECTION_CLASSES = {
        0: 'background', 1: 'text', 2: 'license_plate', 3: 'traffic_sign',
        4: 'autorickshaw', 5: 'tempo', 6: 'truck', 7: 'bus'
    }
    
    # Classes that contain readable text
    TEXT_CLASSES = {1, 2, 3}  # text, license_plate, traffic_sign
    
    def __init__(
        self,
        detection_model_path: Optional[str] = None,
        recognition_model_path: Optional[str] = None,
        charset_path: Optional[str] = None,
        device: str = 'auto',
        detection_threshold: float = 0.5,
        recognition_use_onnx: bool = True,
    ):
        """
        Initialize the OCR pipeline.
        
        Args:
            detection_model_path: Path to SSDLite detection model (.pt or .pth)
            recognition_model_path: Path to CRNN recognition model (.onnx or .pth)
            charset_path: Path to charset.txt
            device: 'cuda', 'cpu', or 'auto'
            detection_threshold: Minimum confidence for detection
            recognition_use_onnx: Whether to use ONNX runtime for recognition
        """
        # Resolve default paths
        base = Path(__file__).parent.parent
        if detection_model_path is None:
            detection_model_path = base / 'models' / 'detection_output' / 'ssdlite_detection.pt'
        if recognition_model_path is None:
            if recognition_use_onnx:
                recognition_model_path = base / 'models' / 'recognition_output' / 'crnn_recognition.onnx'
            else:
                recognition_model_path = base / 'models' / 'recognition_output' / 'best_acc.pth'
        if charset_path is None:
            charset_path = base / 'models' / 'recognition_output' / 'charset.txt'
        
        # Device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.detection_threshold = detection_threshold
        self.recognition_use_onnx = recognition_use_onnx
        
        # Load charset
        self.codec = CharCodec(charset_path)
        
        # Load detection model (TorchScript)
        self._load_detection_model(Path(detection_model_path))
        
        # Load recognition model
        self._load_recognition_model(Path(recognition_model_path))
        
        print(f'DrishTOCR initialized on {self.device}')
        print(f'  Detection: {detection_model_path}')
        print(f'  Recognition: {recognition_model_path} (ONNX={recognition_use_onnx})')

    def _load_detection_model(self, path: Path):
        """Load SSDLite detection model."""
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large
        
        # Always use checkpoint for cross-platform compatibility
        # TorchScript has issues with torchvision ops across environments
        ckpt_path = path.parent / 'best_map.pth' if path.suffix != '.pth' else path
        
        if not ckpt_path.exists():
            raise FileNotFoundError(f'Detection checkpoint not found: {ckpt_path}')
        
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        self.detector = ssdlite320_mobilenet_v3_large(num_classes=8)
        state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))
        self.detector.load_state_dict(state_dict)
        self.detector = self.detector.to(self.device)
        self.detector.eval()
        self._detector_is_torchscript = False

    def _load_recognition_model(self, path: Path):
        """Load CRNN recognition model."""
        if self.recognition_use_onnx and path.suffix == '.onnx':
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device.type == 'cuda' else ['CPUExecutionProvider']
            self.recognizer = ort.InferenceSession(str(path), providers=providers)
            self.recognizer_type = 'onnx'
        else:
            # PyTorch checkpoint
            ckpt = torch.load(path if path.suffix == '.pth' else path.with_suffix('.pth'), 
                             map_location='cpu', weights_only=True)
            self.recognizer = CRNN(
                num_classes=ckpt.get('num_classes', self.codec.num_classes),
                img_h=ckpt.get('img_h', 32),
                num_channels=ckpt.get('num_channels', 1),
            )
            self.recognizer.load_state_dict(ckpt['state_dict'])
            self.recognizer = self.recognizer.to(self.device)
            self.recognizer.eval()
            self.recognizer_type = 'pytorch'

    @torch.no_grad()
    def detect(self, image: Image.Image) -> List[Dict]:
        """
        Run detection on an image.
        
        Returns:
            List of dicts with keys: box, class_id, class_name, score
        """
        # Prepare image - returns (3, 320, 320) float32 tensor
        img_tensor = self._preprocess_detection(image)
        
        # Run detection - SSD expects list of tensors
        outputs = self.detector([img_tensor])[0]
        boxes = outputs['boxes']
        labels = outputs['labels']
        scores = outputs['scores']
        
        # Scale boxes back to original image size
        orig_w, orig_h = image.size
        scale_x, scale_y = orig_w / 320, orig_h / 320
        
        results = []
        for i in range(len(boxes)):
            score = scores[i].item()
            if score < self.detection_threshold:
                continue
            
            box = boxes[i].cpu().numpy()
            x1 = int(box[0] * scale_x)
            y1 = int(box[1] * scale_y)
            x2 = int(box[2] * scale_x)
            y2 = int(box[3] * scale_y)
            
            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)
            
            class_id = labels[i].item()
            results.append({
                'box': (x1, y1, x2, y2),
                'class_id': class_id,
                'class_name': self.DETECTION_CLASSES.get(class_id, 'unknown'),
                'score': score,
            })
        
        return results

    def _preprocess_detection(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for detection (resize to 320x320, normalize to [0,1])."""
        img = image.convert('RGB').resize((320, 320), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))  # (3, 320, 320)
        return tensor.to(self.device)

    @torch.no_grad()
    def recognize(self, crop: Image.Image) -> Tuple[str, float]:
        """
        Run recognition on a cropped text region.
        
        Returns:
            (text, confidence)
        """
        # Preprocess
        tensor = self._preprocess_recognition(crop)
        
        # Run inference
        if self.recognizer_type == 'onnx':
            log_probs = self.recognizer.run(None, {'image': tensor.cpu().numpy()})[0]
            log_probs = torch.from_numpy(log_probs)
        else:
            logits = self.recognizer(tensor.to(self.device))
            log_probs = F.log_softmax(logits, dim=2).cpu()
        
        # Decode
        text, confidence = self._ctc_decode(log_probs[0])
        return text, confidence

    def _preprocess_recognition(self, image: Image.Image) -> torch.Tensor:
        """Preprocess crop for recognition (grayscale, resize, normalize)."""
        img = image.convert('L')  # Grayscale
        w, h = img.size
        
        # Resize to height 32, maintaining aspect ratio
        new_h = 32
        new_w = min(int(w * new_h / h), 128)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        
        # Convert to tensor, pad to width 128
        arr = np.array(img, dtype=np.float32)
        tensor = torch.zeros(1, 1, 32, 128)
        tensor[0, 0, :, :new_w] = torch.from_numpy(arr[:, :new_w])
        
        # Normalize to [-1, 1]
        tensor = (tensor / 255.0 - 0.5) / 0.5
        return tensor

    def _ctc_decode(self, log_probs: torch.Tensor) -> Tuple[str, float]:
        """Greedy CTC decoding with confidence estimation."""
        # log_probs: (T, num_classes)
        probs = log_probs.exp()
        max_probs, indices = probs.max(dim=1)
        
        # Greedy decode (collapse repeats, remove blanks)
        decoded = []
        confidences = []
        prev = -1
        for t, idx in enumerate(indices.tolist()):
            if idx != prev and idx != 0:  # not repeat, not blank
                decoded.append(idx)
                confidences.append(max_probs[t].item())
            prev = idx
        
        text = self.codec.decode(decoded)
        confidence = np.mean(confidences) if confidences else 0.0
        return text, confidence

    def process(
        self, 
        image: Union[str, Path, Image.Image],
        recognize_all_classes: bool = False,
    ) -> List[OCRResult]:
        """
        Full OCR pipeline: detect text regions and recognize text.
        
        Args:
            image: Image path or PIL Image
            recognize_all_classes: If True, run recognition on all detected objects (not just text classes)
            
        Returns:
            List of OCRResult objects
        """
        # Load image
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        # Detect
        detections = self.detect(image)
        
        # Recognize each text region
        results = []
        for det in detections:
            # Skip non-text classes unless requested
            if not recognize_all_classes and det['class_id'] not in self.TEXT_CLASSES:
                continue
            
            # Crop region
            x1, y1, x2, y2 = det['box']
            if x2 - x1 < 5 or y2 - y1 < 5:  # Skip too small
                continue
            
            crop = image.crop((x1, y1, x2, y2))
            
            # Recognize
            text, conf = self.recognize(crop)
            
            results.append(OCRResult(
                text=text,
                confidence=conf * det['score'],  # Combined confidence
                box=det['box'],
                class_name=det['class_name'],
                class_id=det['class_id'],
            ))
        
        return results

    def visualize(
        self, 
        image: Union[str, Path, Image.Image],
        results: Optional[List[OCRResult]] = None,
        output_path: Optional[str] = None,
    ) -> Image.Image:
        """
        Visualize OCR results on image.
        
        Args:
            image: Input image
            results: OCR results (if None, will run process())
            output_path: Optional path to save visualization
            
        Returns:
            PIL Image with annotations
        """
        from PIL import ImageDraw, ImageFont
        
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        image = image.copy().convert('RGB')
        
        if results is None:
            results = self.process(image)
        
        draw = ImageDraw.Draw(image)
        
        # Colors for different classes
        colors = {
            'text': 'green',
            'license_plate': 'blue',
            'traffic_sign': 'red',
            'autorickshaw': 'orange',
            'tempo': 'purple',
            'truck': 'brown',
            'bus': 'pink',
        }
        
        for r in results:
            color = colors.get(r.class_name, 'yellow')
            x1, y1, x2, y2 = r.box
            
            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Draw label
            label = f'{r.text} ({r.confidence:.2f})'
            draw.text((x1, y1 - 15), label, fill=color)
        
        if output_path:
            image.save(output_path)
        
        return image


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='DrishT OCR Pipeline')
    parser.add_argument('image', help='Input image path')
    parser.add_argument('--output', '-o', help='Output visualization path')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help='Detection threshold')
    parser.add_argument('--device', '-d', default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--all-classes', action='store_true', help='Recognize all detected classes')
    args = parser.parse_args()
    
    ocr = DrishTOCR(device=args.device, detection_threshold=args.threshold)
    results = ocr.process(args.image, recognize_all_classes=args.all_classes)
    
    print(f'\nFound {len(results)} text regions:\n')
    for i, r in enumerate(results, 1):
        print(f'{i}. "{r.text}"')
        print(f'   Class: {r.class_name}, Confidence: {r.confidence:.3f}')
        print(f'   Box: {r.box}')
        print()
    
    if args.output:
        ocr.visualize(args.image, results, args.output)
        print(f'Visualization saved to: {args.output}')


if __name__ == '__main__':
    main()
