"""
SSDLite-MobileNetV3 Text Detector — Model Definition
=======================================================
Lightweight SSD with MobileNetV3-Large backbone for mobile-optimized
Indian scene text and object detection.

Upgrade from SSD-VGG16 (2016) to SSDLite-MobileNetV3 (2020+):
  - 3.4M params vs 26M params (7.6× smaller)
  - Depthwise separable convolutions for mobile efficiency
  - 320×320 input (vs 300×300)
  - Pretrained MobileNetV3 backbone (ImageNet)
  - Native torchvision API with built-in NMS and loss computation

Categories: background(0), text(1), license_plate(2), traffic_sign(3),
            autorickshaw(4), tempo(5), truck(6), bus(7)

Usage:
    from src.ssd.model import create_model, save_model, load_model

    model = create_model(num_classes=8, pretrained_backbone=True)
    # Training: loss_dict = model(images, targets)
    # Inference: detections = model(images)
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights
from pathlib import Path

from .config import SSDConfig as cfg


def create_model(num_classes=None, pretrained_backbone=True):
    """Create SSDLite320 with MobileNetV3-Large backbone.

    Args:
        num_classes: Number of detection classes (including background).
                     Defaults to SSDConfig.NUM_CLASSES.
        pretrained_backbone: Use ImageNet-pretrained MobileNetV3-Large backbone.

    Returns:
        torchvision SSD model.

    Note:
        - Training mode:  loss_dict = model(images, targets)
          Returns dict: {'classification': Tensor, 'bbox_regression': Tensor}
        - Eval mode:      detections = model(images)
          Returns list of dicts: [{'boxes':, 'labels':, 'scores':}, ...]
        - Images: list of tensors in [0, 1] range (model normalizes internally)
        - Targets: list of dicts with 'boxes' (x1,y1,x2,y2 absolute) and 'labels'
    """
    nc = num_classes or cfg.NUM_CLASSES
    backbone_weights = (
        MobileNet_V3_Large_Weights.IMAGENET1K_V1
        if pretrained_backbone else None
    )

    model = ssdlite320_mobilenet_v3_large(
        num_classes=nc,
        weights_backbone=backbone_weights,
    )
    return model


def save_model(model, path, epoch=None, optimizer=None, scheduler=None,
               best_metric=None, extra=None):
    """Save model checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"model": model.state_dict()}
    if epoch is not None:
        ckpt["epoch"] = epoch
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    if best_metric is not None:
        ckpt["best_metric"] = best_metric
    if extra is not None:
        ckpt["extra"] = extra
    torch.save(ckpt, path)


def load_model(path, num_classes=None, device="cpu"):
    """Load model from checkpoint."""
    model = create_model(num_classes=num_classes, pretrained_backbone=False)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def model_info(model):
    """Print and return model parameter summary."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

    info = {
        "total_params": total,
        "trainable_params": trainable,
        "model_size_mb": size_mb,
    }
    print(f"  Total parameters:     {total:>12,}")
    print(f"  Trainable:            {trainable:>12,}")
    print(f"  Model size:           {size_mb:>10.1f} MB")
    return info


def freeze_backbone(model, freeze=True):
    """Freeze/unfreeze the MobileNetV3 backbone.

    Strategy: freeze backbone for first few epochs (train head only),
    then unfreeze for full fine-tuning. Saves memory and improves convergence.
    """
    for param in model.backbone.parameters():
        param.requires_grad = not freeze


def export_onnx(model, path, input_size=320, opset=17):
    """Export to ONNX for mobile deployment (TFLite, CoreML, etc.)."""
    model.eval()
    dummy = [torch.randn(3, input_size, input_size)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(path),
        opset_version=opset,
        input_names=["image"],
        output_names=["boxes", "labels", "scores"],
    )
    print(f"  Exported ONNX: {path}")
