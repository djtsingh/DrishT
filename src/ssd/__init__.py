"""Detection module — SSDLite320 with MobileNetV3-Large backbone.

Upgraded from SSD-VGG16 (2016) to SSDLite-MobileNetV3 (2020+):
  - 3.4M params vs 26M params (7.6× smaller)
  - Depthwise separable convolutions for mobile efficiency
  - Native torchvision API with built-in loss and NMS
"""

__all__ = ["model", "config", "train"]
