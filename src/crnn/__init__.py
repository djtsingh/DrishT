"""Recognition module — CRNN-Light with efficient CNN backbone.

Upgraded from basic 7-layer CNN to lightweight encoder with:
  - Depthwise separable convolutions (MobileNet-style)
  - Squeeze-and-Excitation attention blocks
  - ~1.5M CNN params vs ~3M (2× smaller, faster inference)
  - BiLSTM + CTC decoder (proven architecture for multi-script text)
"""

__all__ = ["model", "config", "train"]
