"""
DrishT: Dual-System OCR for Indian Context
===========================================

A complete OCR system optimized for Indian scene text,
supporting 10 major Indian scripts plus Latin.

Components:
- Detection: DBNet++ for scene text localization
- Recognition: Vision Transformer for multi-script text

Usage:
    from drisht.models import build_dbnet_plusplus, build_recognizer
    from drisht.data import IndianTextSynthesizer
"""

__version__ = '2.0.0'
__author__ = 'DrishT Team'

from . import models
from . import data

__all__ = ['models', 'data']
