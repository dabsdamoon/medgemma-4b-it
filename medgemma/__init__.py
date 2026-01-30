"""
MedGemma module for medical document processing.

This module provides:
- Model loading with optional quantization
- Text/image generation
- Configurable prompts by mode
- Image decoding utilities
"""

# Suppress NumPy/Torch compatibility warnings during import
import sys
import os
import warnings

os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")

# Temporarily redirect stderr to suppress NumPy C-level warnings
_stderr = sys.stderr
try:
    sys.stderr = open(os.devnull, "w")
    from .model import load_model, generate
finally:
    sys.stderr.close()
    sys.stderr = _stderr

from .prompts import get_prompt, list_modes, DEFAULT_PROMPTS
from .utils import decode_image, create_dummy_image, encode_image, hash_image

__all__ = [
    "load_model",
    "generate",
    "get_prompt",
    "list_modes",
    "DEFAULT_PROMPTS",
    "decode_image",
    "create_dummy_image",
    "encode_image",
    "hash_image",
]
