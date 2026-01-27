"""
MedGemma module for medical document processing.

This module provides:
- Model loading with optional quantization
- Text/image generation
- Configurable prompts by mode
- Image decoding utilities
"""

from .model import load_model, generate
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
