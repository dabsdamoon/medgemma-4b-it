"""OCR modules for text extraction."""

from .base import OCRBase, OCRResult
from .local_ocr import TesseractOCR, EasyOCR
from .google_vision import GoogleVisionOCR

__all__ = [
    "OCRBase",
    "OCRResult",
    "TesseractOCR",
    "EasyOCR",
    "GoogleVisionOCR",
]
