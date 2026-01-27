"""
Base OCR interface.

Defines the abstract base class for OCR implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image


@dataclass
class TextBlock:
    """A block of detected text."""

    text: str
    confidence: float
    bbox: Optional[tuple[int, int, int, int]] = None  # x, y, width, height


@dataclass
class OCRResult:
    """Result from OCR processing."""

    full_text: str
    blocks: list[TextBlock] = field(default_factory=list)
    language: str = "eng"
    confidence: float = 0.0


class OCRBase(ABC):
    """Abstract base class for OCR implementations."""

    def __init__(self, language: str = "eng"):
        """
        Initialize the OCR engine.

        Args:
            language: Language code for OCR
        """
        self.language = language

    @abstractmethod
    def extract_text(self, image: Image.Image) -> OCRResult:
        """
        Extract text from an image.

        Args:
            image: PIL Image to process

        Returns:
            OCRResult with extracted text and metadata
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this OCR backend is available."""
        pass

    def extract_text_simple(self, image: Image.Image) -> str:
        """
        Simple text extraction returning only the text string.

        Args:
            image: PIL Image to process

        Returns:
            Extracted text as a string
        """
        result = self.extract_text(image)
        return result.full_text
