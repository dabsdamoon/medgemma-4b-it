"""
Google Cloud Vision OCR implementation.

Provides high-quality OCR using Google Cloud Vision API.
"""

import io
from typing import Optional

from PIL import Image

from .base import OCRBase, OCRResult, TextBlock


class GoogleVisionOCR(OCRBase):
    """OCR using Google Cloud Vision API."""

    def __init__(self, language: str = "eng", credentials_path: Optional[str] = None):
        """
        Initialize Google Vision OCR.

        Args:
            language: Language hint for OCR
            credentials_path: Path to service account credentials JSON
        """
        super().__init__(language)
        self.credentials_path = credentials_path
        self._client: Optional[object] = None

    def _get_client(self):
        """Lazy load Google Cloud Vision client."""
        if self._client is None:
            try:
                from google.cloud import vision

                if self.credentials_path:
                    self._client = vision.ImageAnnotatorClient.from_service_account_json(
                        self.credentials_path
                    )
                else:
                    # Use default credentials (ADC)
                    self._client = vision.ImageAnnotatorClient()
            except ImportError:
                raise ImportError(
                    "google-cloud-vision is required for Google Vision OCR. "
                    "Install with: pip install google-cloud-vision"
                )
        return self._client

    def is_available(self) -> bool:
        """Check if Google Vision is available."""
        try:
            self._get_client()
            return True
        except Exception:
            return False

    def extract_text(self, image: Image.Image) -> OCRResult:
        """Extract text using Google Cloud Vision."""
        from google.cloud import vision

        client = self._get_client()

        # Convert PIL Image to bytes
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        content = buffer.getvalue()

        # Create Vision API image
        vision_image = vision.Image(content=content)

        # Set language hints
        image_context = vision.ImageContext(
            language_hints=[self._map_language_code(self.language)]
        )

        # Perform text detection
        response = client.text_detection(image=vision_image, image_context=image_context)

        if response.error.message:
            raise Exception(f"Google Vision API error: {response.error.message}")

        # Process results
        blocks = []
        full_text = ""

        if response.text_annotations:
            # First annotation contains the full text
            full_text = response.text_annotations[0].description

            # Remaining annotations are individual words/blocks
            for annotation in response.text_annotations[1:]:
                vertices = annotation.bounding_poly.vertices
                if vertices:
                    x = min(v.x for v in vertices)
                    y = min(v.y for v in vertices)
                    w = max(v.x for v in vertices) - x
                    h = max(v.y for v in vertices) - y

                    blocks.append(
                        TextBlock(
                            text=annotation.description,
                            confidence=1.0,  # Google Vision doesn't provide per-word confidence
                            bbox=(x, y, w, h),
                        )
                    )

        return OCRResult(
            full_text=full_text.strip(),
            blocks=blocks,
            language=self.language,
            confidence=1.0 if full_text else 0.0,
        )

    def _map_language_code(self, lang: str) -> str:
        """Map Tesseract-style language codes to Google Vision codes."""
        lang_map = {
            "eng": "en",
            "fra": "fr",
            "deu": "de",
            "spa": "es",
            "ita": "it",
            "por": "pt",
            "chi_sim": "zh-CN",
            "chi_tra": "zh-TW",
            "jpn": "ja",
            "kor": "ko",
        }
        return lang_map.get(lang, lang)
