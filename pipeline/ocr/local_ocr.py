"""
Local OCR implementations using Tesseract and EasyOCR.

These provide reproducible OCR without external API dependencies.
"""

from typing import Optional

from PIL import Image

from .base import OCRBase, OCRResult, TextBlock


class TesseractOCR(OCRBase):
    """OCR using Tesseract (pytesseract)."""

    def __init__(self, language: str = "eng", config: str = ""):
        """
        Initialize Tesseract OCR.

        Args:
            language: Tesseract language code
            config: Additional Tesseract configuration
        """
        super().__init__(language)
        self.config = config
        self._pytesseract: Optional[object] = None

    def _get_tesseract(self):
        """Lazy load pytesseract."""
        if self._pytesseract is None:
            try:
                import pytesseract

                self._pytesseract = pytesseract
            except ImportError:
                raise ImportError(
                    "pytesseract is required for Tesseract OCR. "
                    "Install with: pip install pytesseract"
                )
        return self._pytesseract

    def is_available(self) -> bool:
        """Check if Tesseract is available."""
        try:
            pytesseract = self._get_tesseract()
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def extract_text(self, image: Image.Image) -> OCRResult:
        """Extract text using Tesseract."""
        pytesseract = self._get_tesseract()

        # Get full text
        full_text = pytesseract.image_to_string(
            image, lang=self.language, config=self.config
        )

        # Get detailed data for blocks
        blocks = []
        try:
            data = pytesseract.image_to_data(
                image, lang=self.language, output_type=pytesseract.Output.DICT
            )
            n_boxes = len(data["text"])
            total_conf = 0
            conf_count = 0

            for i in range(n_boxes):
                text = data["text"][i].strip()
                conf = float(data["conf"][i])

                if text and conf > 0:
                    blocks.append(
                        TextBlock(
                            text=text,
                            confidence=conf / 100.0,
                            bbox=(
                                data["left"][i],
                                data["top"][i],
                                data["width"][i],
                                data["height"][i],
                            ),
                        )
                    )
                    total_conf += conf
                    conf_count += 1

            avg_confidence = (total_conf / conf_count / 100.0) if conf_count > 0 else 0.0
        except Exception:
            avg_confidence = 0.0

        return OCRResult(
            full_text=full_text.strip(),
            blocks=blocks,
            language=self.language,
            confidence=avg_confidence,
        )


class EasyOCR(OCRBase):
    """OCR using EasyOCR."""

    def __init__(self, language: str = "eng", gpu: bool = False):
        """
        Initialize EasyOCR.

        Args:
            language: Language code (use 'en' for English)
            gpu: Whether to use GPU acceleration
        """
        # Map common language codes
        lang_map = {"eng": "en", "fra": "fr", "deu": "de", "spa": "es"}
        self._easyocr_lang = lang_map.get(language, language)
        super().__init__(language)
        self.gpu = gpu
        self._reader: Optional[object] = None

    def _get_reader(self):
        """Lazy load EasyOCR reader."""
        if self._reader is None:
            try:
                import easyocr

                self._reader = easyocr.Reader([self._easyocr_lang], gpu=self.gpu)
            except ImportError:
                raise ImportError(
                    "easyocr is required for EasyOCR. "
                    "Install with: pip install easyocr"
                )
        return self._reader

    def is_available(self) -> bool:
        """Check if EasyOCR is available."""
        try:
            self._get_reader()
            return True
        except Exception:
            return False

    def extract_text(self, image: Image.Image) -> OCRResult:
        """Extract text using EasyOCR."""
        import numpy as np

        reader = self._get_reader()

        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Run OCR
        results = reader.readtext(img_array)

        # Process results
        blocks = []
        text_parts = []
        total_conf = 0

        for bbox, text, conf in results:
            text_parts.append(text)
            total_conf += conf

            # Convert bbox to x, y, width, height
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            x, y = int(min(x_coords)), int(min(y_coords))
            w = int(max(x_coords) - x)
            h = int(max(y_coords) - y)

            blocks.append(
                TextBlock(
                    text=text,
                    confidence=conf,
                    bbox=(x, y, w, h),
                )
            )

        avg_confidence = total_conf / len(results) if results else 0.0

        return OCRResult(
            full_text=" ".join(text_parts),
            blocks=blocks,
            language=self.language,
            confidence=avg_confidence,
        )
