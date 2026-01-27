"""
Figure detection module using OpenCV.

Detects and crops figures from page images.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass
class DetectedFigure:
    """A detected figure from a page."""

    image: Image.Image
    page_num: int
    figure_id: str
    bbox: tuple[int, int, int, int]  # x, y, width, height
    confidence: float = 1.0
    nearby_text: str = ""


class FigureDetector:
    """Detect and extract figures from page images using contour analysis."""

    def __init__(
        self,
        min_size: tuple[int, int] = (50, 50),
        max_figures: int = 10,
        margin: int = 10,
    ):
        """
        Initialize the figure detector.

        Args:
            min_size: Minimum figure dimensions (width, height)
            max_figures: Maximum figures to detect per page
            margin: Margin to add around detected figures
        """
        if cv2 is None:
            raise ImportError(
                "OpenCV is required for figure detection. "
                "Install with: pip install opencv-python"
            )
        self.min_width, self.min_height = min_size
        self.max_figures = max_figures
        self.margin = margin

    def detect(
        self,
        page_image: Image.Image,
        page_num: int,
        page_text: str = "",
    ) -> list[DetectedFigure]:
        """
        Detect figures in a page image.

        Args:
            page_image: PIL Image of the page
            page_num: Page number for identification
            page_text: Text from the page for context extraction

        Returns:
            List of DetectedFigure objects
        """
        # Convert PIL Image to OpenCV format
        img_array = np.array(page_image)
        if len(img_array.shape) == 2:
            # Grayscale
            gray = img_array
        else:
            # Color - convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Find contours using adaptive thresholding
        figures = self._find_figure_regions(gray, img_array, page_num, page_text)

        # Sort by area (largest first) and limit
        figures.sort(key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
        return figures[: self.max_figures]

    def _find_figure_regions(
        self,
        gray: np.ndarray,
        original: np.ndarray,
        page_num: int,
        page_text: str,
    ) -> list[DetectedFigure]:
        """Find figure regions using contour detection."""
        figures = []

        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter and process contours
        page_height, page_width = gray.shape
        min_area = self.min_width * self.min_height
        max_area = page_width * page_height * 0.9  # Max 90% of page

        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Filter by size
            if area < min_area or area > max_area:
                continue
            if w < self.min_width or h < self.min_height:
                continue

            # Skip if too close to page edges (likely page border)
            if x < 5 or y < 5:
                continue
            if x + w > page_width - 5 or y + h > page_height - 5:
                continue

            # Add margin
            x1 = max(0, x - self.margin)
            y1 = max(0, y - self.margin)
            x2 = min(page_width, x + w + self.margin)
            y2 = min(page_height, y + h + self.margin)

            # Crop the figure
            if len(original.shape) == 2:
                cropped = original[y1:y2, x1:x2]
            else:
                cropped = original[y1:y2, x1:x2]

            figure_image = Image.fromarray(cropped)

            # Generate figure ID
            figure_id = f"p{page_num}_f{idx}"

            # Extract nearby text (simple heuristic)
            nearby_text = self._extract_nearby_text(page_text, y, page_height)

            figures.append(
                DetectedFigure(
                    image=figure_image,
                    page_num=page_num,
                    figure_id=figure_id,
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    nearby_text=nearby_text,
                )
            )

        return figures

    def _extract_nearby_text(
        self, page_text: str, figure_y: int, page_height: int, window: int = 500
    ) -> str:
        """
        Extract text near a figure based on vertical position.

        This is a simple heuristic that estimates text position based on
        the figure's vertical position on the page.
        """
        if not page_text:
            return ""

        # Estimate relative position in text
        relative_pos = figure_y / page_height
        text_len = len(page_text)
        center_pos = int(relative_pos * text_len)

        start = max(0, center_pos - window // 2)
        end = min(text_len, center_pos + window // 2)

        return page_text[start:end].strip()

    def detect_from_embedded(
        self,
        embedded_image: Image.Image,
        page_num: int,
        index: int,
        bbox: Optional[tuple[float, float, float, float]] = None,
        page_text: str = "",
        page_height: int = 0,
    ) -> DetectedFigure:
        """
        Create a DetectedFigure from an already-extracted embedded image.

        Args:
            embedded_image: The extracted image
            page_num: Page number
            index: Image index within the page
            bbox: Bounding box if available
            page_text: Page text for context
            page_height: Page height for text extraction

        Returns:
            DetectedFigure object
        """
        figure_id = f"p{page_num}_e{index}"

        if bbox and page_text and page_height > 0:
            nearby_text = self._extract_nearby_text(
                page_text, int(bbox[1]), page_height
            )
        else:
            nearby_text = ""

        w, h = embedded_image.size
        return DetectedFigure(
            image=embedded_image,
            page_num=page_num,
            figure_id=figure_id,
            bbox=(int(bbox[0]) if bbox else 0, int(bbox[1]) if bbox else 0, w, h),
            nearby_text=nearby_text,
        )
