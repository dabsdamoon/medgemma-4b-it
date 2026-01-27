"""
Figure filtering module.

Filters out non-medical figures (headers, logos, tables) using local heuristics.
No API calls required.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class FilterReason(Enum):
    """Reason for filtering a figure."""
    PASSED = "passed"
    ASPECT_RATIO = "aspect_ratio_too_extreme"
    TOO_SMALL = "too_small"
    LOW_COLOR_VARIANCE = "low_color_variance"
    LOW_EDGE_DENSITY = "low_edge_density"
    MOSTLY_WHITE = "mostly_white_or_empty"


@dataclass
class FilterResult:
    """Result of figure filtering."""
    is_medical: bool
    reason: FilterReason
    confidence: float  # 0-1, higher = more confident
    metrics: dict


class FigureFilter:
    """
    Filter figures to identify likely medical content.

    Uses local heuristics (no API calls):
    - Aspect ratio check
    - Size check
    - Color variance
    - Edge density
    - White space ratio
    """

    def __init__(
        self,
        max_aspect_ratio: float = 4.0,
        min_area: int = 5000,
        min_color_std: float = 20.0,
        min_edge_density: float = 0.02,
        max_white_ratio: float = 0.95,
    ):
        """
        Initialize the figure filter.

        Args:
            max_aspect_ratio: Maximum width/height or height/width ratio
            min_area: Minimum pixel area
            min_color_std: Minimum color standard deviation
            min_edge_density: Minimum ratio of edge pixels
            max_white_ratio: Maximum ratio of white/near-white pixels
        """
        self.max_aspect_ratio = max_aspect_ratio
        self.min_area = min_area
        self.min_color_std = min_color_std
        self.min_edge_density = min_edge_density
        self.max_white_ratio = max_white_ratio

    def filter(self, image: Image.Image) -> FilterResult:
        """
        Check if an image is likely a medical figure.

        Args:
            image: PIL Image to check

        Returns:
            FilterResult with classification and metrics
        """
        metrics = {}

        # Convert to numpy array
        img_array = np.array(image.convert("RGB"))
        height, width = img_array.shape[:2]

        # 1. Aspect ratio check
        aspect_ratio = max(width / height, height / width)
        metrics["aspect_ratio"] = aspect_ratio

        if aspect_ratio > self.max_aspect_ratio:
            return FilterResult(
                is_medical=False,
                reason=FilterReason.ASPECT_RATIO,
                confidence=min(1.0, aspect_ratio / 10),
                metrics=metrics,
            )

        # 2. Size check
        area = width * height
        metrics["area"] = area

        if area < self.min_area:
            return FilterResult(
                is_medical=False,
                reason=FilterReason.TOO_SMALL,
                confidence=0.8,
                metrics=metrics,
            )

        # 3. White space ratio
        gray = np.mean(img_array, axis=2)
        white_ratio = np.mean(gray > 240)
        metrics["white_ratio"] = white_ratio

        if white_ratio > self.max_white_ratio:
            return FilterResult(
                is_medical=False,
                reason=FilterReason.MOSTLY_WHITE,
                confidence=white_ratio,
                metrics=metrics,
            )

        # 4. Color variance check
        color_std = np.std(img_array)
        metrics["color_std"] = color_std

        if color_std < self.min_color_std:
            return FilterResult(
                is_medical=False,
                reason=FilterReason.LOW_COLOR_VARIANCE,
                confidence=0.7,
                metrics=metrics,
            )

        # 5. Edge density check (using simple gradient)
        edge_density = self._compute_edge_density(gray)
        metrics["edge_density"] = edge_density

        if edge_density < self.min_edge_density:
            return FilterResult(
                is_medical=False,
                reason=FilterReason.LOW_EDGE_DENSITY,
                confidence=0.6,
                metrics=metrics,
            )

        # Passed all checks
        confidence = self._compute_medical_confidence(metrics)
        return FilterResult(
            is_medical=True,
            reason=FilterReason.PASSED,
            confidence=confidence,
            metrics=metrics,
        )

    def _compute_edge_density(self, gray: np.ndarray) -> float:
        """Compute ratio of edge pixels using gradient magnitude."""
        # Simple Sobel-like gradient
        gx = np.abs(np.diff(gray, axis=1))
        gy = np.abs(np.diff(gray, axis=0))

        # Threshold for edge detection
        edge_threshold = 30

        edges_x = gx > edge_threshold
        edges_y = gy > edge_threshold

        edge_ratio = (np.mean(edges_x) + np.mean(edges_y)) / 2
        return edge_ratio

    def _compute_medical_confidence(self, metrics: dict) -> float:
        """Compute confidence that image is medical based on metrics."""
        confidence = 0.5

        # Good aspect ratio (closer to 1:1 or 2:1)
        ar = metrics.get("aspect_ratio", 1)
        if ar < 2:
            confidence += 0.2
        elif ar < 3:
            confidence += 0.1

        # Good edge density (complex internal structure)
        ed = metrics.get("edge_density", 0)
        if ed > 0.1:
            confidence += 0.2
        elif ed > 0.05:
            confidence += 0.1

        # Reasonable white ratio (not too empty)
        wr = metrics.get("white_ratio", 0)
        if wr < 0.7:
            confidence += 0.1

        return min(1.0, confidence)

    def filter_batch(
        self,
        images: list[tuple[str, Image.Image]],
    ) -> tuple[list[tuple[str, Image.Image]], list[tuple[str, FilterResult]]]:
        """
        Filter a batch of images.

        Args:
            images: List of (figure_id, image) tuples

        Returns:
            Tuple of (passed_images, filtered_results)
        """
        passed = []
        filtered = []

        for figure_id, image in images:
            result = self.filter(image)
            if result.is_medical:
                passed.append((figure_id, image))
            filtered.append((figure_id, result))

            logger.debug(
                f"Figure {figure_id}: {'KEEP' if result.is_medical else 'SKIP'} "
                f"({result.reason.value}, conf={result.confidence:.2f})"
            )

        return passed, filtered


def test_filter_on_directory(figures_dir: str):
    """Test the filter on a directory of figures."""
    from pathlib import Path

    figures_path = Path(figures_dir)
    filter = FigureFilter()

    print(f"{'Figure':<45} {'Size':<12} {'Result':<8} {'Reason':<25} {'Conf':<6}")
    print("-" * 100)

    kept = 0
    skipped = 0

    for img_path in sorted(figures_path.glob("*.png")):
        image = Image.open(img_path)
        result = filter.filter(image)

        w, h = image.size
        status = "KEEP" if result.is_medical else "SKIP"

        if result.is_medical:
            kept += 1
        else:
            skipped += 1

        print(
            f"{img_path.name:<45} {w:>4}x{h:<6} {status:<8} "
            f"{result.reason.value:<25} {result.confidence:.2f}"
        )

    print("-" * 100)
    print(f"Total: {kept + skipped} | Kept: {kept} | Skipped: {skipped}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_filter_on_directory(sys.argv[1])
    else:
        print("Usage: python figure_filter.py <figures_directory>")
