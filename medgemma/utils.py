"""
Utility functions for MedGemma.

Handles image decoding and other helper functions.
"""

import base64
import io
from PIL import Image


def create_dummy_image(size: tuple = (224, 224)) -> Image.Image:
    """Create a blank white image for text-only inference."""
    return Image.new("RGB", size, color=(255, 255, 255))


def decode_image(image_data: str | bytes | None) -> Image.Image:
    """
    Decode image from various input formats.

    Args:
        image_data: Can be:
            - None: Returns dummy image
            - bytes: Raw image bytes
            - str: Base64-encoded image string

    Returns:
        PIL.Image: Decoded image in RGB format
    """
    if image_data is None:
        return create_dummy_image()

    if isinstance(image_data, str):
        # Handle base64 string (may include data URL prefix)
        if image_data.startswith("data:"):
            # Remove data URL prefix (e.g., "data:image/png;base64,")
            image_data = image_data.split(",", 1)[1]
        image_bytes = base64.b64decode(image_data)
    else:
        image_bytes = image_data

    return Image.open(io.BytesIO(image_bytes)).convert("RGB")
