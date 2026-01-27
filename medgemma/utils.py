"""
Utility functions for MedGemma.

Handles image decoding and other helper functions.
"""

import base64
import hashlib
import io
from PIL import Image


def create_dummy_image(size: tuple = (224, 224)) -> Image.Image:
    """Create a blank white image for text-only inference."""
    return Image.new("RGB", size, color=(255, 255, 255))


def encode_image(image: Image.Image, format: str = "PNG") -> str:
    """
    Encode a PIL Image to a base64 string.

    Args:
        image: PIL Image to encode
        format: Image format (PNG, JPEG, etc.)

    Returns:
        str: Base64-encoded image string
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def hash_image(image: Image.Image) -> str:
    """
    Compute SHA256 hash of an image for traceability.

    Args:
        image: PIL Image to hash

    Returns:
        str: SHA256 hash string (hex digest)
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return hashlib.sha256(buffer.read()).hexdigest()


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
