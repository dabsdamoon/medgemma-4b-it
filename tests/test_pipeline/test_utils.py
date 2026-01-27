"""Tests for extended MedGemma utilities."""

import pytest
from PIL import Image

from medgemma.utils import encode_image, hash_image, decode_image, create_dummy_image


class TestEncodeImage:
    """Tests for encode_image function."""

    def test_encode_basic_image(self):
        """Test encoding a basic image."""
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        encoded = encode_image(img)

        assert isinstance(encoded, str)
        assert len(encoded) > 0

    def test_encode_decode_roundtrip(self):
        """Test that encoding and decoding produces same image."""
        original = Image.new("RGB", (50, 50), color=(128, 64, 32))
        encoded = encode_image(original)
        decoded = decode_image(encoded)

        # Check dimensions match
        assert decoded.size == original.size
        assert decoded.mode == "RGB"

    def test_encode_different_formats(self):
        """Test encoding with different formats."""
        img = Image.new("RGB", (64, 64), color=(0, 255, 0))

        png_encoded = encode_image(img, format="PNG")
        jpeg_encoded = encode_image(img, format="JPEG")

        # Both should be valid base64 strings
        assert isinstance(png_encoded, str)
        assert isinstance(jpeg_encoded, str)

        # PNG is typically larger for simple images
        # (though this depends on the content)
        assert len(png_encoded) > 0
        assert len(jpeg_encoded) > 0


class TestHashImage:
    """Tests for hash_image function."""

    def test_hash_returns_string(self):
        """Test that hash returns a hex string."""
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        hash_val = hash_image(img)

        assert isinstance(hash_val, str)
        assert len(hash_val) == 64  # SHA256 hex digest length

    def test_same_image_same_hash(self):
        """Test that identical images produce same hash."""
        img1 = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img2 = Image.new("RGB", (100, 100), color=(255, 0, 0))

        assert hash_image(img1) == hash_image(img2)

    def test_different_images_different_hash(self):
        """Test that different images produce different hashes."""
        img1 = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img2 = Image.new("RGB", (100, 100), color=(0, 255, 0))

        assert hash_image(img1) != hash_image(img2)

    def test_hash_deterministic(self):
        """Test that hash is deterministic."""
        img = Image.new("RGB", (50, 50), color=(100, 100, 100))

        hash1 = hash_image(img)
        hash2 = hash_image(img)

        assert hash1 == hash2


class TestDecodeImage:
    """Tests for decode_image function."""

    def test_decode_none_returns_dummy(self):
        """Test that None input returns dummy image."""
        result = decode_image(None)
        assert isinstance(result, Image.Image)
        assert result.size == (224, 224)

    def test_decode_base64_string(self):
        """Test decoding base64 string."""
        original = Image.new("RGB", (64, 64), color=(0, 0, 255))
        encoded = encode_image(original)
        decoded = decode_image(encoded)

        assert decoded.size == original.size

    def test_decode_data_url(self):
        """Test decoding base64 with data URL prefix."""
        original = Image.new("RGB", (32, 32), color=(255, 128, 0))
        encoded = encode_image(original)
        data_url = f"data:image/png;base64,{encoded}"

        decoded = decode_image(data_url)
        assert decoded.size == original.size


class TestCreateDummyImage:
    """Tests for create_dummy_image function."""

    def test_default_size(self):
        """Test default dummy image size."""
        img = create_dummy_image()
        assert img.size == (224, 224)

    def test_custom_size(self):
        """Test custom dummy image size."""
        img = create_dummy_image(size=(512, 512))
        assert img.size == (512, 512)

    def test_is_white(self):
        """Test that dummy image is white."""
        img = create_dummy_image()
        # Check center pixel is white
        assert img.getpixel((112, 112)) == (255, 255, 255)
