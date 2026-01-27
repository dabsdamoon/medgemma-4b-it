"""Tests for pipeline configuration."""

import pytest
from pathlib import Path

from pipeline.config import (
    PipelineConfig,
    PipelineMode,
    OCRBackend,
    ImageGeneratorBackend,
    AlignmentThresholds,
)


class TestPipelineConfig:
    """Tests for PipelineConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()
        assert config.mode == PipelineMode.REPRODUCIBLE
        assert config.ocr_backend == OCRBackend.TESSERACT
        assert config.image_generator == ImageGeneratorBackend.CACHED
        assert config.generation_variants == 1

    def test_reproducible_mode(self):
        """Test reproducible mode configuration."""
        config = PipelineConfig.reproducible()
        assert config.mode == PipelineMode.REPRODUCIBLE
        assert config.ocr_backend == OCRBackend.TESSERACT
        assert config.image_generator == ImageGeneratorBackend.CACHED

    def test_full_mode(self):
        """Test full mode configuration."""
        config = PipelineConfig.full()
        assert config.mode == PipelineMode.FULL
        assert config.ocr_backend == OCRBackend.GOOGLE_VISION
        assert config.image_generator == ImageGeneratorBackend.NANO_BANANA

    def test_from_mode_string(self):
        """Test creating config from mode string."""
        config = PipelineConfig.from_mode("full")
        assert config.mode == PipelineMode.FULL

        config = PipelineConfig.from_mode("reproducible")
        assert config.mode == PipelineMode.REPRODUCIBLE

    def test_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        config = PipelineConfig(output_dir="test/output")
        assert isinstance(config.output_dir, Path)
        assert config.output_dir == Path("test/output")

    def test_generation_variants_validation(self):
        """Test that generation variants are validated."""
        with pytest.raises(ValueError):
            PipelineConfig(generation_variants=0)

        with pytest.raises(ValueError):
            PipelineConfig(generation_variants=5)

    def test_alignment_thresholds(self):
        """Test alignment threshold defaults."""
        config = PipelineConfig()
        assert config.alignment_thresholds.pass_threshold == 0.80
        assert config.alignment_thresholds.flag_threshold == 0.65

    def test_custom_thresholds(self):
        """Test custom alignment thresholds."""
        thresholds = AlignmentThresholds(
            pass_threshold=0.90,
            flag_threshold=0.70,
        )
        config = PipelineConfig(alignment_thresholds=thresholds)
        assert config.alignment_thresholds.pass_threshold == 0.90
        assert config.alignment_thresholds.flag_threshold == 0.70


class TestPipelineMode:
    """Tests for PipelineMode enum."""

    def test_mode_values(self):
        """Test mode enum values."""
        assert PipelineMode.FULL.value == "full"
        assert PipelineMode.REPRODUCIBLE.value == "reproducible"


class TestOCRBackend:
    """Tests for OCRBackend enum."""

    def test_backend_values(self):
        """Test OCR backend enum values."""
        assert OCRBackend.TESSERACT.value == "tesseract"
        assert OCRBackend.EASYOCR.value == "easyocr"
        assert OCRBackend.GOOGLE_VISION.value == "google_vision"
