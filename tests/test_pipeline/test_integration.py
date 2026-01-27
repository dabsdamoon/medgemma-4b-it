"""Integration tests for the pipeline.

These tests verify that all components work together correctly.
Note: Some tests require optional dependencies or may be skipped.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from PIL import Image


class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        return Image.new("RGB", (200, 200), color=(128, 128, 128))

    def test_config_to_orchestrator(self, temp_output_dir):
        """Test that config properly initializes orchestrator."""
        from pipeline.config import PipelineConfig
        from pipeline.orchestrator import DocumentModernizer

        config = PipelineConfig.reproducible(output_dir=temp_output_dir)
        modernizer = DocumentModernizer(config)

        assert modernizer.config.mode.value == "reproducible"

    def test_bundler_creates_structure(self, temp_output_dir):
        """Test that bundler creates correct directory structure."""
        from pipeline.output.bundler import OutputBundler
        from pipeline.output.metadata import DocumentMetadata

        bundler = OutputBundler(temp_output_dir)

        metadata = DocumentMetadata(
            source_filename="test.pdf",
            source_hash="abc123",
            page_count=5,
        )

        bundle = bundler.initialize(metadata)

        assert bundle.output_dir.exists()
        assert bundle.figures_dir.exists()

    def test_bundler_saves_text(self, temp_output_dir):
        """Test that bundler saves OCR text correctly."""
        from pipeline.output.bundler import OutputBundler
        from pipeline.output.metadata import DocumentMetadata

        bundler = OutputBundler(temp_output_dir)
        metadata = DocumentMetadata(
            source_filename="test.pdf",
            source_hash="abc123",
            page_count=1,
        )
        bundler.initialize(metadata)

        ocr_text = "Sample OCR extracted text"
        path = bundler.save_ocr_text(ocr_text)

        assert path.exists()
        assert path.read_text() == ocr_text

    def test_bundler_saves_figures(self, temp_output_dir, sample_image):
        """Test that bundler saves figure images correctly."""
        from pipeline.output.bundler import OutputBundler
        from pipeline.output.metadata import DocumentMetadata, FigureMetadata

        bundler = OutputBundler(temp_output_dir)
        metadata = DocumentMetadata(
            source_filename="test.pdf",
            source_hash="abc123",
            page_count=1,
        )
        bundler.initialize(metadata)

        fig_meta = FigureMetadata(
            figure_id="p0_f0",
            page_num=0,
            original_hash="orghash",
        )

        original_path, generated_path = bundler.save_figure(
            "p0_f0",
            sample_image,
            sample_image,  # Use same image as "generated"
            fig_meta,
        )

        assert original_path.exists()
        assert generated_path.exists()

    def test_bundler_finalize(self, temp_output_dir, sample_image):
        """Test that bundler creates complete output bundle."""
        from pipeline.output.bundler import OutputBundler
        from pipeline.output.metadata import DocumentMetadata, FigureMetadata

        bundler = OutputBundler(temp_output_dir)
        metadata = DocumentMetadata(
            source_filename="test.pdf",
            source_hash="abc123",
            page_count=1,
        )
        bundler.initialize(metadata)
        bundler.save_ocr_text("Test text")

        fig_meta = FigureMetadata(
            figure_id="p0_f0",
            page_num=0,
            original_hash="orghash",
            alignment_score=0.85,
            alignment_status="pass",
        )
        bundler.save_figure("p0_f0", sample_image, sample_image, fig_meta)

        bundle = bundler.finalize()

        assert bundle.metadata_path.exists()
        assert bundle.ocr_text_path.exists()
        assert len(bundle.original_images) == 1

    def test_prompt_adapter_to_brief(self, sample_image):
        """Test prompt adapter produces valid brief."""
        from pipeline.generation.prompt_adapter import PromptAdapter
        from pipeline.understanding.schema import FigureAnalysis, FigureType

        adapter = PromptAdapter()
        analysis = FigureAnalysis(
            figure_type=FigureType.DIAGRAM,
            anatomical_region="heart",
            entities=["left ventricle", "right ventricle"],
            teaching_point="Cardiac anatomy",
        )

        brief = adapter.convert(analysis)

        assert brief.prompt
        assert "heart" in brief.prompt.lower()
        assert brief.constraints

    @pytest.mark.skipif(
        True,  # Skip by default as it requires model
        reason="Requires MedGemma model loaded"
    )
    def test_captioner_generates_caption(self, sample_image):
        """Test that captioner can generate captions."""
        from pipeline.understanding.captioner import FigureCaptioner

        captioner = FigureCaptioner(use_quantization=True)
        caption = captioner.generate_caption(sample_image, context="Test image")

        assert isinstance(caption, str)
        assert len(caption) > 0


class TestAlignmentScoring:
    """Tests for alignment scoring functionality."""

    @pytest.fixture
    def sample_images(self):
        """Create sample test images."""
        original = Image.new("RGB", (256, 256), color=(100, 100, 100))
        generated = Image.new("RGB", (256, 256), color=(110, 110, 110))
        return original, generated

    def test_alignment_result_status_pass(self):
        """Test alignment result with passing score."""
        from pipeline.understanding.schema import AlignmentResult, AlignmentStatus

        result = AlignmentResult.from_score(
            score=0.85,
            anatomical_accuracy=0.85,
            content_preservation=0.85,
            educational_value=0.85,
            reasoning="Good alignment",
        )

        assert result.status == AlignmentStatus.PASS

    def test_alignment_result_status_flag(self):
        """Test alignment result with flagged score."""
        from pipeline.understanding.schema import AlignmentResult, AlignmentStatus

        result = AlignmentResult.from_score(
            score=0.72,
            anatomical_accuracy=0.72,
            content_preservation=0.72,
            educational_value=0.72,
            reasoning="Partial alignment",
        )

        assert result.status == AlignmentStatus.FLAG

    def test_alignment_result_status_fail(self):
        """Test alignment result with failing score."""
        from pipeline.understanding.schema import AlignmentResult, AlignmentStatus

        result = AlignmentResult.from_score(
            score=0.45,
            anatomical_accuracy=0.45,
            content_preservation=0.45,
            educational_value=0.45,
            reasoning="Poor alignment",
        )

        assert result.status == AlignmentStatus.FAIL

    def test_scorer_summary_stats(self):
        """Test alignment scorer summary statistics."""
        from pipeline.evaluation.alignment_scorer import AlignmentScorer
        from pipeline.understanding.schema import AlignmentResult, AlignmentStatus

        results = [
            AlignmentResult.from_score(
                score=0.90,
                anatomical_accuracy=0.90,
                content_preservation=0.90,
                educational_value=0.90,
            ),
            AlignmentResult.from_score(
                score=0.70,
                anatomical_accuracy=0.70,
                content_preservation=0.70,
                educational_value=0.70,
            ),
            AlignmentResult.from_score(
                score=0.50,
                anatomical_accuracy=0.50,
                content_preservation=0.50,
                educational_value=0.50,
            ),
        ]

        scorer = AlignmentScorer()
        stats = scorer.get_summary_stats(results)

        assert stats["count"] == 3
        assert stats["pass_count"] == 1
        assert stats["flag_count"] == 1
        assert stats["fail_count"] == 1
        assert stats["avg_score"] == pytest.approx(0.70, 0.01)
