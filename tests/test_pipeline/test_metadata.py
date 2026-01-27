"""Tests for metadata schema."""

import pytest
from datetime import datetime

from pipeline.output.metadata import DocumentMetadata, FigureMetadata
from pipeline.understanding.schema import AlignmentStatus


class TestFigureMetadata:
    """Tests for FigureMetadata class."""

    def test_minimal_creation(self):
        """Test creating figure metadata with minimal data."""
        meta = FigureMetadata(
            figure_id="p0_f0",
            page_num=0,
            original_hash="abc123",
        )
        assert meta.figure_id == "p0_f0"
        assert meta.page_num == 0
        assert meta.original_hash == "abc123"
        assert meta.generated_hash is None

    def test_full_creation(self):
        """Test creating figure metadata with all fields."""
        meta = FigureMetadata(
            figure_id="p1_f2",
            page_num=1,
            original_hash="abc123",
            generated_hash="def456",
            figure_type="diagram",
            anatomical_region="chest",
            caption="Chest anatomy diagram",
            teaching_point="Normal chest anatomy",
            generation_prompt="medical diagram of chest",
            generation_seed=42,
            alignment_score=0.85,
            alignment_status="pass",
            alignment_reasoning="Good anatomical accuracy",
            bbox=(10, 20, 100, 150),
            nearby_text="Figure 1 shows...",
        )
        assert meta.alignment_score == 0.85
        assert meta.bbox == (10, 20, 100, 150)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        meta = FigureMetadata(
            figure_id="p0_f0",
            page_num=0,
            original_hash="abc123",
            figure_type="x-ray",
        )
        d = meta.to_dict()
        assert d["figure_id"] == "p0_f0"
        assert d["figure_type"] == "x-ray"
        assert "processed_at" in d


class TestDocumentMetadata:
    """Tests for DocumentMetadata class."""

    def test_creation(self):
        """Test creating document metadata."""
        meta = DocumentMetadata(
            source_filename="test.pdf",
            source_hash="abc123",
            page_count=10,
        )
        assert meta.source_filename == "test.pdf"
        assert meta.page_count == 10
        assert meta.figures_processed == 0

    def test_add_figure(self):
        """Test adding figure to metadata."""
        meta = DocumentMetadata(
            source_filename="test.pdf",
            source_hash="abc123",
            page_count=5,
        )

        fig = FigureMetadata(
            figure_id="p0_f0",
            page_num=0,
            original_hash="fig123",
            alignment_score=0.85,
            alignment_status=AlignmentStatus.PASS.value,
        )

        meta.add_figure(fig)

        assert meta.figures_processed == 1
        assert meta.alignment_pass_count == 1
        assert len(meta.figures) == 1

    def test_alignment_tracking(self):
        """Test alignment statistics tracking."""
        meta = DocumentMetadata(
            source_filename="test.pdf",
            source_hash="abc123",
            page_count=5,
        )

        # Add passing figure
        meta.add_figure(FigureMetadata(
            figure_id="p0_f0",
            page_num=0,
            original_hash="h1",
            alignment_score=0.90,
            alignment_status=AlignmentStatus.PASS.value,
        ))

        # Add flagged figure
        meta.add_figure(FigureMetadata(
            figure_id="p0_f1",
            page_num=0,
            original_hash="h2",
            alignment_score=0.70,
            alignment_status=AlignmentStatus.FLAG.value,
        ))

        # Add failing figure
        meta.add_figure(FigureMetadata(
            figure_id="p1_f0",
            page_num=1,
            original_hash="h3",
            alignment_score=0.50,
            alignment_status=AlignmentStatus.FAIL.value,
        ))

        assert meta.alignment_pass_count == 1
        assert meta.alignment_flag_count == 1
        assert meta.alignment_fail_count == 1
        assert meta.average_alignment_score == pytest.approx(0.70, 0.01)

    def test_mark_completed(self):
        """Test marking processing as completed."""
        meta = DocumentMetadata(
            source_filename="test.pdf",
            source_hash="abc123",
            page_count=5,
        )
        assert meta.processing_completed is None

        meta.mark_completed()
        assert meta.processing_completed is not None

    def test_add_error(self):
        """Test recording errors."""
        meta = DocumentMetadata(
            source_filename="test.pdf",
            source_hash="abc123",
            page_count=5,
        )

        meta.add_error("Failed to process figure p0_f0")
        meta.add_error("OCR timeout on page 3")

        assert len(meta.errors) == 2

    def test_to_dict(self):
        """Test conversion to dictionary for JSON serialization."""
        meta = DocumentMetadata(
            source_filename="test.pdf",
            source_hash="abc123",
            page_count=5,
            pipeline_mode="reproducible",
        )

        d = meta.to_dict()

        assert d["source"]["filename"] == "test.pdf"
        assert d["pipeline"]["mode"] == "reproducible"
        assert "figures" in d
        assert "alignment" in d
