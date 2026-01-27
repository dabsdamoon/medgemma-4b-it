"""Tests for pipeline schema models."""

import pytest

from pipeline.understanding.schema import (
    FigureAnalysis,
    FigureType,
    AlignmentResult,
    AlignmentStatus,
    AnalysisConstraints,
)


class TestFigureType:
    """Tests for FigureType enum."""

    def test_figure_type_values(self):
        """Test figure type enum values."""
        assert FigureType.XRAY.value == "x-ray"
        assert FigureType.CT.value == "ct"
        assert FigureType.MRI.value == "mri"
        assert FigureType.DIAGRAM.value == "diagram"
        assert FigureType.OTHER.value == "other"


class TestFigureAnalysis:
    """Tests for FigureAnalysis model."""

    def test_minimal_analysis(self):
        """Test creating analysis with minimal data."""
        analysis = FigureAnalysis(
            figure_type=FigureType.DIAGRAM,
            anatomical_region="chest",
        )
        assert analysis.figure_type == FigureType.DIAGRAM
        assert analysis.anatomical_region == "chest"
        assert analysis.entities == []
        assert analysis.findings == []

    def test_full_analysis(self):
        """Test creating analysis with all fields."""
        analysis = FigureAnalysis(
            figure_type=FigureType.XRAY,
            anatomical_region="chest",
            entities=["lungs", "heart", "ribs"],
            relationships=["heart positioned centrally"],
            findings=["mild cardiomegaly"],
            teaching_point="Demonstrates normal vs enlarged heart",
            constraints=AnalysisConstraints(
                anatomical=["correct rib count"],
                style=["grayscale"],
                labels=["L", "R"],
            ),
            caption="Chest X-ray showing mild cardiomegaly",
        )
        assert len(analysis.entities) == 3
        assert len(analysis.constraints.anatomical) == 1
        assert analysis.caption is not None


class TestAlignmentResult:
    """Tests for AlignmentResult model."""

    def test_from_score_pass(self):
        """Test creating result with passing score."""
        result = AlignmentResult.from_score(
            score=0.85,
            anatomical_accuracy=0.85,
            content_preservation=0.85,
            educational_value=0.85,
        )
        assert result.status == AlignmentStatus.PASS
        assert result.alignment_score == 0.85

    def test_from_score_flag(self):
        """Test creating result with flagged score."""
        result = AlignmentResult.from_score(
            score=0.70,
            anatomical_accuracy=0.70,
            content_preservation=0.70,
            educational_value=0.70,
        )
        assert result.status == AlignmentStatus.FLAG
        assert result.alignment_score == 0.70

    def test_from_score_fail(self):
        """Test creating result with failing score."""
        result = AlignmentResult.from_score(
            score=0.50,
            anatomical_accuracy=0.50,
            content_preservation=0.50,
            educational_value=0.50,
        )
        assert result.status == AlignmentStatus.FAIL
        assert result.alignment_score == 0.50

    def test_custom_thresholds(self):
        """Test with custom thresholds."""
        result = AlignmentResult.from_score(
            score=0.75,
            pass_threshold=0.70,
            flag_threshold=0.50,
            anatomical_accuracy=0.75,
            content_preservation=0.75,
            educational_value=0.75,
        )
        assert result.status == AlignmentStatus.PASS

    def test_score_bounds(self):
        """Test that scores are bounded between 0 and 1."""
        with pytest.raises(ValueError):
            AlignmentResult(
                alignment_score=1.5,
                status=AlignmentStatus.PASS,
                anatomical_accuracy=0.8,
                content_preservation=0.8,
                educational_value=0.8,
            )

        with pytest.raises(ValueError):
            AlignmentResult(
                alignment_score=-0.1,
                status=AlignmentStatus.FAIL,
                anatomical_accuracy=0.8,
                content_preservation=0.8,
                educational_value=0.8,
            )
