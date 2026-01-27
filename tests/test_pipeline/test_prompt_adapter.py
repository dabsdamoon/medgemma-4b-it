"""Tests for prompt adapter."""

import pytest

from pipeline.generation.prompt_adapter import PromptAdapter, ImageBrief
from pipeline.understanding.schema import (
    FigureAnalysis,
    FigureType,
    AnalysisConstraints,
)


class TestPromptAdapter:
    """Tests for PromptAdapter class."""

    def test_basic_conversion(self):
        """Test basic analysis to brief conversion."""
        adapter = PromptAdapter()

        analysis = FigureAnalysis(
            figure_type=FigureType.DIAGRAM,
            anatomical_region="chest cavity",
            entities=["lungs", "heart"],
            teaching_point="Demonstrates normal chest anatomy",
        )

        brief = adapter.convert(analysis)

        assert isinstance(brief, ImageBrief)
        assert "medical diagram" in brief.style.lower()
        assert "chest cavity" in brief.prompt.lower()
        assert brief.aspect_ratio == "1:1"

    def test_safety_constraints_included(self):
        """Test that safety constraints are included by default."""
        adapter = PromptAdapter(include_safety=True)

        analysis = FigureAnalysis(
            figure_type=FigureType.PHOTOGRAPH,
            anatomical_region="wound",
        )

        brief = adapter.convert(analysis)

        assert "gore" in brief.negative_prompt.lower()
        assert any("no gore" in c.lower() for c in brief.constraints)

    def test_safety_constraints_disabled(self):
        """Test that safety constraints can be disabled."""
        adapter = PromptAdapter(include_safety=False, include_anatomical=False)

        analysis = FigureAnalysis(
            figure_type=FigureType.DIAGRAM,
            anatomical_region="test",
        )

        brief = adapter.convert(analysis)

        # Safety constraints should not be in the list
        assert not any("no gore" in c.lower() for c in brief.constraints)

    def test_figure_type_styling(self):
        """Test that different figure types get appropriate styles."""
        adapter = PromptAdapter()

        xray_analysis = FigureAnalysis(
            figure_type=FigureType.XRAY,
            anatomical_region="chest",
        )
        xray_brief = adapter.convert(xray_analysis)
        assert "x-ray" in xray_brief.style.lower() or "radiographic" in xray_brief.style.lower()

        diagram_analysis = FigureAnalysis(
            figure_type=FigureType.DIAGRAM,
            anatomical_region="anatomy",
        )
        diagram_brief = adapter.convert(diagram_analysis)
        assert "diagram" in diagram_brief.style.lower()

    def test_aspect_ratio_by_type(self):
        """Test that aspect ratios vary by figure type."""
        adapter = PromptAdapter()

        chart_analysis = FigureAnalysis(
            figure_type=FigureType.CHART,
            anatomical_region="data",
        )
        chart_brief = adapter.convert(chart_analysis)
        assert chart_brief.aspect_ratio == "16:9"

        ultrasound_analysis = FigureAnalysis(
            figure_type=FigureType.ULTRASOUND,
            anatomical_region="fetal",
        )
        ultrasound_brief = adapter.convert(ultrasound_analysis)
        assert ultrasound_brief.aspect_ratio == "4:3"

    def test_simple_prompt_creation(self):
        """Test creating simple prompt from description."""
        adapter = PromptAdapter()

        brief = adapter.create_simple_prompt(
            "diagram of human heart with labeled chambers",
            figure_type=FigureType.DIAGRAM,
        )

        assert "heart" in brief.prompt.lower()
        assert "labeled" in brief.prompt.lower()
        assert isinstance(brief.constraints, list)

    def test_metadata_populated(self):
        """Test that metadata is populated in brief."""
        adapter = PromptAdapter()

        analysis = FigureAnalysis(
            figure_type=FigureType.MRI,
            anatomical_region="brain",
            entities=["cortex", "ventricles", "white matter"],
        )

        brief = adapter.convert(analysis)

        assert brief.metadata["figure_type"] == "mri"
        assert brief.metadata["anatomical_region"] == "brain"
        assert brief.metadata["entity_count"] == 3
