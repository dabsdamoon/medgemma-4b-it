"""Tests for extended MedGemma prompts."""

import pytest

from medgemma.prompts import DEFAULT_PROMPTS, get_prompt, list_modes


class TestExtendedPrompts:
    """Tests for the new prompt modes."""

    def test_figure_caption_mode_exists(self):
        """Test that figure_caption mode is available."""
        assert "figure_caption" in DEFAULT_PROMPTS
        assert "figure_caption" in list_modes()

    def test_figure_analysis_mode_exists(self):
        """Test that figure_analysis mode is available."""
        assert "figure_analysis" in DEFAULT_PROMPTS
        assert "figure_analysis" in list_modes()

    def test_alignment_eval_mode_exists(self):
        """Test that alignment_eval mode is available."""
        assert "alignment_eval" in DEFAULT_PROMPTS
        assert "alignment_eval" in list_modes()

    def test_figure_caption_prompt_content(self):
        """Test figure_caption prompt has expected content."""
        prompt = get_prompt("figure_caption")
        assert "caption" in prompt.lower()
        assert "medical" in prompt.lower()
        assert "terminology" in prompt.lower()

    def test_figure_analysis_prompt_content(self):
        """Test figure_analysis prompt has expected content."""
        prompt = get_prompt("figure_analysis")
        assert "json" in prompt.lower()
        assert "entities" in prompt.lower()
        assert "relationships" in prompt.lower()
        assert "constraints" in prompt.lower()

    def test_alignment_eval_prompt_content(self):
        """Test alignment_eval prompt has expected content."""
        prompt = get_prompt("alignment_eval")
        assert "alignment_score" in prompt.lower()
        assert "pass" in prompt.lower()
        assert "0.80" in prompt or "0.8" in prompt

    def test_get_prompt_with_custom_override(self):
        """Test that custom prompts override defaults."""
        custom = "My custom prompt"
        prompt = get_prompt("figure_caption", custom_prompt=custom)
        assert prompt == custom

    def test_get_prompt_fallback(self):
        """Test that unknown modes fall back to general."""
        prompt = get_prompt("nonexistent_mode")
        assert prompt == DEFAULT_PROMPTS["general"]
