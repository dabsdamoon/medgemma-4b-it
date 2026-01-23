"""
Prompt templates for MedGemma.

This module manages system prompts for different modes.
Prompts can be overridden via API input without rebuilding the Docker image.
"""

# Default prompts by mode
DEFAULT_PROMPTS = {
    "summarize": (
        "You are an expert medical professional. "
        "Provide a clear, accurate, and concise summary of the following medical document. "
        "Focus on key findings, diagnoses, treatments, and recommendations."
    ),
    "general": (
        "You are an expert medical professional. "
        "Provide a clear, accurate, and concise response to the following inputs. "
        "Focus on key findings, diagnoses, treatments, and recommendations if applicable."
    ),
}


def get_prompt(mode: str, custom_prompt: str | None = None) -> str:
    """
    Get the system prompt for a given mode.

    Args:
        mode: The processing mode ("summarize", "general", or custom)
        custom_prompt: Optional custom prompt that overrides the default

    Returns:
        str: The system prompt to use
    """
    if custom_prompt:
        return custom_prompt

    return DEFAULT_PROMPTS.get(mode, DEFAULT_PROMPTS["general"])


def list_modes() -> list[str]:
    """Return available default modes."""
    return list(DEFAULT_PROMPTS.keys())
