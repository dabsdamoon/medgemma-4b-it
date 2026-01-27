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
    "figure_caption": (
        "You are an expert medical professional specializing in medical imaging and documentation. "
        "Generate a concise, accurate caption for this medical figure. "
        "The caption should:\n"
        "- Identify the type of image (X-ray, CT, MRI, diagram, illustration, etc.)\n"
        "- Describe the anatomical region or medical concept shown\n"
        "- Note any key findings, abnormalities, or teaching points visible\n"
        "- Use precise medical terminology\n"
        "Keep the caption to 1-3 sentences."
    ),
    "figure_analysis": (
        "You are an expert medical professional. Analyze this medical figure and provide "
        "a structured JSON response with the following format:\n"
        "{\n"
        '  "figure_type": "string (x-ray|ct|mri|ultrasound|diagram|illustration|photograph|chart|other)",\n'
        '  "anatomical_region": "string describing the body region/system",\n'
        '  "entities": ["list of medical entities/structures visible"],\n'
        '  "relationships": ["list of spatial/functional relationships between entities"],\n'
        '  "findings": ["list of notable findings or abnormalities"],\n'
        '  "teaching_point": "string summarizing the main educational value",\n'
        '  "constraints": {\n'
        '    "anatomical": ["anatomical accuracy requirements for reproduction"],\n'
        '    "style": ["visual style requirements (schematic, realistic, etc.)"],\n'
        '    "labels": ["required labels or annotations"]\n'
        "  }\n"
        "}\n"
        "Respond ONLY with valid JSON, no additional text."
    ),
    "alignment_eval": (
        "You are an expert medical professional evaluating figure reproduction accuracy. "
        "You are given two images: an ORIGINAL medical figure and a GENERATED reproduction. "
        "Evaluate how well the generated image captures the medical content of the original.\n\n"
        "Provide a JSON response with this format:\n"
        "{\n"
        '  "alignment_score": 0.0-1.0 (float),\n'
        '  "status": "pass|flag|fail",\n'
        '  "anatomical_accuracy": 0.0-1.0,\n'
        '  "content_preservation": 0.0-1.0,\n'
        '  "educational_value": 0.0-1.0,\n'
        '  "reasoning": "detailed explanation of the evaluation",\n'
        '  "missing_elements": ["list of elements from original not in generated"],\n'
        '  "added_elements": ["list of elements in generated not in original"]\n'
        "}\n"
        "Score thresholds: pass >= 0.80, flag 0.65-0.80, fail < 0.65\n"
        "Respond ONLY with valid JSON, no additional text."
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
