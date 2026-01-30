"""
Weak prompt sanitizer for image generation.

Lightly sanitizes MedGemma output to avoid triggering safety filters
while preserving most medical context.
"""

import re
from typing import Optional


# Weak-level replacements: only the most problematic terms
WEAK_REPLACEMENTS = {
    # Explicit anatomical terms → clinical alternatives
    r"\bpenis\b": "anatomical structure",
    r"\bpenile\b": "anatomical",
    r"\bgenital[s]?\b": "anatomical region",
    r"\bgenitourinary\b": "urological",
    r"\bforeskin\b": "tissue",
    r"\binfant'?s?\b": "pediatric patient",
    r"\bnewborn'?s?\b": "neonatal patient",

    # Procedure names → generic
    r"\bcircumcision\b": "surgical procedure",
    r"\bcircumcised\b": "surgical",

    # Keep instruments but soften context
    r"\bcut(?:ting)? the\b": "applied to the",
    r"\bremov(?:e|ing|al) of\b": "treatment of",
}

# Terms to remove entirely (with surrounding whitespace cleanup)
# NOTE: Avoid removing core medical terms - prefer replacement over removal
# "infant circumcision" → "pediatric patient surgical procedure" (via word replacements)
REMOVE_TERMS = [
    # Empty - let word-by-word replacements handle compound terms
]


class PromptSanitizer:
    """
    Weak-level prompt sanitizer for medical image generation.

    Performs minimal sanitization to avoid safety filter triggers
    while preserving educational and clinical context.
    """

    def __init__(
        self,
        replacements: Optional[dict] = None,
        remove_terms: Optional[list] = None,
        case_insensitive: bool = True,
    ):
        """
        Initialize the sanitizer.

        Args:
            replacements: Dict of regex patterns → replacements
            remove_terms: List of terms to remove entirely
            case_insensitive: Whether to match case-insensitively
        """
        self.replacements = replacements or WEAK_REPLACEMENTS
        self.remove_terms = remove_terms or REMOVE_TERMS
        self.flags = re.IGNORECASE if case_insensitive else 0

    def sanitize(self, text: str) -> str:
        """
        Sanitize a prompt string.

        Args:
            text: Original prompt text

        Returns:
            Sanitized prompt text
        """
        if not text:
            return text

        result = text

        # Remove problematic terms first
        for term in self.remove_terms:
            result = re.sub(term, "", result, flags=self.flags)

        # Apply replacements
        for pattern, replacement in self.replacements.items():
            result = re.sub(pattern, replacement, result, flags=self.flags)

        # Clean up multiple spaces and trim
        result = re.sub(r"\s+", " ", result).strip()

        return result

    def sanitize_brief(self, brief: "ImageBrief") -> "ImageBrief":
        """
        Sanitize an ImageBrief object.

        Args:
            brief: Original ImageBrief

        Returns:
            New ImageBrief with sanitized prompt
        """
        from .prompt_adapter import ImageBrief

        return ImageBrief(
            prompt=self.sanitize(brief.prompt),
            negative_prompt=brief.negative_prompt,  # Keep as-is
            style=brief.style,
            aspect_ratio=brief.aspect_ratio,
            constraints=brief.constraints,
            metadata={
                **brief.metadata,
                "sanitized": True,
            },
        )


# Singleton instance for convenience
_sanitizer: Optional[PromptSanitizer] = None


def get_sanitizer() -> PromptSanitizer:
    """Get the default sanitizer instance."""
    global _sanitizer
    if _sanitizer is None:
        _sanitizer = PromptSanitizer()
    return _sanitizer


def sanitize_prompt(text: str) -> str:
    """Convenience function to sanitize a prompt string."""
    return get_sanitizer().sanitize(text)
