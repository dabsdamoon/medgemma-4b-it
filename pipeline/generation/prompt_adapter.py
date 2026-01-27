"""
Prompt adapter for converting figure analysis to image generation prompts.

Converts MedGemma analysis JSON to prompts suitable for Nano Banana Pro.
"""

from dataclasses import dataclass, field
from typing import Optional

from ..understanding.schema import FigureAnalysis, FigureType


@dataclass
class ImageBrief:
    """Structured brief for image generation."""

    prompt: str
    negative_prompt: str = ""
    style: str = "medical diagram"
    aspect_ratio: str = "1:1"
    constraints: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class PromptAdapter:
    """
    Convert FigureAnalysis to image generation prompts.

    Applies safety constraints and medical diagram styling.
    """

    # Style mappings for different figure types
    STYLE_MAP = {
        FigureType.XRAY: "medical x-ray style, grayscale, radiographic",
        FigureType.CT: "CT scan style, cross-sectional medical imaging",
        FigureType.MRI: "MRI scan style, high contrast soft tissue imaging",
        FigureType.ULTRASOUND: "ultrasound imaging style, grayscale echoic",
        FigureType.DIAGRAM: "clean medical diagram, educational illustration",
        FigureType.ILLUSTRATION: "medical textbook illustration, detailed anatomical",
        FigureType.PHOTOGRAPH: "medical photography, clinical documentation",
        FigureType.CHART: "medical chart, data visualization, clean graphs",
        FigureType.OTHER: "medical educational illustration",
    }

    # Safety constraints to always include
    SAFETY_CONSTRAINTS = [
        "no gore",
        "no graphic injuries",
        "educational medical context",
        "appropriate for textbook",
        "professional medical illustration",
    ]

    # Anatomical plausibility constraints
    ANATOMICAL_CONSTRAINTS = [
        "anatomically correct proportions",
        "plausible joint articulation",
        "accurate spatial relationships",
    ]

    def __init__(
        self,
        include_safety: bool = True,
        include_anatomical: bool = True,
        custom_style_suffix: str = "",
    ):
        """
        Initialize the prompt adapter.

        Args:
            include_safety: Include safety constraints in prompts
            include_anatomical: Include anatomical plausibility constraints
            custom_style_suffix: Additional style text to append
        """
        self.include_safety = include_safety
        self.include_anatomical = include_anatomical
        self.custom_style_suffix = custom_style_suffix

    def convert(
        self,
        analysis: FigureAnalysis,
        caption: Optional[str] = None,
    ) -> ImageBrief:
        """
        Convert a FigureAnalysis to an ImageBrief.

        Args:
            analysis: Structured analysis from MedGemma
            caption: Optional caption to include

        Returns:
            ImageBrief suitable for image generation
        """
        # Build the main prompt
        prompt_parts = []

        # Add style based on figure type
        style = self.STYLE_MAP.get(analysis.figure_type, self.STYLE_MAP[FigureType.OTHER])
        prompt_parts.append(style)

        # Add anatomical region
        if analysis.anatomical_region and analysis.anatomical_region != "unknown":
            prompt_parts.append(f"showing {analysis.anatomical_region}")

        # Add main entities
        if analysis.entities:
            entities_str = ", ".join(analysis.entities[:5])  # Limit to 5
            prompt_parts.append(f"depicting {entities_str}")

        # Add key relationships
        if analysis.relationships:
            rel_str = "; ".join(analysis.relationships[:3])  # Limit to 3
            prompt_parts.append(f"demonstrating {rel_str}")

        # Add teaching point
        if analysis.teaching_point:
            prompt_parts.append(f"illustrating: {analysis.teaching_point}")

        # Add caption if provided
        if caption or analysis.caption:
            cap = caption or analysis.caption
            prompt_parts.append(f"({cap})")

        # Add custom style suffix
        if self.custom_style_suffix:
            prompt_parts.append(self.custom_style_suffix)

        prompt = ", ".join(prompt_parts)

        # Build negative prompt
        negative_parts = [
            "blurry",
            "low quality",
            "distorted anatomy",
            "incorrect proportions",
            "unrealistic",
            "cartoon",
            "anime",
        ]

        if self.include_safety:
            negative_parts.extend([
                "gore",
                "graphic",
                "violent",
                "disturbing",
                "inappropriate",
            ])

        negative_prompt = ", ".join(negative_parts)

        # Collect all constraints
        constraints = []
        if self.include_safety:
            constraints.extend(self.SAFETY_CONSTRAINTS)
        if self.include_anatomical:
            constraints.extend(self.ANATOMICAL_CONSTRAINTS)

        # Add analysis-specific constraints
        if analysis.constraints:
            constraints.extend(analysis.constraints.anatomical)
            constraints.extend(analysis.constraints.style)

        # Determine aspect ratio based on figure type
        aspect_ratio = self._determine_aspect_ratio(analysis)

        return ImageBrief(
            prompt=prompt,
            negative_prompt=negative_prompt,
            style=style,
            aspect_ratio=aspect_ratio,
            constraints=constraints,
            metadata={
                "figure_type": analysis.figure_type.value,
                "anatomical_region": analysis.anatomical_region,
                "entity_count": len(analysis.entities),
            },
        )

    def _determine_aspect_ratio(self, analysis: FigureAnalysis) -> str:
        """Determine appropriate aspect ratio based on figure type."""
        # X-rays and scans often have specific aspect ratios
        if analysis.figure_type in [FigureType.XRAY, FigureType.CT, FigureType.MRI]:
            return "1:1"
        elif analysis.figure_type == FigureType.ULTRASOUND:
            return "4:3"
        elif analysis.figure_type == FigureType.CHART:
            return "16:9"
        else:
            return "1:1"

    def create_simple_prompt(
        self,
        description: str,
        figure_type: FigureType = FigureType.DIAGRAM,
    ) -> ImageBrief:
        """
        Create a simple prompt from a text description.

        Args:
            description: Text description of desired image
            figure_type: Type of medical figure

        Returns:
            ImageBrief for generation
        """
        style = self.STYLE_MAP.get(figure_type, self.STYLE_MAP[FigureType.OTHER])

        prompt = f"{style}, {description}"

        if self.custom_style_suffix:
            prompt += f", {self.custom_style_suffix}"

        negative_prompt = "blurry, low quality, distorted, unrealistic"
        if self.include_safety:
            negative_prompt += ", gore, graphic, violent"

        constraints = []
        if self.include_safety:
            constraints.extend(self.SAFETY_CONSTRAINTS)
        if self.include_anatomical:
            constraints.extend(self.ANATOMICAL_CONSTRAINTS)

        return ImageBrief(
            prompt=prompt,
            negative_prompt=negative_prompt,
            style=style,
            aspect_ratio="1:1",
            constraints=constraints,
            metadata={"figure_type": figure_type.value},
        )
