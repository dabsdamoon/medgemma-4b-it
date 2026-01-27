"""
Pydantic models for MedGemma analysis outputs.

Defines structured schemas for figure analysis and alignment evaluation.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class FigureType(str, Enum):
    """Types of medical figures."""

    XRAY = "x-ray"
    CT = "ct"
    MRI = "mri"
    ULTRASOUND = "ultrasound"
    DIAGRAM = "diagram"
    ILLUSTRATION = "illustration"
    PHOTOGRAPH = "photograph"
    CHART = "chart"
    OTHER = "other"


class AlignmentStatus(str, Enum):
    """Alignment evaluation status."""

    PASS = "pass"
    FLAG = "flag"
    FAIL = "fail"


class AnalysisConstraints(BaseModel):
    """Constraints for figure reproduction."""

    anatomical: list[str] = Field(
        default_factory=list,
        description="Anatomical accuracy requirements",
    )
    style: list[str] = Field(
        default_factory=list,
        description="Visual style requirements",
    )
    labels: list[str] = Field(
        default_factory=list,
        description="Required labels or annotations",
    )


class FigureAnalysis(BaseModel):
    """Structured analysis of a medical figure."""

    figure_type: FigureType = Field(
        description="Type of medical figure",
    )
    anatomical_region: str = Field(
        description="Body region or system shown",
    )
    entities: list[str] = Field(
        default_factory=list,
        description="Medical entities/structures visible",
    )
    relationships: list[str] = Field(
        default_factory=list,
        description="Spatial/functional relationships between entities",
    )
    findings: list[str] = Field(
        default_factory=list,
        description="Notable findings or abnormalities",
    )
    teaching_point: str = Field(
        default="",
        description="Main educational value of the figure",
    )
    constraints: AnalysisConstraints = Field(
        default_factory=AnalysisConstraints,
        description="Reproduction constraints",
    )

    # Optional metadata
    caption: Optional[str] = Field(
        default=None,
        description="Generated caption for the figure",
    )
    context: Optional[str] = Field(
        default=None,
        description="Surrounding text context",
    )


class AlignmentResult(BaseModel):
    """Result of alignment evaluation between original and generated figures."""

    alignment_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall alignment score (0-1)",
    )
    status: AlignmentStatus = Field(
        description="Evaluation status (pass/flag/fail)",
    )
    anatomical_accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Anatomical accuracy score",
    )
    content_preservation: float = Field(
        ge=0.0,
        le=1.0,
        description="Content preservation score",
    )
    educational_value: float = Field(
        ge=0.0,
        le=1.0,
        description="Educational value preservation score",
    )
    reasoning: str = Field(
        default="",
        description="Detailed explanation of the evaluation",
    )
    missing_elements: list[str] = Field(
        default_factory=list,
        description="Elements from original not in generated",
    )
    added_elements: list[str] = Field(
        default_factory=list,
        description="Elements in generated not in original",
    )

    @classmethod
    def from_score(
        cls,
        score: float,
        pass_threshold: float = 0.80,
        flag_threshold: float = 0.65,
        **kwargs,
    ) -> "AlignmentResult":
        """Create an AlignmentResult with status derived from score."""
        if score >= pass_threshold:
            status = AlignmentStatus.PASS
        elif score >= flag_threshold:
            status = AlignmentStatus.FLAG
        else:
            status = AlignmentStatus.FAIL

        return cls(alignment_score=score, status=status, **kwargs)
