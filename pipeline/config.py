"""
Pipeline configuration module.

Defines configuration settings and modes for the document modernizer pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class PipelineMode(Enum):
    """Pipeline execution modes."""

    FULL = "full"  # Full pipeline with all API calls
    REPRODUCIBLE = "reproducible"  # Local OCR, cached outputs for demos


class OCRBackend(Enum):
    """OCR backend options."""

    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    GOOGLE_VISION = "google_vision"


class ImageGeneratorBackend(Enum):
    """Image generation backend options."""

    # Vertex AI backends (recommended)
    VERTEX_FLASH = "vertex_flash"  # Nano Banana (Gemini 2.5 Flash) - cheaper, default
    VERTEX_PRO = "vertex_pro"  # Nano Banana Pro (Gemini 3 Pro) - higher quality

    # Direct API (legacy, requires separate API key)
    NANO_BANANA = "nano_banana"

    # Cached images for testing/demos
    CACHED = "cached"


@dataclass
class AlignmentThresholds:
    """Thresholds for alignment evaluation."""

    pass_threshold: float = 0.80
    flag_threshold: float = 0.65
    # Below flag_threshold is considered fail


@dataclass
class PipelineConfig:
    """Configuration for the document modernizer pipeline."""

    # Pipeline mode
    mode: PipelineMode = PipelineMode.REPRODUCIBLE
    extract_only: bool = False  # If True, skip AI analysis (local extraction only)
    analyze_only: bool = False  # If True, run analysis but skip image generation

    # Input/Output paths
    input_path: Optional[Path] = None
    output_dir: Path = field(default_factory=lambda: Path("output"))
    cache_dir: Path = field(default_factory=lambda: Path("demo/cached_examples"))

    # OCR settings
    ocr_backend: OCRBackend = OCRBackend.TESSERACT
    ocr_language: str = "eng"

    # Image generation settings
    image_generator: ImageGeneratorBackend = ImageGeneratorBackend.CACHED
    generation_variants: int = 1  # Number of variants per figure (1-4)
    generation_strength: float = 0.75  # How much to transform (0=identical, 1=full change)
    sanitize_prompts: bool = True  # Sanitize prompts to avoid safety filter blocks

    # Vertex AI settings (for VERTEX_FLASH and VERTEX_PRO backends)
    gcp_project_id: Optional[str] = None  # Defaults to GOOGLE_CLOUD_PROJECT env var
    gcp_location: str = "us-central1"

    # Direct Nano Banana API settings (legacy, for NANO_BANANA backend)
    nano_banana_api_key: Optional[str] = None
    nano_banana_base_url: str = "https://api.nanobanana.com/v1"

    # MedGemma settings
    use_quantization: bool = True
    max_tokens: int = 1024

    # Alignment evaluation
    run_alignment_eval: bool = False  # If True, run alignment evaluation after generation
    alignment_thresholds: AlignmentThresholds = field(
        default_factory=AlignmentThresholds
    )

    # Processing options
    extract_context_window: int = 500  # Characters around figure to extract
    min_figure_size: tuple[int, int] = (50, 50)  # Minimum figure dimensions
    max_figures_per_page: int = 10

    # Figure filtering (local heuristics to skip non-medical figures)
    filter_figures: bool = True  # Enable local figure filtering
    filter_max_aspect_ratio: float = 4.0  # Skip figures with extreme aspect ratios

    # Debug/logging
    verbose: bool = False
    save_intermediates: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        if isinstance(self.input_path, str):
            self.input_path = Path(self.input_path)

        # Validate generation variants
        if not 1 <= self.generation_variants <= 4:
            raise ValueError("generation_variants must be between 1 and 4")

        # Set defaults based on mode
        if self.mode == PipelineMode.FULL:
            if self.ocr_backend == OCRBackend.TESSERACT:
                self.ocr_backend = OCRBackend.GOOGLE_VISION
            if self.image_generator == ImageGeneratorBackend.CACHED:
                # Default to Vertex Flash (Nano Banana) - cheaper option
                self.image_generator = ImageGeneratorBackend.VERTEX_FLASH

    @classmethod
    def from_mode(cls, mode: str | PipelineMode, **kwargs) -> "PipelineConfig":
        """Create configuration from a mode string."""
        if isinstance(mode, str):
            mode = PipelineMode(mode)
        return cls(mode=mode, **kwargs)

    @classmethod
    def reproducible(cls, **kwargs) -> "PipelineConfig":
        """Create a reproducible mode configuration."""
        return cls(
            mode=PipelineMode.REPRODUCIBLE,
            ocr_backend=OCRBackend.TESSERACT,
            image_generator=ImageGeneratorBackend.CACHED,
            **kwargs,
        )

    @classmethod
    def full(cls, **kwargs) -> "PipelineConfig":
        """Create a full mode configuration with Vertex AI (Nano Banana)."""
        return cls(
            mode=PipelineMode.FULL,
            ocr_backend=OCRBackend.GOOGLE_VISION,
            image_generator=ImageGeneratorBackend.VERTEX_FLASH,
            **kwargs,
        )

    @classmethod
    def full_pro(cls, **kwargs) -> "PipelineConfig":
        """Create a full mode configuration with Vertex AI Pro (Nano Banana Pro)."""
        return cls(
            mode=PipelineMode.FULL,
            ocr_backend=OCRBackend.GOOGLE_VISION,
            image_generator=ImageGeneratorBackend.VERTEX_PRO,
            **kwargs,
        )
