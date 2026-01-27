"""
Vertex AI Image Generator using Google GenAI SDK.

Supports both Nano Banana (Gemini 2.5 Flash Image) and Nano Banana Pro (Gemini 3 Pro).
Uses img2img mode for medical figure modernization.
"""

import base64
import io
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from PIL import Image

from .prompt_adapter import ImageBrief

logger = logging.getLogger(__name__)

# Check for google-genai availability
try:
    from google import genai
    from google.genai.types import GenerateContentConfig, Modality, Part, Content

    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class ImageModel(str, Enum):
    """Available image generation models."""

    # Nano Banana (Gemini 2.5 Flash Image) - cheaper, faster
    # Location: us-central1
    # See: https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash-image
    FLASH = "gemini-2.5-flash-image"

    # Nano Banana Pro (Gemini 3 Pro Image) - higher quality
    # Location: global (required)
    PRO = "gemini-3-pro-image-preview"


# Model-specific location requirements
MODEL_LOCATIONS = {
    ImageModel.FLASH: "us-central1",
    ImageModel.PRO: "global",
}


@dataclass
class GeneratedImage:
    """A generated image from Vertex AI."""

    image: Image.Image
    prompt: str
    seed: int
    variant_index: int
    generation_time_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


def encode_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Encode PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Convert PIL Image to bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return buffer.read()


class VertexImageGenerator:
    """
    Image generator using Google GenAI SDK with Vertex AI.

    Supports img2img conversion for medical figure modernization.
    Default model is Nano Banana (Gemini 2.5 Flash Image) for cost efficiency.
    """

    def __init__(
        self,
        model: ImageModel = ImageModel.FLASH,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
    ):
        """
        Initialize the Vertex AI image generator.

        Args:
            model: Image generation model (FLASH or PRO)
            project_id: GCP project ID (defaults to GOOGLE_CLOUD_PROJECT env var)
            location: GCP location (auto-detected based on model if not specified)
                      - FLASH: us-central1
                      - PRO: global
        """
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-genai is required for Vertex AI image generation. "
                "Install with: pip install google-genai"
            )

        self.model = model
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        # Auto-select location based on model if not specified
        self.location = location or MODEL_LOCATIONS.get(model, "us-central1")
        self._client: Optional[genai.Client] = None
        self._initialized = False

    def _initialize(self):
        """Initialize the GenAI client for Vertex AI."""
        if self._initialized:
            return

        if not self.project_id:
            raise ValueError(
                "GCP project ID is required. Set GOOGLE_CLOUD_PROJECT env var "
                "or pass project_id parameter."
            )

        # Set environment variables for Vertex AI mode
        os.environ["GOOGLE_CLOUD_PROJECT"] = self.project_id
        os.environ["GOOGLE_CLOUD_LOCATION"] = self.location
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

        self._client = genai.Client()
        self._initialized = True
        logger.info(f"Initialized Vertex AI client with model: {self.model.value}")

    def generate(
        self,
        brief: ImageBrief,
        original_image: Optional[Image.Image] = None,
        num_variants: int = 1,
        seed: Optional[int] = None,
        width: int = 512,
        height: int = 512,
        strength: float = 0.75,
    ) -> list[GeneratedImage]:
        """
        Generate images using img2img conversion.

        Args:
            brief: Image generation brief with prompt and constraints
            original_image: Source image to convert/modernize (required for img2img)
            num_variants: Number of variants to generate (1-4)
            seed: Random seed (not directly supported, included for API consistency)
            width: Target image width
            height: Target image height
            strength: Transformation strength (used in prompt guidance)

        Returns:
            List of GeneratedImage objects
        """
        self._initialize()

        if original_image is None:
            raise ValueError(
                "original_image is required for img2img conversion. "
                "The goal is to modernize existing figures, not create new ones."
            )

        num_variants = min(max(1, num_variants), 4)

        # Resize original image to target dimensions
        resized_original = original_image.resize(
            (width, height), Image.Resampling.LANCZOS
        )

        # Build the prompt for img2img
        # Include strength guidance in prompt
        strength_guidance = self._get_strength_guidance(strength)

        full_prompt = f"""Transform this medical figure into a modern, high-quality illustration.

{strength_guidance}

Style requirements:
{brief.prompt}

Constraints:
- Preserve all anatomical accuracy from the original
- Maintain the same layout and structure
- Keep all labels and annotations readable
- Use clean, professional medical illustration style

Avoid:
{brief.negative_prompt}

Generate {num_variants} variant(s)."""

        logger.info(f"Generating {num_variants} variant(s) with model {self.model.value}")

        # Prepare the content with image
        image_bytes = image_to_bytes(resized_original)

        # Create multimodal content (image + text)
        response = self._client.models.generate_content(
            model=self.model.value,
            contents=[
                Part.from_bytes(data=image_bytes, mime_type="image/png"),
                full_prompt,
            ],
            config=GenerateContentConfig(
                response_modalities=[Modality.IMAGE, Modality.TEXT],
            ),
        )

        # Parse response into GeneratedImage objects
        return self._parse_response(response, brief, seed or 0, num_variants)

    def _get_strength_guidance(self, strength: float) -> str:
        """Convert strength parameter to prompt guidance."""
        if strength < 0.3:
            return "Make minimal changes - only enhance clarity and colors while preserving the original style exactly."
        elif strength < 0.5:
            return "Moderately modernize - improve quality while keeping the original composition and style recognizable."
        elif strength < 0.7:
            return "Significantly modernize - create a cleaner, more professional version while preserving key elements."
        else:
            return "Fully modernize - create a contemporary medical illustration style while preserving anatomical accuracy."

    def _parse_response(
        self,
        response,
        brief: ImageBrief,
        base_seed: int,
        expected_count: int,
    ) -> list[GeneratedImage]:
        """Parse API response into GeneratedImage objects."""
        images = []

        if not response.candidates:
            logger.warning("No candidates in response")
            return images

        for candidate in response.candidates:
            if not candidate.content or not candidate.content.parts:
                continue

            for part in candidate.content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    try:
                        image_bytes = part.inline_data.data
                        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                        images.append(
                            GeneratedImage(
                                image=pil_image,
                                prompt=brief.prompt,
                                seed=base_seed + len(images),
                                variant_index=len(images),
                                metadata={
                                    **brief.metadata,
                                    "model": self.model.value,
                                    "source": "vertex_ai",
                                },
                            )
                        )

                        if len(images) >= expected_count:
                            break
                    except Exception as e:
                        logger.warning(f"Failed to parse image: {e}")

        if not images:
            logger.warning("No images found in response, creating placeholder")
            placeholder = Image.new("RGB", (512, 512), color=(200, 200, 200))
            images.append(
                GeneratedImage(
                    image=placeholder,
                    prompt=brief.prompt,
                    seed=base_seed,
                    variant_index=0,
                    metadata={**brief.metadata, "placeholder": True},
                )
            )

        return images

    def generate_text_to_image(
        self,
        brief: ImageBrief,
        num_variants: int = 1,
    ) -> list[GeneratedImage]:
        """
        Generate images from text only (no source image).

        Use this only when there's no original image to modernize.

        Args:
            brief: Image generation brief with prompt
            num_variants: Number of variants to generate

        Returns:
            List of GeneratedImage objects
        """
        self._initialize()

        full_prompt = f"""Create a medical illustration:

{brief.prompt}

Style: Clean, professional medical diagram suitable for educational textbooks.

Avoid:
{brief.negative_prompt}"""

        logger.info(f"Text-to-image generation with model {self.model.value}")

        response = self._client.models.generate_content(
            model=self.model.value,
            contents=full_prompt,
            config=GenerateContentConfig(
                response_modalities=[Modality.IMAGE],
            ),
        )

        return self._parse_response(response, brief, 0, num_variants)


# Convenience function
def create_generator(
    use_pro: bool = False,
    project_id: Optional[str] = None,
) -> VertexImageGenerator:
    """
    Create an image generator instance.

    Args:
        use_pro: If True, use Nano Banana Pro (more expensive, higher quality)
                 If False (default), use Nano Banana (cheaper, faster)
        project_id: GCP project ID

    Returns:
        VertexImageGenerator instance
    """
    model = ImageModel.PRO if use_pro else ImageModel.FLASH
    return VertexImageGenerator(model=model, project_id=project_id)
