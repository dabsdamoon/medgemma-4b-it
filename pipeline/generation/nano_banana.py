"""
Nano Banana Pro API client for image generation.

Provides async image generation with retry logic and variant support.
"""

import asyncio
import base64
import io
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

from .prompt_adapter import ImageBrief

logger = logging.getLogger(__name__)

# Check for httpx availability
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


@dataclass
class GeneratedImage:
    """A generated image from Nano Banana Pro."""

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


class NanoBananaClient:
    """
    Async client for Nano Banana Pro image generation API.

    Supports multiple variants per request and automatic retry with
    exponential backoff.
    """

    DEFAULT_BASE_URL = "https://api.nanobanana.com/v1"
    DEFAULT_TIMEOUT = 60.0
    MAX_RETRIES = 3
    RETRY_BACKOFF = 2.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """
        Initialize the Nano Banana Pro client.

        Args:
            api_key: API key (defaults to NANO_BANANA_API_KEY env var)
            base_url: API base URL
            timeout: Request timeout in seconds
        """
        if not HTTPX_AVAILABLE:
            raise ImportError(
                "httpx is required for Nano Banana Pro client. "
                "Install with: pip install httpx"
            )

        self.api_key = api_key or os.environ.get("NANO_BANANA_API_KEY")
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.timeout = timeout

        if not self.api_key:
            logger.warning(
                "No API key provided. Set NANO_BANANA_API_KEY or pass api_key parameter."
            )

    def _get_headers(self) -> dict:
        """Get request headers with authentication."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate_async(
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
        Generate images asynchronously using img2img conversion.

        Args:
            brief: Image generation brief with prompt and constraints
            original_image: Source image to convert/modernize (required for img2img)
            num_variants: Number of variants to generate (1-4)
            seed: Random seed for reproducibility
            width: Image width
            height: Image height
            strength: How much to transform (0.0 = identical, 1.0 = ignore original)
                      Lower values preserve more of original structure.
                      Recommended: 0.5-0.75 for medical figure modernization.

        Returns:
            List of GeneratedImage objects
        """
        if not self.api_key:
            raise ValueError("API key is required for image generation")

        if original_image is None:
            raise ValueError(
                "original_image is required for img2img conversion. "
                "The goal is to modernize existing figures, not create new ones."
            )

        num_variants = min(max(1, num_variants), 4)

        # Adjust dimensions based on aspect ratio
        width, height = self._apply_aspect_ratio(brief.aspect_ratio, width, height)

        # Resize original image to target dimensions
        resized_original = original_image.resize((width, height), Image.Resampling.LANCZOS)

        payload = {
            # img2img mode - include the source image
            "image": encode_image_to_base64(resized_original),
            "prompt": brief.prompt,
            "negative_prompt": brief.negative_prompt,
            "width": width,
            "height": height,
            "num_images": num_variants,
            "guidance_scale": 7.5,
            "strength": strength,  # Controls how much to deviate from original
        }

        if seed is not None:
            payload["seed"] = seed

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await self._request_with_retry(
                client,
                "POST",
                f"{self.base_url}/img2img",  # img2img endpoint
                json=payload,
            )

        return self._parse_response(response, brief, seed or 0)

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
        Generate images synchronously using img2img conversion.

        Args:
            brief: Image generation brief with prompt and constraints
            original_image: Source image to convert/modernize (required)
            num_variants: Number of variants to generate (1-4)
            seed: Random seed for reproducibility
            width: Image width
            height: Image height
            strength: How much to transform (0.0 = identical, 1.0 = ignore original)

        Returns:
            List of GeneratedImage objects
        """
        return asyncio.run(
            self.generate_async(
                brief, original_image, num_variants, seed, width, height, strength
            )
        )

    async def _request_with_retry(
        self,
        client: "httpx.AsyncClient",
        method: str,
        url: str,
        **kwargs,
    ) -> dict:
        """Make a request with exponential backoff retry."""
        last_error = None

        for attempt in range(self.MAX_RETRIES):
            try:
                response = await client.request(
                    method,
                    url,
                    headers=self._get_headers(),
                    **kwargs,
                )
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 429:  # Rate limited
                    wait_time = self.RETRY_BACKOFF ** (attempt + 1)
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                elif e.response.status_code >= 500:  # Server error
                    wait_time = self.RETRY_BACKOFF ** attempt
                    logger.warning(f"Server error, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise

            except httpx.RequestError as e:
                last_error = e
                wait_time = self.RETRY_BACKOFF ** attempt
                logger.warning(f"Request error: {e}, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

        raise last_error or Exception("Max retries exceeded")

    def _parse_response(
        self,
        response: dict,
        brief: ImageBrief,
        base_seed: int,
    ) -> list[GeneratedImage]:
        """Parse API response into GeneratedImage objects."""
        images = []
        image_data_list = response.get("images", [])

        for idx, img_data in enumerate(image_data_list):
            # Decode base64 image
            if isinstance(img_data, str):
                image_bytes = base64.b64decode(img_data)
            elif isinstance(img_data, dict):
                image_bytes = base64.b64decode(img_data.get("data", ""))
            else:
                continue

            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            images.append(
                GeneratedImage(
                    image=pil_image,
                    prompt=brief.prompt,
                    seed=base_seed + idx,
                    variant_index=idx,
                    generation_time_ms=response.get("generation_time_ms", 0),
                    metadata={
                        **brief.metadata,
                        "api_response": {
                            k: v
                            for k, v in response.items()
                            if k not in ["images"]
                        },
                    },
                )
            )

        return images

    def _apply_aspect_ratio(
        self,
        aspect_ratio: str,
        width: int,
        height: int,
    ) -> tuple[int, int]:
        """Apply aspect ratio to dimensions."""
        try:
            w_ratio, h_ratio = map(int, aspect_ratio.split(":"))
            # Maintain total pixel count approximately
            total_pixels = width * height
            ratio = w_ratio / h_ratio
            new_height = int((total_pixels / ratio) ** 0.5)
            new_width = int(new_height * ratio)
            # Round to nearest 64 (common requirement)
            new_width = (new_width // 64) * 64
            new_height = (new_height // 64) * 64
            return max(64, new_width), max(64, new_height)
        except (ValueError, ZeroDivisionError):
            return width, height


class CachedImageGenerator:
    """
    Cached image generator for reproducible mode.

    Returns pre-computed images instead of making API calls.
    """

    def __init__(self, cache_dir: str = "demo/cached_examples"):
        """
        Initialize the cached generator.

        Args:
            cache_dir: Directory containing cached images
        """
        from pathlib import Path

        self.cache_dir = Path(cache_dir)

    def generate(
        self,
        brief: ImageBrief,
        figure_id: str,
        num_variants: int = 1,
        original_image: Optional[Image.Image] = None,  # Unused, for API consistency
    ) -> list[GeneratedImage]:
        """
        Get cached images for a figure.

        Args:
            brief: Image brief (used for metadata)
            figure_id: Figure identifier for cache lookup
            num_variants: Number of variants requested
            original_image: Ignored (cached mode uses pre-computed images)

        Returns:
            List of cached GeneratedImage objects
        """
        images = []

        for idx in range(num_variants):
            # Look for cached image
            cache_patterns = [
                self.cache_dir / figure_id / f"generated_{idx}.png",
                self.cache_dir / f"{figure_id}_generated_{idx}.png",
                self.cache_dir / f"{figure_id}_new.png",
            ]

            for cache_path in cache_patterns:
                if cache_path.exists():
                    pil_image = Image.open(cache_path).convert("RGB")
                    images.append(
                        GeneratedImage(
                            image=pil_image,
                            prompt=brief.prompt,
                            seed=idx,
                            variant_index=idx,
                            metadata={
                                **brief.metadata,
                                "cached": True,
                                "cache_path": str(cache_path),
                            },
                        )
                    )
                    break

        if not images:
            logger.warning(f"No cached images found for {figure_id}")
            # Return a placeholder
            placeholder = Image.new("RGB", (512, 512), color=(200, 200, 200))
            images.append(
                GeneratedImage(
                    image=placeholder,
                    prompt=brief.prompt,
                    seed=0,
                    variant_index=0,
                    metadata={**brief.metadata, "placeholder": True},
                )
            )

        return images
