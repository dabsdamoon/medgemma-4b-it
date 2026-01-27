"""
Figure captioning and analysis using MedGemma.

Wraps MedGemma inference for figure understanding tasks via RunPod API.
Supports both single-image and batch processing for efficiency.
"""

import io
import json
import logging
import os
import tempfile
from typing import Optional, List

from PIL import Image

from .schema import FigureAnalysis, AlignmentResult, FigureType, AnalysisConstraints

logger = logging.getLogger(__name__)


class FigureCaptioner:
    """
    Generate captions and structured analysis for medical figures using MedGemma.

    This class wraps the MedGemma RunPod API for figure-specific tasks:
    - Caption generation
    - Structured JSON analysis
    - Alignment evaluation

    Requires RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID environment variables.
    """

    def __init__(
        self,
        max_tokens: int = 1024,
        use_async: bool = False,
        **kwargs,  # Accept but ignore use_quantization for backwards compat
    ):
        """
        Initialize the figure captioner.

        Args:
            max_tokens: Maximum tokens for generation
            use_async: Use async RunPod API (better for batch processing)
        """
        self.max_tokens = max_tokens
        self.use_async = use_async
        self._client = None

    def _ensure_client(self):
        """Ensure the RunPod client is initialized (lazy loading)."""
        if self._client is None:
            # Import the client from the project root
            import sys
            from pathlib import Path

            # Add project root to path if needed
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            from client import MedGemmaClient

            self._client = MedGemmaClient()

    def _save_image_temp(self, image: Image.Image) -> str:
        """Save PIL Image to a temporary file and return the path."""
        # Create temp file that won't be auto-deleted
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        image.save(path, "PNG")
        return path

    def _generate(
        self,
        image: Image.Image,
        mode: str,
        context: str = "",
    ) -> str:
        """
        Generate response using MedGemma via RunPod.

        Args:
            image: Image to analyze
            mode: Prompt mode (figure_caption, figure_analysis, alignment_eval)
            context: Additional context text

        Returns:
            Generated text response
        """
        self._ensure_client()

        # Save image to temp file (client expects file path)
        image_path = self._save_image_temp(image)

        try:
            # Get the prompt for this mode
            from medgemma.prompts import get_prompt

            system_prompt = get_prompt(mode)

            if self.use_async:
                result = self._client.process_async(
                    text=context,
                    image_path=image_path,
                    max_tokens=self.max_tokens,
                    mode="general",  # Use general mode, custom prompt via system_prompt
                    system_prompt=system_prompt,
                )
            else:
                result = self._client.process_sync(
                    text=context,
                    image_path=image_path,
                    max_tokens=self.max_tokens,
                    mode="general",
                    system_prompt=system_prompt,
                )
            return result
        finally:
            # Clean up temp file
            if os.path.exists(image_path):
                os.remove(image_path)

    def generate_caption(
        self,
        image: Image.Image,
        context: str = "",
    ) -> str:
        """
        Generate a caption for a medical figure.

        Args:
            image: Medical figure image
            context: Surrounding text context

        Returns:
            Generated caption string
        """
        prompt_context = context
        if context:
            prompt_context = f"Context from surrounding text: {context}\n\nGenerate a caption for this medical figure."
        else:
            prompt_context = "Generate a caption for this medical figure."

        return self._generate(image, mode="figure_caption", context=prompt_context)

    def analyze_figure(
        self,
        image: Image.Image,
        context: str = "",
    ) -> FigureAnalysis:
        """
        Generate structured analysis of a medical figure.

        Args:
            image: Medical figure image
            context: Surrounding text context

        Returns:
            FigureAnalysis object with structured data
        """
        prompt_context = "Analyze this medical figure."
        if context:
            prompt_context = f"Context: {context}\n\n{prompt_context}"

        response = self._generate(image, mode="figure_analysis", context=prompt_context)

        # Parse JSON response
        analysis = self._parse_analysis_json(response)
        analysis.context = context

        # Also generate a caption
        caption = self.generate_caption(image, context)
        analysis.caption = caption

        return analysis

    def analyze_figures_batch(
        self,
        images: List[Image.Image],
        contexts: List[str] = None,
    ) -> List[FigureAnalysis]:
        """
        Analyze multiple figures in a single batch API call.

        This is more efficient than calling analyze_figure() multiple times
        because it reduces API call overhead (1 call instead of N).

        Args:
            images: List of medical figure images
            contexts: List of surrounding text contexts (one per image)

        Returns:
            List of FigureAnalysis objects
        """
        if not images:
            return []

        self._ensure_client()

        # Default contexts to empty strings
        if contexts is None:
            contexts = [""] * len(images)
        elif len(contexts) < len(images):
            contexts = contexts + [""] * (len(images) - len(contexts))

        # Save all images to temp files
        image_paths = []
        for image in images:
            path = self._save_image_temp(image)
            image_paths.append(path)

        try:
            # Get the prompt for figure analysis
            from medgemma.prompts import get_prompt
            system_prompt = get_prompt("figure_analysis")

            # Build context texts for each image
            texts = []
            for ctx in contexts:
                if ctx:
                    texts.append(f"Context: {ctx}\n\nAnalyze this medical figure.")
                else:
                    texts.append("Analyze this medical figure.")

            logger.info(f"Batch processing {len(images)} figures...")

            # Call batch API
            results = self._client.process_batch_async(
                image_paths=image_paths,
                texts=texts,
                max_tokens=self.max_tokens,
                mode="figure_analysis",
                system_prompt=system_prompt,
            )

            # Parse results into FigureAnalysis objects
            analyses = []
            for idx, (response, ctx) in enumerate(zip(results, contexts)):
                if response is None:
                    logger.warning(f"Batch item {idx} returned None")
                    analysis = FigureAnalysis(
                        figure_type=FigureType.OTHER,
                        anatomical_region="unknown",
                        teaching_point="Failed to analyze figure",
                    )
                else:
                    analysis = self._parse_analysis_json(response)
                    analysis.context = ctx

                analyses.append(analysis)

            logger.info(f"Batch processing complete: {len(analyses)} results")
            return analyses

        finally:
            # Clean up temp files
            for path in image_paths:
                if os.path.exists(path):
                    os.remove(path)

    def generate_captions_batch(
        self,
        images: List[Image.Image],
        contexts: List[str] = None,
    ) -> List[str]:
        """
        Generate captions for multiple figures in a single batch API call.

        Args:
            images: List of medical figure images
            contexts: List of surrounding text contexts (one per image)

        Returns:
            List of caption strings
        """
        if not images:
            return []

        self._ensure_client()

        # Default contexts to empty strings
        if contexts is None:
            contexts = [""] * len(images)
        elif len(contexts) < len(images):
            contexts = contexts + [""] * (len(images) - len(contexts))

        # Save all images to temp files
        image_paths = []
        for image in images:
            path = self._save_image_temp(image)
            image_paths.append(path)

        try:
            # Get the prompt for figure caption
            from medgemma.prompts import get_prompt
            system_prompt = get_prompt("figure_caption")

            # Build context texts for each image
            texts = []
            for ctx in contexts:
                if ctx:
                    texts.append(f"Context from surrounding text: {ctx}\n\nGenerate a caption for this medical figure.")
                else:
                    texts.append("Generate a caption for this medical figure.")

            logger.info(f"Batch captioning {len(images)} figures...")

            # Call batch API
            results = self._client.process_batch_async(
                image_paths=image_paths,
                texts=texts,
                max_tokens=self.max_tokens,
                mode="figure_caption",
                system_prompt=system_prompt,
            )

            logger.info(f"Batch captioning complete: {len(results)} results")
            return [r if r else "Failed to generate caption" for r in results]

        finally:
            # Clean up temp files
            for path in image_paths:
                if os.path.exists(path):
                    os.remove(path)

    def _parse_analysis_json(self, response: str) -> FigureAnalysis:
        """Parse JSON response into FigureAnalysis object."""
        try:
            # Try to extract JSON from response
            json_str = response.strip()

            # Handle cases where JSON is wrapped in markdown code blocks
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            data = json.loads(json_str)

            # Map figure_type string to enum
            figure_type_str = data.get("figure_type", "other").lower()
            try:
                figure_type = FigureType(figure_type_str)
            except ValueError:
                figure_type = FigureType.OTHER

            # Parse constraints
            constraints_data = data.get("constraints", {})
            constraints = AnalysisConstraints(
                anatomical=constraints_data.get("anatomical", []),
                style=constraints_data.get("style", []),
                labels=constraints_data.get("labels", []),
            )

            return FigureAnalysis(
                figure_type=figure_type,
                anatomical_region=data.get("anatomical_region", "unknown"),
                entities=data.get("entities", []),
                relationships=data.get("relationships", []),
                findings=data.get("findings", []),
                teaching_point=data.get("teaching_point", ""),
                constraints=constraints,
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse analysis JSON: {e}")
            # Return a default analysis
            return FigureAnalysis(
                figure_type=FigureType.OTHER,
                anatomical_region="unknown",
                teaching_point=response[:500] if response else "",
            )

    def evaluate_alignment(
        self,
        original: Image.Image,
        generated: Image.Image,
        pass_threshold: float = 0.80,
        flag_threshold: float = 0.65,
    ) -> AlignmentResult:
        """
        Evaluate alignment between original and generated figures.

        Args:
            original: Original medical figure
            generated: Generated reproduction
            pass_threshold: Score threshold for pass status
            flag_threshold: Score threshold for flag status

        Returns:
            AlignmentResult with scores and status
        """
        # Create a combined image for comparison
        combined = self._create_comparison_image(original, generated)

        context = (
            "Compare these two medical figures. The left image is the ORIGINAL "
            "and the right image is the GENERATED reproduction. "
            "Evaluate how well the generated image preserves the medical content."
        )

        response = self._generate(combined, mode="alignment_eval", context=context)

        return self._parse_alignment_json(response, pass_threshold, flag_threshold)

    def _create_comparison_image(
        self,
        original: Image.Image,
        generated: Image.Image,
    ) -> Image.Image:
        """Create a side-by-side comparison image."""
        # Resize to same height
        target_height = 512
        orig_ratio = original.width / original.height
        gen_ratio = generated.width / generated.height

        orig_resized = original.resize(
            (int(target_height * orig_ratio), target_height)
        )
        gen_resized = generated.resize((int(target_height * gen_ratio), target_height))

        # Create combined image
        total_width = orig_resized.width + gen_resized.width + 20  # 20px gap
        combined = Image.new("RGB", (total_width, target_height), color=(255, 255, 255))
        combined.paste(orig_resized, (0, 0))
        combined.paste(gen_resized, (orig_resized.width + 20, 0))

        return combined

    def _parse_alignment_json(
        self,
        response: str,
        pass_threshold: float,
        flag_threshold: float,
    ) -> AlignmentResult:
        """Parse JSON response into AlignmentResult object."""
        try:
            json_str = response.strip()

            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            data = json.loads(json_str)

            score = float(data.get("alignment_score", 0.5))

            return AlignmentResult.from_score(
                score=score,
                pass_threshold=pass_threshold,
                flag_threshold=flag_threshold,
                anatomical_accuracy=float(data.get("anatomical_accuracy", score)),
                content_preservation=float(data.get("content_preservation", score)),
                educational_value=float(data.get("educational_value", score)),
                reasoning=data.get("reasoning", ""),
                missing_elements=data.get("missing_elements", []),
                added_elements=data.get("added_elements", []),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse alignment JSON: {e}")
            # Return a default flagged result
            return AlignmentResult.from_score(
                score=0.7,
                pass_threshold=pass_threshold,
                flag_threshold=flag_threshold,
                anatomical_accuracy=0.7,
                content_preservation=0.7,
                educational_value=0.7,
                reasoning=f"Could not parse evaluation response: {response[:200]}",
            )
