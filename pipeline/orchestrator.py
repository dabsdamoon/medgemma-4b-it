"""
Main pipeline orchestrator.

Coordinates all stages of the document modernization pipeline.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm

from .config import PipelineConfig, PipelineMode, OCRBackend, ImageGeneratorBackend
from .ingest import PDFExtractor, FigureDetector, DetectedFigure, FigureFilter
from .ocr import TesseractOCR, EasyOCR, GoogleVisionOCR, OCRBase
from .understanding import FigureCaptioner, FigureAnalysis
from .generation import (
    PromptAdapter,
    NanoBananaClient,
    CachedImageGenerator,
    VertexImageGenerator,
    ImageModel,
    SafetyFilterError,
    get_sanitizer,
)
from .evaluation import AlignmentScorer
from .output import OutputBundler, OutputBundle, DocumentMetadata, FigureMetadata
from .utils import timed_operation

logger = logging.getLogger(__name__)


class DocumentModernizer:
    """
    Main orchestrator for the Legacy Medical Document Modernizer pipeline.

    This class coordinates all stages:
    1. PDF ingestion and figure extraction
    2. OCR text extraction
    3. MedGemma figure analysis
    4. Image generation (or cached retrieval)
    5. Alignment evaluation
    6. Output bundling
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the document modernizer.

        Args:
            config: Pipeline configuration (defaults to reproducible mode)
        """
        self.config = config or PipelineConfig.reproducible()

        # Initialize components (lazy loaded)
        self._pdf_extractor: Optional[PDFExtractor] = None
        self._figure_detector: Optional[FigureDetector] = None
        self._figure_filter: Optional[FigureFilter] = None
        self._ocr: Optional[OCRBase] = None
        self._captioner: Optional[FigureCaptioner] = None
        self._prompt_adapter: Optional[PromptAdapter] = None
        self._image_generator = None
        self._alignment_scorer: Optional[AlignmentScorer] = None

    @property
    def pdf_extractor(self) -> PDFExtractor:
        """Get or create PDF extractor."""
        if self._pdf_extractor is None:
            self._pdf_extractor = PDFExtractor()
        return self._pdf_extractor

    @property
    def figure_detector(self) -> FigureDetector:
        """Get or create figure detector."""
        if self._figure_detector is None:
            self._figure_detector = FigureDetector(
                min_size=self.config.min_figure_size,
                max_figures=self.config.max_figures_per_page,
            )
        return self._figure_detector

    @property
    def figure_filter(self) -> FigureFilter:
        """Get or create figure filter."""
        if self._figure_filter is None:
            self._figure_filter = FigureFilter(
                max_aspect_ratio=self.config.filter_max_aspect_ratio,
            )
        return self._figure_filter

    @property
    def ocr(self) -> OCRBase:
        """Get or create OCR engine."""
        if self._ocr is None:
            if self.config.ocr_backend == OCRBackend.TESSERACT:
                self._ocr = TesseractOCR(language=self.config.ocr_language)
            elif self.config.ocr_backend == OCRBackend.EASYOCR:
                self._ocr = EasyOCR(language=self.config.ocr_language)
            elif self.config.ocr_backend == OCRBackend.GOOGLE_VISION:
                self._ocr = GoogleVisionOCR(language=self.config.ocr_language)
            else:
                raise ValueError(f"Unknown OCR backend: {self.config.ocr_backend}")
        return self._ocr

    @property
    def captioner(self) -> FigureCaptioner:
        """Get or create figure captioner."""
        if self._captioner is None:
            self._captioner = FigureCaptioner(
                use_quantization=self.config.use_quantization,
                max_tokens=self.config.max_tokens,
            )
        return self._captioner

    @property
    def prompt_adapter(self) -> PromptAdapter:
        """Get or create prompt adapter."""
        if self._prompt_adapter is None:
            self._prompt_adapter = PromptAdapter()
        return self._prompt_adapter

    @property
    def image_generator(self):
        """Get or create image generator."""
        if self._image_generator is None:
            backend = self.config.image_generator

            if backend == ImageGeneratorBackend.VERTEX_FLASH:
                # Nano Banana (Gemini 2.5 Flash) - cheaper, default
                self._image_generator = VertexImageGenerator(
                    model=ImageModel.FLASH,
                    project_id=self.config.gcp_project_id,
                    location=self.config.gcp_location,
                )
            elif backend == ImageGeneratorBackend.VERTEX_PRO:
                # Nano Banana Pro (Gemini 3 Pro) - higher quality
                self._image_generator = VertexImageGenerator(
                    model=ImageModel.PRO,
                    project_id=self.config.gcp_project_id,
                    location=self.config.gcp_location,
                )
            elif backend == ImageGeneratorBackend.NANO_BANANA:
                # Direct Nano Banana API (legacy)
                self._image_generator = NanoBananaClient(
                    api_key=self.config.nano_banana_api_key,
                    base_url=self.config.nano_banana_base_url,
                )
            else:
                # Cached images for testing
                self._image_generator = CachedImageGenerator(
                    cache_dir=str(self.config.cache_dir)
                )
        return self._image_generator

    @property
    def alignment_scorer(self) -> AlignmentScorer:
        """Get or create alignment scorer."""
        if self._alignment_scorer is None:
            self._alignment_scorer = AlignmentScorer(
                pass_threshold=self.config.alignment_thresholds.pass_threshold,
                flag_threshold=self.config.alignment_thresholds.flag_threshold,
                captioner=self.captioner,
            )
        return self._alignment_scorer

    def process_image(
        self,
        image_path: Path | str,
        output_dir: Optional[Path | str] = None,
    ) -> OutputBundle:
        """
        Process a single image directly (no PDF extraction).

        Args:
            image_path: Path to image file (png, jpg, etc.)
            output_dir: Output directory

        Returns:
            OutputBundle with results
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir) if output_dir else self.config.output_dir

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info(f"Processing image: {image_path}")

        # Load image
        from PIL import Image
        from medgemma import hash_image

        image = Image.open(image_path).convert("RGB")

        # Compute source hash
        source_hash = hash_image(image)

        # Initialize output bundler
        bundler = OutputBundler(output_dir)

        # Initialize metadata
        metadata = DocumentMetadata(
            source_filename=image_path.name,
            source_hash=source_hash,
            page_count=1,
            pipeline_mode=self.config.mode.value,
            ocr_backend="none",
            ocr_language=self.config.ocr_language,
            config={
                "input_type": "image",
                "generation_variants": self.config.generation_variants,
            },
        )
        metadata.figures_detected = 1

        bundle = bundler.initialize(metadata)

        # Create a synthetic DetectedFigure
        figure = DetectedFigure(
            image=image,
            page_num=0,
            figure_id="img_0",
            bbox=(0, 0, image.width, image.height),
            nearby_text="",
        )

        # Process based on mode
        if self.config.extract_only:
            logger.info("Extract-only mode: saving image")
            self._extract_figure_only(figure, bundler)
        elif self.config.analyze_only:
            logger.info("Analyze-only mode: running MedGemma analysis")
            analysis = self._analyze_figure_only(figure, bundler)
            if analysis:
                self._save_analysis_results([analysis], output_dir)
        else:
            logger.info("Full mode: analysis + generation")
            self._process_figure(figure, bundler)

        return bundler.finalize()

    def process_images(
        self,
        image_paths: list[Path | str],
        output_dir: Optional[Path | str] = None,
    ) -> OutputBundle:
        """
        Process multiple images directly (no PDF extraction).

        Args:
            image_paths: List of paths to image files
            output_dir: Output directory

        Returns:
            OutputBundle with results for all images
        """
        from PIL import Image
        from medgemma import hash_image

        output_dir = Path(output_dir) if output_dir else self.config.output_dir

        logger.info(f"Processing {len(image_paths)} images")

        # Initialize output bundler
        bundler = OutputBundler(output_dir)

        # Compute combined hash from first image
        first_image = Image.open(image_paths[0]).convert("RGB")
        source_hash = hash_image(first_image)

        # Initialize metadata
        metadata = DocumentMetadata(
            source_filename=f"{len(image_paths)}_images",
            source_hash=source_hash,
            page_count=len(image_paths),
            pipeline_mode=self.config.mode.value,
            ocr_backend="none",
            ocr_language=self.config.ocr_language,
            config={
                "input_type": "images",
                "generation_variants": self.config.generation_variants,
            },
        )
        metadata.figures_detected = len(image_paths)

        bundle = bundler.initialize(metadata)

        # Collect analysis results
        analysis_results = []

        # Build figures list
        figures = []
        for idx, img_path in enumerate(image_paths):
            img_path = Path(img_path)
            image = Image.open(img_path).convert("RGB")

            figure = DetectedFigure(
                image=image,
                page_num=0,
                figure_id=f"img_{idx}",
                bbox=(0, 0, image.width, image.height),
                nearby_text=img_path.stem,  # Use filename as context
            )
            figures.append(figure)

        # Use batch processing for multiple images in analyze-only mode
        if self.config.analyze_only and len(figures) > 1:
            try:
                analysis_results = self._analyze_figures_batch(figures, bundler)
            except Exception as e:
                logger.error(f"Batch analysis failed: {e}")
                metadata.add_error(f"Batch analysis: {str(e)}")
        else:
            # Process each image sequentially
            for figure in tqdm(
                figures,
                desc="Processing images",
                disable=not self.config.verbose,
            ):
                try:
                    if self.config.extract_only:
                        self._extract_figure_only(figure, bundler)
                    elif self.config.analyze_only:
                        analysis = self._analyze_figure_only(figure, bundler)
                        if analysis:
                            analysis_results.append(analysis)
                    else:
                        self._process_figure(figure, bundler)
                except Exception as e:
                    logger.error(f"Error processing {figure.figure_id}: {e}")
                    metadata.add_error(f"Image {figure.figure_id}: {str(e)}")

        # Save analysis results
        if self.config.analyze_only and analysis_results:
            self._save_analysis_results(analysis_results, output_dir)

        return bundler.finalize()

    def process(
        self,
        input_path: Path | str,
        output_dir: Optional[Path | str] = None,
    ) -> OutputBundle:
        """
        Process a document or image through the pipeline.

        Automatically detects input type (PDF vs image) and routes accordingly.

        Args:
            input_path: Path to input file (PDF or image)
            output_dir: Output directory (defaults to config)

        Returns:
            OutputBundle with all results
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir) if output_dir else self.config.output_dir

        # Check if input is an image (not PDF)
        image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}
        if input_path.suffix.lower() in image_extensions:
            return self.process_image(input_path, output_dir)

        logger.info(f"Processing {input_path}")

        # Compute source hash
        source_hash = self._compute_file_hash(input_path)

        # Initialize output bundler
        bundler = OutputBundler(output_dir)

        # Get page count
        page_count = self.pdf_extractor.get_page_count(input_path)

        # Initialize metadata
        metadata = DocumentMetadata(
            source_filename=input_path.name,
            source_hash=source_hash,
            page_count=page_count,
            pipeline_mode=self.config.mode.value,
            ocr_backend=self.config.ocr_backend.value,
            ocr_language=self.config.ocr_language,
            config={
                "generation_variants": self.config.generation_variants,
                "pass_threshold": self.config.alignment_thresholds.pass_threshold,
                "flag_threshold": self.config.alignment_thresholds.flag_threshold,
            },
        )

        bundle = bundler.initialize(metadata)

        # Process document
        all_text = []
        all_figures: list[DetectedFigure] = []

        # Extract pages and figures
        logger.info("Extracting pages and figures...")
        for page in tqdm(
            self.pdf_extractor.extract(input_path),
            total=page_count,
            desc="Extracting",
            disable=not self.config.verbose,
        ):
            # Collect page text
            all_text.append(page.text)

            # Detect figures from page render
            detected = self.figure_detector.detect(
                page.image, page.page_num, page.text
            )
            all_figures.extend(detected)

            # Also include embedded images
            for emb in page.embedded_images:
                if (
                    emb.image.width >= self.config.min_figure_size[0]
                    and emb.image.height >= self.config.min_figure_size[1]
                ):
                    embedded_fig = self.figure_detector.detect_from_embedded(
                        emb.image,
                        page.page_num,
                        emb.index,
                        emb.bbox,
                        page.text,
                        page.height,
                    )
                    all_figures.append(embedded_fig)

        metadata.figures_detected = len(all_figures)

        # Filter figures to remove non-medical content (headers, logos, etc.)
        if self.config.filter_figures and all_figures:
            logger.info(f"Filtering {len(all_figures)} detected figures...")
            filtered_figures = []
            skipped_count = 0

            for figure in all_figures:
                result = self.figure_filter.filter(figure.image)
                if result.is_medical:
                    filtered_figures.append(figure)
                else:
                    skipped_count += 1
                    logger.debug(
                        f"Skipped {figure.figure_id}: {result.reason.value}"
                    )

            logger.info(
                f"Kept {len(filtered_figures)} figures, skipped {skipped_count} "
                f"(non-medical: headers, logos, etc.)"
            )
            all_figures = filtered_figures

        # Save OCR text
        combined_text = "\n\n".join(all_text)
        bundler.save_ocr_text(combined_text)

        # Process each figure
        if self.config.extract_only:
            logger.info(f"Extracting {len(all_figures)} figures (extract-only mode)...")
            process_desc = "Extracting figures"
        elif self.config.analyze_only:
            logger.info(f"Analyzing {len(all_figures)} figures (analyze-only mode)...")
            process_desc = "Analyzing figures"
        else:
            logger.info(f"Processing {len(all_figures)} figures...")
            process_desc = "Processing figures"

        # Collect analysis results for analyze_only mode
        analysis_results = []

        if self.config.analyze_only and len(all_figures) > 1:
            # Use batch processing for multiple figures in analyze-only mode
            try:
                analysis_results = self._analyze_figures_batch(all_figures, bundler)
            except Exception as e:
                logger.error(f"Batch analysis failed: {e}")
                metadata.add_error(f"Batch analysis: {str(e)}")
        else:
            for figure in tqdm(
                all_figures,
                desc=process_desc,
                disable=not self.config.verbose,
            ):
                try:
                    if self.config.extract_only:
                        self._extract_figure_only(figure, bundler)
                    elif self.config.analyze_only:
                        analysis = self._analyze_figure_only(figure, bundler)
                        if analysis:
                            analysis_results.append(analysis)
                    else:
                        self._process_figure(figure, bundler)
                except Exception as e:
                    logger.error(f"Error processing figure {figure.figure_id}: {e}")
                    metadata.add_error(f"Figure {figure.figure_id}: {str(e)}")

        # Save analysis results JSON for analyze_only mode
        if self.config.analyze_only and analysis_results:
            self._save_analysis_results(analysis_results, output_dir)

        # Finalize and return
        return bundler.finalize()

    def _extract_figure_only(
        self,
        figure: DetectedFigure,
        bundler: OutputBundler,
    ):
        """Extract figure without AI analysis (extract-only mode)."""
        from medgemma import hash_image

        # Compute original hash
        original_hash = hash_image(figure.image)

        # Create minimal metadata (no AI analysis)
        fig_metadata = FigureMetadata(
            figure_id=figure.figure_id,
            page_num=figure.page_num,
            original_hash=original_hash,
            bbox=figure.bbox,
            nearby_text=figure.nearby_text[:500] if figure.nearby_text else "",
        )

        # Save only the original figure (no generated image)
        bundler.save_figure(
            figure.figure_id,
            figure.image,
            None,  # No generated image
            fig_metadata,
        )

    def _analyze_figure_only(
        self,
        figure: DetectedFigure,
        bundler: OutputBundler,
    ) -> Optional[dict]:
        """Analyze figure with MedGemma but skip image generation."""
        from medgemma import hash_image

        verbose = self.config.verbose

        # Compute original hash
        original_hash = hash_image(figure.image)

        # Analyze figure with MedGemma
        with timed_operation(f"MedGemma analysis [{figure.figure_id}]", verbose=verbose):
            analysis = self.captioner.analyze_figure(
                figure.image,
                context=figure.nearby_text,
            )

        # Convert to image generation prompt (for saving)
        brief = self.prompt_adapter.convert(analysis)

        # Create figure metadata
        fig_metadata = FigureMetadata(
            figure_id=figure.figure_id,
            page_num=figure.page_num,
            original_hash=original_hash,
            figure_type=analysis.figure_type.value,
            anatomical_region=analysis.anatomical_region,
            caption=analysis.caption or "",
            teaching_point=analysis.teaching_point,
            generation_prompt=brief.prompt,
            bbox=figure.bbox,
            nearby_text=figure.nearby_text[:500] if figure.nearby_text else "",
        )

        # Save the original figure (no generated image)
        bundler.save_figure(
            figure.figure_id,
            figure.image,
            None,  # No generated image in analyze-only mode
            fig_metadata,
        )

        # Return analysis dict for JSON export
        return {
            "figure_id": figure.figure_id,
            "page_num": figure.page_num,
            "original_hash": original_hash,
            "image_path": f"figures/figure_{figure.figure_id}_original.png",
            "analysis": {
                "figure_type": analysis.figure_type.value,
                "anatomical_region": analysis.anatomical_region,
                "entities": analysis.entities,
                "relationships": analysis.relationships,
                "findings": analysis.findings,
                "teaching_point": analysis.teaching_point,
                "caption": analysis.caption,
                "constraints": {
                    "anatomical": analysis.constraints.anatomical,
                    "style": analysis.constraints.style,
                    "labels": analysis.constraints.labels,
                },
            },
            "generation_brief": {
                "prompt": brief.prompt,
                "negative_prompt": brief.negative_prompt,
                "style": brief.style,
                "aspect_ratio": brief.aspect_ratio,
                "constraints": brief.constraints,
            },
            "nearby_text": figure.nearby_text[:500] if figure.nearby_text else "",
        }

    def _analyze_figures_batch(
        self,
        figures: list[DetectedFigure],
        bundler: OutputBundler,
    ) -> list[dict]:
        """
        Analyze multiple figures in a single batch API call.

        This is more efficient than calling _analyze_figure_only() multiple times
        because it reduces API call overhead (1 call instead of N).

        Args:
            figures: List of detected figures
            bundler: Output bundler for saving results

        Returns:
            List of analysis result dictionaries
        """
        from medgemma import hash_image

        verbose = self.config.verbose

        # Extract images and contexts
        images = [fig.image for fig in figures]
        contexts = [fig.nearby_text or "" for fig in figures]

        # Run batch analysis
        with timed_operation(f"MedGemma batch analysis [{len(figures)} figures]", verbose=verbose):
            analyses = self.captioner.analyze_figures_batch(images, contexts)

        # Process each analysis result
        results = []
        for figure, analysis in zip(figures, analyses):
            original_hash = hash_image(figure.image)

            # Convert to image generation prompt
            brief = self.prompt_adapter.convert(analysis)

            # Create figure metadata
            fig_metadata = FigureMetadata(
                figure_id=figure.figure_id,
                page_num=figure.page_num,
                original_hash=original_hash,
                figure_type=analysis.figure_type.value,
                anatomical_region=analysis.anatomical_region,
                caption=analysis.caption or "",
                teaching_point=analysis.teaching_point,
                generation_prompt=brief.prompt,
                bbox=figure.bbox,
                nearby_text=figure.nearby_text[:500] if figure.nearby_text else "",
            )

            # Save the original figure
            bundler.save_figure(
                figure.figure_id,
                figure.image,
                None,
                fig_metadata,
            )

            # Build result dict
            results.append({
                "figure_id": figure.figure_id,
                "page_num": figure.page_num,
                "original_hash": original_hash,
                "image_path": f"figures/figure_{figure.figure_id}_original.png",
                "analysis": {
                    "figure_type": analysis.figure_type.value,
                    "anatomical_region": analysis.anatomical_region,
                    "entities": analysis.entities,
                    "relationships": analysis.relationships,
                    "findings": analysis.findings,
                    "teaching_point": analysis.teaching_point,
                    "caption": analysis.caption,
                    "constraints": {
                        "anatomical": analysis.constraints.anatomical,
                        "style": analysis.constraints.style,
                        "labels": analysis.constraints.labels,
                    },
                },
                "generation_brief": {
                    "prompt": brief.prompt,
                    "negative_prompt": brief.negative_prompt,
                    "style": brief.style,
                    "aspect_ratio": brief.aspect_ratio,
                    "constraints": brief.constraints,
                },
                "nearby_text": figure.nearby_text[:500] if figure.nearby_text else "",
            })

        logger.info(f"Batch analysis complete: {len(results)} figures processed")
        return results

    def _save_analysis_results(
        self,
        results: list[dict],
        output_dir: Path,
    ):
        """Save analysis results to JSON file."""
        import json

        output_path = output_dir / "analysis_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved analysis results to {output_path}")

    def _process_figure(
        self,
        figure: DetectedFigure,
        bundler: OutputBundler,
    ):
        """Process a single figure through analysis, generation, and evaluation."""
        from medgemma import hash_image

        verbose = self.config.verbose

        # Compute original hash
        original_hash = hash_image(figure.image)

        # Analyze figure with MedGemma
        with timed_operation(f"MedGemma analysis [{figure.figure_id}]", verbose=verbose):
            analysis = self.captioner.analyze_figure(
                figure.image,
                context=figure.nearby_text,
            )

        # Convert to image generation prompt
        brief = self.prompt_adapter.convert(analysis)
        original_brief = brief  # Keep original for fallback

        # Sanitize prompt to avoid safety filter blocks (if enabled)
        is_sanitized = False
        if self.config.sanitize_prompts:
            sanitizer = get_sanitizer()
            brief = sanitizer.sanitize_brief(brief)
            is_sanitized = True
            logger.debug(f"Sanitized prompt for {figure.figure_id}")
        else:
            logger.info(f"Using original (unsanitized) prompt for {figure.figure_id}")
            logger.debug(f"Original prompt: {brief.prompt[:200]}...")

        # Generate new image (img2img conversion from original)
        # With fallback: if safety filter blocks and not sanitized, retry with sanitization
        with timed_operation(f"Image generation [{figure.figure_id}]", verbose=verbose):
            generated_images, was_sanitized, sanitization_note = self._generate_with_fallback(
                brief=brief,
                original_brief=original_brief,
                figure=figure,
                is_sanitized=is_sanitized,
            )

        # Log if sanitization was applied during fallback
        if was_sanitized and not is_sanitized:
            logger.info(f"[{figure.figure_id}] {sanitization_note}")

        # Use first generated image
        generated = generated_images[0] if generated_images else None
        generated_image = generated.image if generated else None
        generated_hash = hash_image(generated_image) if generated_image else None

        # Evaluate alignment (only if enabled)
        alignment_result = None
        if self.config.run_alignment_eval and generated_image:
            with timed_operation(f"Alignment evaluation [{figure.figure_id}]", verbose=verbose):
                alignment_result = self.alignment_scorer.score(
                    figure.image,
                    generated_image,
                    figure.nearby_text,
                )

        # Create figure metadata
        fig_metadata = FigureMetadata(
            figure_id=figure.figure_id,
            page_num=figure.page_num,
            original_hash=original_hash,
            generated_hash=generated_hash,
            figure_type=analysis.figure_type.value,
            anatomical_region=analysis.anatomical_region,
            caption=analysis.caption or "",
            teaching_point=analysis.teaching_point,
            generation_prompt=brief.prompt,
            generation_seed=generated.seed if generated else None,
            prompt_sanitized=was_sanitized,
            sanitization_note=sanitization_note,
            alignment_score=alignment_result.alignment_score if alignment_result else None,
            alignment_status=alignment_result.status.value if alignment_result else None,
            alignment_reasoning=alignment_result.reasoning if alignment_result else "",
            bbox=figure.bbox,
            nearby_text=figure.nearby_text[:500],  # Truncate
        )

        # Save to bundle
        bundler.save_figure(
            figure.figure_id,
            figure.image,
            generated_image,
            fig_metadata,
        )

    def _generate_with_fallback(
        self,
        brief,
        original_brief,
        figure: DetectedFigure,
        is_sanitized: bool,
        max_retries: int = 3,
    ) -> tuple[list, bool, str]:
        """
        Generate image with retry logic and fallback to sanitized prompt.

        Strategy:
        1. Try original prompt up to max_retries times
        2. If all retries fail due to safety filter, apply sanitization
        3. Track whether sanitization was applied

        Args:
            brief: Current image brief (may be sanitized or not)
            original_brief: Original unsanitized brief for reference
            figure: The figure being processed
            is_sanitized: Whether the current brief is already sanitized
            max_retries: Number of retries before falling back to sanitization

        Returns:
            Tuple of (generated_images, was_sanitized, sanitization_note)
        """
        last_error = None

        # Try with current prompt (original or pre-sanitized) up to max_retries
        for attempt in range(1, max_retries + 1):
            try:
                if isinstance(self.image_generator, CachedImageGenerator):
                    result = self.image_generator.generate(
                        brief,
                        figure.figure_id,
                        self.config.generation_variants,
                    )
                else:
                    result = self.image_generator.generate(
                        brief,
                        original_image=figure.image,
                        num_variants=self.config.generation_variants,
                        strength=self.config.generation_strength,
                    )
                # Success
                return (result, is_sanitized, "" if not is_sanitized else "Pre-sanitized by config")

            except SafetyFilterError as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        f"Safety filter blocked [{figure.figure_id}] "
                        f"(attempt {attempt}/{max_retries}), retrying..."
                    )
                else:
                    logger.warning(
                        f"Safety filter blocked [{figure.figure_id}] "
                        f"after {max_retries} attempts"
                    )

        # All retries failed - try with sanitized prompt if not already sanitized
        if not is_sanitized:
            logger.info(
                f"Applying sanitization for [{figure.figure_id}] after {max_retries} failed attempts"
            )
            logger.debug(f"Original prompt: {original_brief.prompt[:200]}...")

            sanitizer = get_sanitizer()
            sanitized_brief = sanitizer.sanitize_brief(original_brief)

            logger.info(f"Sanitized prompt: {sanitized_brief.prompt[:200]}...")

            try:
                if isinstance(self.image_generator, CachedImageGenerator):
                    result = self.image_generator.generate(
                        sanitized_brief,
                        figure.figure_id,
                        self.config.generation_variants,
                    )
                else:
                    result = self.image_generator.generate(
                        sanitized_brief,
                        original_image=figure.image,
                        num_variants=self.config.generation_variants,
                        strength=self.config.generation_strength,
                    )

                # Build sanitization note
                sanitization_note = (
                    f"Prompt was sanitized after {max_retries} safety filter blocks. "
                    f"Original terms replaced for content policy compliance."
                )
                return (result, True, sanitization_note)

            except SafetyFilterError:
                logger.error(
                    f"Safety filter blocked [{figure.figure_id}] even with sanitized prompt"
                )
                raise
        else:
            # Already sanitized but still blocked
            logger.error(
                f"Safety filter blocked [{figure.figure_id}] even with sanitized prompt"
            )
            raise last_error

    def _compute_file_hash(self, path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def process_cached(
        self,
        cache_name: str,
        output_dir: Optional[Path | str] = None,
    ) -> OutputBundle:
        """
        Process using cached example data.

        Args:
            cache_name: Name of cached example (e.g., "breech_types")
            output_dir: Output directory

        Returns:
            OutputBundle with cached results
        """
        cache_path = self.config.cache_dir / cache_name / "input.pdf"
        if not cache_path.exists():
            raise FileNotFoundError(f"Cached example not found: {cache_path}")

        return self.process(cache_path, output_dir)


def run_pipeline(
    input_path: str,
    output_dir: str = "output",
    mode: str = "reproducible",
    verbose: bool = False,
) -> OutputBundle:
    """
    Convenience function to run the pipeline.

    Args:
        input_path: Path to input PDF
        output_dir: Output directory
        mode: Pipeline mode ("full" or "reproducible")
        verbose: Enable verbose logging

    Returns:
        OutputBundle with results
    """
    config = PipelineConfig.from_mode(
        mode,
        output_dir=Path(output_dir),
        verbose=verbose,
    )

    modernizer = DocumentModernizer(config)
    return modernizer.process(input_path)
