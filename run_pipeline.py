#!/usr/bin/env python3
"""
CLI entry point for the Legacy Medical Document Modernizer pipeline.

Usage:
    python run_pipeline.py --input path/to/document.pdf --mode reproducible
    python run_pipeline.py --input path/to/document.pdf --mode full --output results/
    python run_pipeline.py --cached breech_types
"""

# Suppress NumPy/Torch compatibility warnings before any imports
import os
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*NumPy.*")
warnings.filterwarnings("ignore", message=".*_ARRAY_API.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    # Set base level for our code
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Suppress noisy third-party library logs
    noisy_loggers = [
        "PIL",
        "urllib3",
        "httpcore",
        "httpx",
        "google",
        "google_genai",
        "google.auth",
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Legacy Medical Document Modernizer Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single image directly (no PDF extraction)
  python run_pipeline.py --input figure.png --analyze-only

  # Process multiple images directly
  python run_pipeline.py --images fig1.png fig2.png fig3.png --analyze-only

  # Extract figures and OCR from PDF (no API calls needed)
  python run_pipeline.py --input document.pdf --extract-only

  # Run MedGemma analysis on PDF, save results as JSON
  python run_pipeline.py --input document.pdf --analyze-only

  # Full pipeline with MedGemma + image generation
  python run_pipeline.py --input document.pdf --mode reproducible

  # Verbose output
  python run_pipeline.py --input document.pdf --analyze-only -v
        """,
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", "-i",
        type=str,
        help="Path to input file (PDF or image: png, jpg, etc.)",
    )
    input_group.add_argument(
        "--images",
        type=str,
        nargs="+",
        help="One or more image files to process directly (skips PDF extraction)",
    )
    input_group.add_argument(
        "--cached",
        type=str,
        help="Name of cached example to use (e.g., 'breech_types')",
    )

    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output",
        help="Output directory (default: output/)",
    )

    # Pipeline mode
    parser.add_argument(
        "--mode", "-m",
        choices=["full", "reproducible"],
        default="reproducible",
        help="Pipeline mode (default: reproducible)",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Extract figures and OCR only, skip AI analysis (no API calls needed)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Run MedGemma analysis but skip image generation. Saves analysis_results.json",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable local figure filtering (process all detected figures)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run alignment evaluation after image generation (disabled by default)",
    )
    parser.add_argument(
        "--no-sanitize",
        action="store_true",
        help="Disable prompt sanitization (may trigger safety filters)",
    )

    # OCR options
    parser.add_argument(
        "--ocr",
        choices=["tesseract", "easyocr", "google_vision"],
        help="OCR backend (overrides mode default)",
    )
    parser.add_argument(
        "--ocr-language",
        type=str,
        default="eng",
        help="OCR language code (default: eng)",
    )

    # Generation options
    parser.add_argument(
        "--variants",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Number of image variants to generate (default: 1)",
    )
    parser.add_argument(
        "--generator",
        choices=["vertex_flash", "vertex_pro", "cached"],
        default="vertex_flash",
        help="Image generator backend: vertex_flash (Nano Banana, default), vertex_pro (Nano Banana Pro), cached",
    )

    # Evaluation thresholds
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=0.80,
        help="Alignment score threshold for pass (default: 0.80)",
    )
    parser.add_argument(
        "--flag-threshold",
        type=float,
        default=0.65,
        help="Alignment score threshold for flag (default: 0.65)",
    )

    # Debug options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Save intermediate results",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Import pipeline modules
    try:
        from pipeline.config import (
            PipelineConfig,
            PipelineMode,
            OCRBackend,
            AlignmentThresholds,
            ImageGeneratorBackend,
        )
        from pipeline.orchestrator import DocumentModernizer
    except ImportError as e:
        logger.error(f"Failed to import pipeline modules: {e}")
        logger.error("Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)

    # Build configuration
    try:
        # Start with mode defaults
        if args.mode == "full":
            config = PipelineConfig.full()
        else:
            config = PipelineConfig.reproducible()

        # Override with CLI arguments
        config.output_dir = Path(args.output)
        config.verbose = args.verbose
        config.save_intermediates = args.save_intermediates
        config.extract_only = args.extract_only
        config.analyze_only = args.analyze_only
        config.filter_figures = not args.no_filter
        config.generation_variants = args.variants
        config.ocr_language = args.ocr_language
        config.run_alignment_eval = args.eval
        config.sanitize_prompts = not args.no_sanitize

        config.alignment_thresholds = AlignmentThresholds(
            pass_threshold=args.pass_threshold,
            flag_threshold=args.flag_threshold,
        )

        # Override OCR backend if specified
        if args.ocr:
            config.ocr_backend = OCRBackend(args.ocr)

        # Set image generator backend
        config.image_generator = ImageGeneratorBackend(args.generator)

    except Exception as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Create modernizer
    modernizer = DocumentModernizer(config)

    # Process document
    try:
        if args.cached:
            logger.info(f"Processing cached example: {args.cached}")
            bundle = modernizer.process_cached(args.cached, args.output)
        elif args.images:
            # Process multiple images directly
            image_paths = [Path(p) for p in args.images]
            for img_path in image_paths:
                if not img_path.exists():
                    logger.error(f"Image file not found: {img_path}")
                    sys.exit(1)

            logger.info(f"Processing {len(image_paths)} image(s) directly")
            bundle = modernizer.process_images(image_paths, args.output)
        else:
            input_path = Path(args.input)
            if not input_path.exists():
                logger.error(f"Input file not found: {input_path}")
                sys.exit(1)

            logger.info(f"Processing: {input_path}")
            bundle = modernizer.process(input_path, args.output)

        # Print summary
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Output directory: {bundle.output_dir}")
        print(f"Figures processed: {bundle.metadata.figures_processed}")
        print(f"Figures generated: {bundle.metadata.figures_generated}")

        if bundle.metadata.average_alignment_score is not None:
            print(f"\nAlignment Summary:")
            print(f"  Average score: {bundle.metadata.average_alignment_score:.2f}")
            print(f"  Pass: {bundle.metadata.alignment_pass_count}")
            print(f"  Flag: {bundle.metadata.alignment_flag_count}")
            print(f"  Fail: {bundle.metadata.alignment_fail_count}")

        if bundle.metadata.errors:
            print(f"\nErrors: {len(bundle.metadata.errors)}")
            for error in bundle.metadata.errors[:5]:
                print(f"  - {error}")

        print("\nOutput files:")
        if bundle.ocr_text_path:
            print(f"  - {bundle.ocr_text_path}")
        print(f"  - {bundle.metadata_path}")
        if args.analyze_only:
            analysis_path = bundle.output_dir / "analysis_results.json"
            if analysis_path.exists():
                print(f"  - {analysis_path}")
        print(f"  - {bundle.figures_dir}/ ({len(bundle.original_images)} original, {len(bundle.generated_images)} generated)")

    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
