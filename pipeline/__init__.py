"""
Legacy Medical Document Modernizer Pipeline.

This pipeline modernizes old medical documents by:
1. Extracting text/figures from PDFs
2. Using MedGemma to understand medical figures
3. Generating prompts for image generation
4. Evaluating alignment between old and new figures
"""

from .config import PipelineConfig, PipelineMode
from .orchestrator import DocumentModernizer

__all__ = [
    "PipelineConfig",
    "PipelineMode",
    "DocumentModernizer",
]
