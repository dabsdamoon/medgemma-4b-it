"""
Alignment scoring module using MedGemma.

Evaluates how well generated figures preserve medical content from originals.
"""

import logging
from typing import Optional

from PIL import Image

from ..understanding.captioner import FigureCaptioner
from ..understanding.schema import AlignmentResult, AlignmentStatus

logger = logging.getLogger(__name__)


class AlignmentScorer:
    """
    Score alignment between original and generated medical figures.

    Uses MedGemma to evaluate content preservation, anatomical accuracy,
    and educational value.
    """

    def __init__(
        self,
        pass_threshold: float = 0.80,
        flag_threshold: float = 0.65,
        captioner: Optional[FigureCaptioner] = None,
        use_quantization: bool = True,
    ):
        """
        Initialize the alignment scorer.

        Args:
            pass_threshold: Score threshold for pass status (>= this = pass)
            flag_threshold: Score threshold for flag status (>= this, < pass = flag)
            captioner: Optional pre-configured FigureCaptioner
            use_quantization: Whether to use quantized model if creating captioner
        """
        self.pass_threshold = pass_threshold
        self.flag_threshold = flag_threshold

        if captioner is not None:
            self._captioner = captioner
        else:
            self._captioner = FigureCaptioner(use_quantization=use_quantization)

    def score(
        self,
        original: Image.Image,
        generated: Image.Image,
        context: str = "",
    ) -> AlignmentResult:
        """
        Score alignment between original and generated figures.

        Args:
            original: Original medical figure
            generated: Generated reproduction
            context: Optional context text

        Returns:
            AlignmentResult with scores and status
        """
        return self._captioner.evaluate_alignment(
            original=original,
            generated=generated,
            pass_threshold=self.pass_threshold,
            flag_threshold=self.flag_threshold,
        )

    def score_batch(
        self,
        pairs: list[tuple[Image.Image, Image.Image]],
        contexts: Optional[list[str]] = None,
    ) -> list[AlignmentResult]:
        """
        Score multiple original/generated pairs.

        Args:
            pairs: List of (original, generated) image tuples
            contexts: Optional list of context strings

        Returns:
            List of AlignmentResult objects
        """
        if contexts is None:
            contexts = [""] * len(pairs)

        results = []
        for (original, generated), context in zip(pairs, contexts):
            try:
                result = self.score(original, generated, context)
                results.append(result)
            except Exception as e:
                logger.error(f"Error scoring pair: {e}")
                # Return a flagged result for errors
                results.append(
                    AlignmentResult(
                        alignment_score=0.5,
                        status=AlignmentStatus.FLAG,
                        anatomical_accuracy=0.5,
                        content_preservation=0.5,
                        educational_value=0.5,
                        reasoning=f"Error during evaluation: {str(e)}",
                    )
                )

        return results

    def get_summary_stats(self, results: list[AlignmentResult]) -> dict:
        """
        Get summary statistics for a batch of results.

        Args:
            results: List of AlignmentResult objects

        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {
                "count": 0,
                "avg_score": 0.0,
                "pass_count": 0,
                "flag_count": 0,
                "fail_count": 0,
                "pass_rate": 0.0,
            }

        scores = [r.alignment_score for r in results]
        statuses = [r.status for r in results]

        pass_count = sum(1 for s in statuses if s == AlignmentStatus.PASS)
        flag_count = sum(1 for s in statuses if s == AlignmentStatus.FLAG)
        fail_count = sum(1 for s in statuses if s == AlignmentStatus.FAIL)

        return {
            "count": len(results),
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "pass_count": pass_count,
            "flag_count": flag_count,
            "fail_count": fail_count,
            "pass_rate": pass_count / len(results),
            "avg_anatomical_accuracy": sum(r.anatomical_accuracy for r in results)
            / len(results),
            "avg_content_preservation": sum(r.content_preservation for r in results)
            / len(results),
            "avg_educational_value": sum(r.educational_value for r in results)
            / len(results),
        }
