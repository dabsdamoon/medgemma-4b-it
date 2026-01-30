"""
Pipeline utilities.

Common utilities for the document modernizer pipeline.
"""

import logging
import time
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)


@contextmanager
def timed_operation(
    operation_name: str,
    verbose: bool = True,
    log_level: int = logging.INFO,
):
    """
    Context manager to measure and log execution time of an operation.

    Args:
        operation_name: Name of the operation (for logging)
        verbose: If True, log the timing information
        log_level: Logging level to use (default: INFO)

    Usage:
        with timed_operation("Figure analysis", verbose=True):
            result = analyze_figure(image)
    """
    start_time = time.perf_counter()

    if verbose:
        logger.log(log_level, f"Starting: {operation_name}")

    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time

        if verbose:
            if elapsed < 1:
                time_str = f"{elapsed * 1000:.0f}ms"
            elif elapsed < 60:
                time_str = f"{elapsed:.2f}s"
            else:
                minutes = int(elapsed // 60)
                seconds = elapsed % 60
                time_str = f"{minutes}m {seconds:.1f}s"

            logger.log(log_level, f"Completed: {operation_name} ({time_str})")


class Timer:
    """
    Reusable timer for measuring multiple operations.

    Usage:
        timer = Timer()

        timer.start("operation1")
        # ... do work ...
        timer.stop("operation1")

        timer.start("operation2")
        # ... do work ...
        timer.stop("operation2")

        timer.summary()  # Print all timings
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._starts: dict[str, float] = {}
        self._durations: dict[str, float] = {}

    def start(self, name: str):
        """Start timing an operation."""
        self._starts[name] = time.perf_counter()
        if self.verbose:
            logger.info(f"Starting: {name}")

    def stop(self, name: str) -> float:
        """Stop timing an operation and return duration."""
        if name not in self._starts:
            raise ValueError(f"Timer '{name}' was never started")

        elapsed = time.perf_counter() - self._starts[name]
        self._durations[name] = elapsed

        if self.verbose:
            logger.info(f"Completed: {name} ({self._format_time(elapsed)})")

        return elapsed

    def get(self, name: str) -> Optional[float]:
        """Get duration for a completed operation."""
        return self._durations.get(name)

    def summary(self) -> dict[str, float]:
        """Log and return summary of all timings."""
        if self.verbose and self._durations:
            logger.info("Timing Summary:")
            for name, duration in self._durations.items():
                logger.info(f"  {name}: {self._format_time(duration)}")

            total = sum(self._durations.values())
            logger.info(f"  Total: {self._format_time(total)}")

        return self._durations.copy()

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into human-readable string."""
        if seconds < 1:
            return f"{seconds * 1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        else:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
