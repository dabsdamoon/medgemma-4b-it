"""
RunPod API Client for MedGemma Medical Document Processing

Usage:
    python client.py --text "Your medical document..."
    python client.py --file document.txt
    python client.py --text "Describe this scan" --image scan.png
    python client.py --mode general --text "What conditions cause this?"

Environment variables:
    RUNPOD_API_KEY: Your RunPod API key
    RUNPOD_ENDPOINT_ID: Your endpoint ID
"""

import logging
import os
import sys
import time
import base64
import argparse
import requests
from pathlib import Path

logger = logging.getLogger(__name__)


class MedGemmaClient:
    """Client for MedGemma RunPod Serverless API."""

    def __init__(self, api_key: str = None, endpoint_id: str = None):
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY")
        self.endpoint_id = endpoint_id or os.getenv("RUNPOD_ENDPOINT_ID")

        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY not set")
        if not self.endpoint_id:
            raise ValueError("RUNPOD_ENDPOINT_ID not set")

        self.base_url = f"https://api.runpod.ai/v2/{self.endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _build_payload(
        self,
        text: str,
        image_path: str = None,
        max_tokens: int = 512,
        mode: str = "summarize",
        system_prompt: str = None,
    ) -> dict:
        """Build the request payload."""
        payload = {
            "input": {
                "text": text,
                "max_tokens": max_tokens,
                "mode": mode,
            }
        }

        if system_prompt:
            payload["input"]["system_prompt"] = system_prompt

        if image_path:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            with open(image_path, "rb") as f:
                image_bytes = f.read()
            payload["input"]["image"] = base64.b64encode(image_bytes).decode("utf-8")

        return payload

    def _build_batch_payload(
        self,
        image_paths: list[str],
        texts: list[str] = None,
        max_tokens: int = 512,
        mode: str = "figure_analysis",
        system_prompt: str = None,
    ) -> dict:
        """Build the batch request payload."""
        # Encode all images
        images_b64 = []
        for image_path in image_paths:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            with open(image_path, "rb") as f:
                image_bytes = f.read()
            images_b64.append(base64.b64encode(image_bytes).decode("utf-8"))

        # Default texts to empty strings
        if texts is None:
            texts = [""] * len(image_paths)

        payload = {
            "input": {
                "batch": True,
                "images": images_b64,
                "texts": texts,
                "max_tokens": max_tokens,
                "mode": mode,
            }
        }

        if system_prompt:
            payload["input"]["system_prompt"] = system_prompt

        return payload

    def process_sync(
        self,
        text: str,
        image_path: str = None,
        max_tokens: int = 512,
        mode: str = "summarize",
        system_prompt: str = None,
        timeout: int = 300,
        poll_interval: int = 2,
    ) -> str:
        """
        Synchronous processing (waits for result).
        Best for short documents and quick responses.
        Falls back to polling if runsync times out.

        Args:
            text: Input text or query
            image_path: Optional path to an image file
            max_tokens: Maximum tokens to generate
            mode: "summarize" for document summarization, "general" for Q&A
            system_prompt: Custom system prompt (overrides mode default)
            timeout: Request timeout in seconds
            poll_interval: Seconds between status checks if polling

        Returns:
            str: Generated response
        """
        payload = self._build_payload(text, image_path, max_tokens, mode, system_prompt)

        response = requests.post(
            f"{self.base_url}/runsync",
            headers=self.headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        result = response.json()

        if result.get("status") == "COMPLETED":
            return result["output"]["output"]
        elif result.get("status") in ("IN_PROGRESS", "IN_QUEUE"):
            # runsync timed out or still queueing, fall back to polling
            job_id = result.get("id")
            if not job_id:
                raise Exception(f"{result.get('status')} but no job ID: {result}")
            logger.debug(f"Job {result.get('status')}, polling for completion: {job_id}")
            return self._poll_for_result(job_id, poll_interval)
        elif "error" in result.get("output", {}):
            raise Exception(f"API Error: {result['output']['error']}")
        elif "error" in result:
            raise Exception(f"API Error: {result['error']}")
        else:
            raise Exception(f"Unexpected response: {result}")

    def _poll_for_result(self, job_id: str, poll_interval: int = 2) -> str:
        """Poll for job completion and return result."""
        while True:
            status_response = requests.get(
                f"{self.base_url}/status/{job_id}",
                headers=self.headers
            )
            status_response.raise_for_status()
            status = status_response.json()

            if status["status"] == "COMPLETED":
                output = status.get("output", {})
                if "error" in output:
                    raise Exception(f"Job error: {output['error']}")
                return output.get("output", output.get("summary", str(output)))
            elif status["status"] == "FAILED":
                raise Exception(f"Job failed: {status.get('error', 'Unknown error')}")
            elif status["status"] in ["IN_QUEUE", "IN_PROGRESS"]:
                time.sleep(poll_interval)
            else:
                raise Exception(f"Unknown status: {status['status']}")

    def process_async(
        self,
        text: str,
        image_path: str = None,
        max_tokens: int = 512,
        mode: str = "summarize",
        system_prompt: str = None,
        poll_interval: int = 2,
    ) -> str:
        """
        Asynchronous processing (polls for result).
        Better for long documents to avoid timeout.

        Args:
            text: Input text or query
            image_path: Optional path to an image file
            max_tokens: Maximum tokens to generate
            mode: "summarize" for document summarization, "general" for Q&A
            system_prompt: Custom system prompt (overrides mode default)
            poll_interval: Seconds between status checks

        Returns:
            str: Generated response
        """
        payload = self._build_payload(text, image_path, max_tokens, mode, system_prompt)

        # Start the job
        response = requests.post(
            f"{self.base_url}/run",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        job_id = response.json()["id"]
        logger.debug(f"Job started: {job_id}")

        # Poll for completion
        while True:
            status_response = requests.get(
                f"{self.base_url}/status/{job_id}",
                headers=self.headers
            )
            status_response.raise_for_status()
            status = status_response.json()

            logger.debug(f"Status: {status['status']}")

            if status["status"] == "COMPLETED":
                output = status.get("output", {})
                if "error" in output:
                    raise Exception(f"Job error: {output['error']}")
                return output.get("output", output.get("summary", str(output)))
            elif status["status"] == "FAILED":
                raise Exception(f"Job failed: {status.get('error', 'Unknown error')}")
            elif status["status"] in ["IN_QUEUE", "IN_PROGRESS"]:
                time.sleep(poll_interval)
            else:
                raise Exception(f"Unknown status: {status['status']}")

    def process_batch_sync(
        self,
        image_paths: list[str],
        texts: list[str] = None,
        max_tokens: int = 512,
        mode: str = "figure_analysis",
        system_prompt: str = None,
        timeout: int = 600,
        poll_interval: int = 5,
    ) -> list[str]:
        """
        Synchronous batch processing (waits for all results).
        Best for small batches (< 5 images).
        Falls back to polling if runsync times out.

        Args:
            image_paths: List of paths to image files
            texts: Optional list of context texts (one per image)
            max_tokens: Maximum tokens to generate per image
            mode: Processing mode
            system_prompt: Custom system prompt (overrides mode default)
            timeout: Request timeout in seconds (longer for batches)
            poll_interval: Seconds between status checks if polling

        Returns:
            list[str]: List of generated responses
        """
        payload = self._build_batch_payload(
            image_paths, texts, max_tokens, mode, system_prompt
        )

        response = requests.post(
            f"{self.base_url}/runsync",
            headers=self.headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        result = response.json()

        if result.get("status") == "COMPLETED":
            output = result.get("output", {})
            if "error" in output:
                raise Exception(f"API Error: {output['error']}")
            return output.get("outputs", [])
        elif result.get("status") in ("IN_PROGRESS", "IN_QUEUE"):
            # runsync timed out or still queueing, fall back to polling
            job_id = result.get("id")
            if not job_id:
                raise Exception(f"{result.get('status')} but no job ID: {result}")
            logger.debug(f"Batch job {result.get('status')}, polling for completion: {job_id}")
            return self._poll_for_batch_result(job_id, poll_interval)
        elif "error" in result:
            raise Exception(f"API Error: {result['error']}")
        else:
            raise Exception(f"Unexpected response: {result}")

    def _poll_for_batch_result(self, job_id: str, poll_interval: int = 5) -> list[str]:
        """Poll for batch job completion and return results."""
        while True:
            status_response = requests.get(
                f"{self.base_url}/status/{job_id}",
                headers=self.headers
            )
            status_response.raise_for_status()
            status = status_response.json()

            if status["status"] == "COMPLETED":
                output = status.get("output", {})
                if "error" in output:
                    raise Exception(f"Batch job error: {output['error']}")
                outputs = output.get("outputs", [])
                errors = output.get("errors", [])
                if errors:
                    logger.warning(f"{len(errors)} errors in batch: {errors}")
                return outputs
            elif status["status"] == "FAILED":
                raise Exception(f"Batch job failed: {status.get('error', 'Unknown error')}")
            elif status["status"] in ["IN_QUEUE", "IN_PROGRESS"]:
                time.sleep(poll_interval)
            else:
                raise Exception(f"Unknown status: {status['status']}")

    def process_batch_async(
        self,
        image_paths: list[str],
        texts: list[str] = None,
        max_tokens: int = 512,
        mode: str = "figure_analysis",
        system_prompt: str = None,
        poll_interval: int = 5,
    ) -> list[str]:
        """
        Asynchronous batch processing (polls for results).
        Better for larger batches to avoid timeout.

        Args:
            image_paths: List of paths to image files
            texts: Optional list of context texts (one per image)
            max_tokens: Maximum tokens to generate per image
            mode: Processing mode
            system_prompt: Custom system prompt (overrides mode default)
            poll_interval: Seconds between status checks

        Returns:
            list[str]: List of generated responses
        """
        payload = self._build_batch_payload(
            image_paths, texts, max_tokens, mode, system_prompt
        )

        # Start the job
        response = requests.post(
            f"{self.base_url}/run",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        job_id = response.json()["id"]
        logger.debug(f"Batch job started: {job_id} ({len(image_paths)} images)")

        # Poll for completion
        while True:
            status_response = requests.get(
                f"{self.base_url}/status/{job_id}",
                headers=self.headers
            )
            status_response.raise_for_status()
            status = status_response.json()

            logger.debug(f"Status: {status['status']}")

            if status["status"] == "COMPLETED":
                output = status.get("output", {})
                if "error" in output:
                    raise Exception(f"Batch job error: {output['error']}")
                outputs = output.get("outputs", [])
                errors = output.get("errors", [])
                if errors:
                    logger.warning(f"{len(errors)} errors in batch: {errors}")
                return outputs
            elif status["status"] == "FAILED":
                raise Exception(f"Batch job failed: {status.get('error', 'Unknown error')}")
            elif status["status"] in ["IN_QUEUE", "IN_PROGRESS"]:
                time.sleep(poll_interval)
            else:
                raise Exception(f"Unknown status: {status['status']}")

    def health_check(self) -> dict:
        """Check endpoint health."""
        response = requests.get(
            f"{self.base_url}/health",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    # Convenience aliases
    def summarize(self, text: str, image_path: str = None, max_tokens: int = 512) -> str:
        """Summarize a medical document."""
        return self.process_sync(text, image_path, max_tokens, mode="summarize")

    def query(self, text: str, image_path: str = None, max_tokens: int = 512) -> str:
        """Ask a general medical question."""
        return self.process_sync(text, image_path, max_tokens, mode="general")

    def analyze_figures(
        self,
        image_paths: list[str],
        contexts: list[str] = None,
        max_tokens: int = 512,
        system_prompt: str = None,
    ) -> list[str]:
        """
        Analyze multiple medical figures in a single batch request.

        Args:
            image_paths: List of paths to medical figure images
            contexts: Optional list of context texts (e.g., nearby OCR text)
            max_tokens: Maximum tokens to generate per figure
            system_prompt: Custom system prompt for analysis

        Returns:
            list[str]: List of analysis results (one per image)
        """
        # Use async for batches > 3, sync for smaller batches
        if len(image_paths) > 3:
            return self.process_batch_async(
                image_paths, contexts, max_tokens, "figure_analysis", system_prompt
            )
        else:
            return self.process_batch_sync(
                image_paths, contexts, max_tokens, "figure_analysis", system_prompt
            )


def main():
    parser = argparse.ArgumentParser(
        description="MedGemma Medical Document Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Summarize a medical document
  python client.py --text "Patient presents with..."

  # Summarize from file
  python client.py --file report.txt

  # Analyze medical image with query
  python client.py --text "What abnormalities are visible?" --image xray.png --mode general

  # Use async mode for long documents
  python client.py --file long_report.txt --async
        """
    )
    parser.add_argument("--text", type=str, help="Text to process")
    parser.add_argument("--file", type=str, help="File containing text to process")
    parser.add_argument("--image", type=str, help="Path to medical image file")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--mode", type=str, default="summarize",
                       choices=["summarize", "general"],
                       help="Processing mode: 'summarize' or 'general'")
    parser.add_argument("--async", dest="use_async", action="store_true",
                       help="Use async mode (better for long documents)")
    parser.add_argument("--health", action="store_true", help="Check endpoint health")

    args = parser.parse_args()

    try:
        client = MedGemmaClient()

        if args.health:
            health = client.health_check()
            print(f"Endpoint Health: {health}")
            return

        # Get text from argument or file
        text = args.text
        if args.file:
            with open(args.file, "r") as f:
                text = f.read()

        if not text:
            print("Error: Provide --text or --file argument")
            sys.exit(1)

        print(f"Processing document (mode: {args.mode})...")
        if args.image:
            print(f"With image: {args.image}")
        print("-" * 50)

        if args.use_async:
            result = client.process_async(
                text, args.image, args.max_tokens, args.mode
            )
        else:
            result = client.process_sync(
                text, args.image, args.max_tokens, args.mode
            )

        print("\nResult:")
        print("-" * 50)
        print(result)

    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("\nSet environment variables:")
        print("  export RUNPOD_API_KEY=your_api_key")
        print("  export RUNPOD_ENDPOINT_ID=your_endpoint_id")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"File Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
