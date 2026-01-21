"""
RunPod API Client for MedGemma Medical Document Summarization

Usage:
    python client.py --text "Your medical document..."
    python client.py --file document.txt

Environment variables:
    RUNPOD_API_KEY: Your RunPod API key
    RUNPOD_ENDPOINT_ID: Your endpoint ID
"""

import os
import sys
import time
import argparse
import requests


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

    def summarize_sync(self, text: str, max_tokens: int = 512, timeout: int = 300) -> str:
        """
        Synchronous summarization (waits for result).
        Best for short documents and quick responses.
        """
        response = requests.post(
            f"{self.base_url}/runsync",
            headers=self.headers,
            json={"input": {"text": text, "max_tokens": max_tokens}},
            timeout=timeout
        )
        response.raise_for_status()
        result = response.json()

        if result.get("status") == "COMPLETED":
            return result["output"]["summary"]
        elif "error" in result:
            raise Exception(f"API Error: {result['error']}")
        else:
            raise Exception(f"Unexpected response: {result}")

    def summarize_async(self, text: str, max_tokens: int = 512) -> str:
        """
        Asynchronous summarization (polls for result).
        Better for long documents to avoid timeout.
        """
        # Start the job
        response = requests.post(
            f"{self.base_url}/run",
            headers=self.headers,
            json={"input": {"text": text, "max_tokens": max_tokens}}
        )
        response.raise_for_status()
        job_id = response.json()["id"]
        print(f"Job started: {job_id}")

        # Poll for completion
        while True:
            status_response = requests.get(
                f"{self.base_url}/status/{job_id}",
                headers=self.headers
            )
            status_response.raise_for_status()
            status = status_response.json()

            print(f"Status: {status['status']}")

            if status["status"] == "COMPLETED":
                return status["output"]["summary"]
            elif status["status"] == "FAILED":
                raise Exception(f"Job failed: {status.get('error', 'Unknown error')}")
            elif status["status"] in ["IN_QUEUE", "IN_PROGRESS"]:
                time.sleep(2)
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


def main():
    parser = argparse.ArgumentParser(description="MedGemma Medical Document Summarizer")
    parser.add_argument("--text", type=str, help="Text to summarize")
    parser.add_argument("--file", type=str, help="File containing text to summarize")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--async", dest="use_async", action="store_true", help="Use async mode")
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

        print("Summarizing document...")
        print("-" * 50)

        if args.use_async:
            summary = client.summarize_async(text, args.max_tokens)
        else:
            summary = client.summarize_sync(text, args.max_tokens)

        print("\nSummary:")
        print("-" * 50)
        print(summary)

    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("\nSet environment variables:")
        print("  export RUNPOD_API_KEY=your_api_key")
        print("  export RUNPOD_ENDPOINT_ID=your_endpoint_id")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
