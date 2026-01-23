"""
RunPod Serverless Handler for MedGemma-1.5-4B-IT

This handler processes medical document requests with optional image input.
Supports custom prompts via API input - no rebuild needed for prompt changes.

Environment variables:
    USE_QUANTIZATION: "true" for 4-bit, "false" for bfloat16 (default: "true")
"""

import runpod
import logging
from medgemma import load_model, generate, decode_image

# Configure logging for RunPod
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global model instances
model = None
processor = None


def warmup():
    """Load model at startup for warm starts."""
    global model, processor
    logger.info("Warming up: Loading model...")
    model, processor = load_model()
    logger.info("Model ready for inference")


def handler(job):
    """
    RunPod serverless handler function.

    Expected input format:
    {
        "input": {
            "text": "Medical document text or query...",
            "image": "base64_encoded_image_string",  // optional
            "max_tokens": 512,                       // optional, default 512
            "mode": "summarize",                     // optional: "summarize" or "general"
            "system_prompt": "Custom prompt..."      // optional: overrides mode default
        }
    }

    Returns:
    {
        "output": "Generated response...",
        "mode": "summarize"
    }

    Or on error:
    {
        "error": "Error message..."
    }
    """
    global model, processor

    job_input = job.get("input", {})
    job_id = job.get("id", "unknown")

    logger.info(f"Processing job {job_id}")

    # Validate input
    text = job_input.get("text")
    if not text:
        logger.error(f"Job {job_id}: Missing 'text' field")
        return {"error": "Missing 'text' field in input"}

    # Get optional parameters
    image_data = job_input.get("image")
    max_tokens = job_input.get("max_tokens", 512)
    mode = job_input.get("mode", "summarize")
    system_prompt = job_input.get("system_prompt")  # Custom prompt (optional)

    try:
        # Ensure model is loaded
        if model is None or processor is None:
            logger.info("Model not loaded, loading now...")
            model, processor = load_model()

        # Decode image if provided
        image = decode_image(image_data)
        logger.info(f"Job {job_id}: Image {'provided' if image_data else 'not provided (using dummy)'}")

        if system_prompt:
            logger.info(f"Job {job_id}: Using custom prompt")

        # Generate response
        result = generate(
            text=text,
            model=model,
            processor=processor,
            image=image,
            max_new_tokens=max_tokens,
            mode=mode,
            system_prompt=system_prompt,
        )

        logger.info(f"Job {job_id}: Completed successfully")
        return {
            "output": result,
            "mode": mode,
        }

    except Exception as e:
        logger.exception(f"Job {job_id}: Error during processing")
        return {"error": str(e)}


if __name__ == "__main__":
    # Load model at startup for warm starts
    warmup()

    # Start the serverless handler
    runpod.serverless.start({"handler": handler})
