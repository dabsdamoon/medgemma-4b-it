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

    Supports two modes:

    1. Single image mode (backwards compatible):
    {
        "input": {
            "text": "Medical document text or query...",
            "image": "base64_encoded_image_string",  // optional
            "max_tokens": 512,                       // optional, default 512
            "mode": "summarize",                     // optional: "summarize" or "general"
            "system_prompt": "Custom prompt..."      // optional: overrides mode default
        }
    }

    2. Batch mode (multiple images in one request):
    {
        "input": {
            "batch": true,
            "images": ["base64_img1", "base64_img2", ...],
            "texts": ["context1", "context2", ...],  // optional, defaults to empty strings
            "max_tokens": 512,
            "mode": "figure_analysis",
            "system_prompt": "Custom prompt..."
        }
    }

    Returns (single mode):
    {
        "output": "Generated response...",
        "mode": "summarize"
    }

    Returns (batch mode):
    {
        "outputs": ["result1", "result2", ...],
        "mode": "figure_analysis",
        "count": 5
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

    try:
        # Ensure model is loaded
        if model is None or processor is None:
            logger.info("Model not loaded, loading now...")
            model, processor = load_model()

        # Check if batch mode
        if job_input.get("batch") or "images" in job_input:
            return _handle_batch(job_id, job_input)
        else:
            return _handle_single(job_id, job_input)

    except Exception as e:
        logger.exception(f"Job {job_id}: Error during processing")
        return {"error": str(e)}


def _handle_single(job_id: str, job_input: dict) -> dict:
    """Handle single image request (backwards compatible)."""
    global model, processor

    # Validate input
    text = job_input.get("text")
    if not text:
        logger.error(f"Job {job_id}: Missing 'text' field")
        return {"error": "Missing 'text' field in input"}

    # Get optional parameters
    image_data = job_input.get("image")
    max_tokens = job_input.get("max_tokens", 512)
    mode = job_input.get("mode", "summarize")
    system_prompt = job_input.get("system_prompt")

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


def _handle_batch(job_id: str, job_input: dict) -> dict:
    """Handle batch image request (multiple images in one call)."""
    global model, processor

    # Get images list
    images_data = job_input.get("images", [])
    if not images_data:
        logger.error(f"Job {job_id}: Missing 'images' field for batch mode")
        return {"error": "Missing 'images' field for batch mode"}

    # Get texts (optional, defaults to empty strings)
    texts = job_input.get("texts", [""] * len(images_data))
    if len(texts) != len(images_data):
        texts = texts + [""] * (len(images_data) - len(texts))

    # Get optional parameters
    max_tokens = job_input.get("max_tokens", 512)
    mode = job_input.get("mode", "figure_analysis")
    system_prompt = job_input.get("system_prompt")

    logger.info(f"Job {job_id}: Batch mode with {len(images_data)} images")

    if system_prompt:
        logger.info(f"Job {job_id}: Using custom prompt")

    # Process each image sequentially (same GPU memory as single image)
    results = []
    errors = []

    for idx, (image_data, text) in enumerate(zip(images_data, texts)):
        try:
            image = decode_image(image_data)

            result = generate(
                text=text,
                model=model,
                processor=processor,
                image=image,
                max_new_tokens=max_tokens,
                mode=mode,
                system_prompt=system_prompt,
            )

            results.append(result)
            logger.info(f"Job {job_id}: Processed image {idx + 1}/{len(images_data)}")

        except Exception as e:
            error_msg = f"Error processing image {idx}: {str(e)}"
            logger.error(f"Job {job_id}: {error_msg}")
            results.append(None)
            errors.append(error_msg)

    logger.info(f"Job {job_id}: Batch completed - {len(results)} results, {len(errors)} errors")

    response = {
        "outputs": results,
        "mode": mode,
        "count": len(results),
    }

    if errors:
        response["errors"] = errors

    return response


if __name__ == "__main__":
    # Load model at startup for warm starts
    warmup()

    # Start the serverless handler
    runpod.serverless.start({"handler": handler})
