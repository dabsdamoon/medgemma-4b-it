"""
RunPod Serverless Handler for MedGemma-1.5-4B-IT

This handler processes medical document summarization requests.
"""

import runpod
import torch
from transformers import pipeline
from PIL import Image

# Global model instance (loaded once, reused across requests)
pipe = None


def load_model():
    """Load the MedGemma model."""
    global pipe
    if pipe is None:
        print("Loading MedGemma-1.5-4B-IT model...")
        pipe = pipeline(
            "image-text-to-text",
            model="google/medgemma-1.5-4b-it",
            torch_dtype=torch.bfloat16,  # Required for Gemma 3 models
            device="cuda",
        )
        print("Model loaded successfully!")
    return pipe


def create_dummy_image(size=(224, 224)):
    """Create a blank image for text-only inference."""
    return Image.new("RGB", size, color=(255, 255, 255))


def summarize_document(text: str, max_tokens: int = 512) -> str:
    """Summarize a medical document."""
    model = load_model()

    prompt = f"""You are an expert medical professional.
Provide a clear, accurate, and concise summary of the following medical document.
Focus on key findings, diagnoses, treatments, and recommendations.

Medical Document:
{text}

Summary:"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": create_dummy_image()},
                {"type": "text", "text": prompt},
            ]
        }
    ]

    output = model(text=messages, max_new_tokens=max_tokens, do_sample=False)
    return output[0]["generated_text"][-1]["content"]


def handler(job):
    """
    RunPod serverless handler function.

    Expected input format:
    {
        "input": {
            "text": "Medical document text to summarize...",
            "max_tokens": 512  # optional
        }
    }

    Returns:
    {
        "summary": "Generated summary..."
    }
    """
    job_input = job.get("input", {})

    # Validate input
    text = job_input.get("text")
    if not text:
        return {"error": "Missing 'text' field in input"}

    max_tokens = job_input.get("max_tokens", 512)

    try:
        summary = summarize_document(text, max_tokens)
        return {"summary": summary}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Load model at startup (warm start)
    load_model()

    # Start the serverless handler
    runpod.serverless.start({"handler": handler})
