"""
Local test script for RunPod handler.
Run this to test the handler before deploying to RunPod.

Usage: python test_handler_local.py
"""

import torch
from transformers import pipeline
from PIL import Image

# Global model instance
pipe = None

def load_model():
    """Load the MedGemma model."""
    global pipe
    if pipe is None:
        print("Loading MedGemma-1.5-4B-IT model...")
        pipe = pipeline(
            "image-text-to-text",
            model="google/medgemma-1.5-4b-it",
            torch_dtype=torch.bfloat16,
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

def test_handler(job):
    """Test the handler function locally."""
    job_input = job.get("input", {})
    text = job_input.get("text")
    if not text:
        return {"error": "Missing 'text' field in input"}

    max_tokens = job_input.get("max_tokens", 512)

    try:
        summary = summarize_document(text, max_tokens)
        return {"summary": summary}
    except Exception as e:
        return {"error": str(e)}

def main():
    print("=" * 60)
    print("Local Test for MedGemma Handler")
    print("=" * 60)

    # Simulate a RunPod job request
    test_job = {
        "input": {
            "text": """
            Patient: John Doe, 65-year-old male
            Date of Visit: January 15, 2026
            Chief Complaint: Chest pain and shortness of breath

            Assessment: Acute coronary syndrome - NSTEMI

            Plan:
            1. Admit to cardiac care unit
            2. Start dual antiplatelet therapy
            3. Cardiology consult for angiography
            """,
            "max_tokens": 256
        }
    }

    print("\nTest Input:")
    print("-" * 40)
    print(test_job["input"]["text"].strip())
    print("-" * 40)

    print("\nProcessing...")
    result = test_handler(test_job)

    print("\nResult:")
    print("-" * 40)
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(result["summary"])
    print("-" * 40)
    print("\nLocal test completed!")

if __name__ == "__main__":
    main()
