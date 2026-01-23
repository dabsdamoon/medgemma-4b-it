"""
Local test script for RunPod handler (requires GPU).
Run this to test the handler before deploying to RunPod.

Usage: python tests/test_handler_local.py
       python tests/test_handler_local.py --with-image path/to/image.png
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import base64

# Import the actual handler
from handler import handler, warmup


def create_test_job(text: str, image_path: str = None, max_tokens: int = 256, mode: str = "summarize") -> dict:
    """Create a test job in RunPod format."""
    job = {
        "id": "test-job-001",
        "input": {
            "text": text,
            "max_tokens": max_tokens,
            "mode": mode,
        }
    }

    if image_path:
        image_path = Path(image_path)
        if image_path.exists():
            with open(image_path, "rb") as f:
                job["input"]["image"] = base64.b64encode(f.read()).decode("utf-8")
            print(f"Loaded image: {image_path}")
        else:
            print(f"Warning: Image not found: {image_path}")

    return job


def main():
    parser = argparse.ArgumentParser(description="Local test for MedGemma RunPod handler")
    parser.add_argument("--with-image", type=str, help="Path to test image")
    parser.add_argument("--mode", type=str, default="summarize", choices=["summarize", "general"])
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    print("=" * 60)
    print("Local Test for MedGemma Handler (Requires GPU)")
    print("=" * 60)

    # Warm up (load model)
    print("\nLoading model...")
    warmup()

    # Test document
    test_text = """
    Patient: John Doe, 65-year-old male
    Date of Visit: January 15, 2026
    Chief Complaint: Chest pain and shortness of breath

    History of Present Illness:
    Patient presented to ED with acute onset of chest pain radiating to left arm,
    associated with diaphoresis and dyspnea. Pain started 2 hours prior to arrival.
    Patient has history of hypertension and hyperlipidemia.

    Physical Examination:
    - BP: 165/95 mmHg
    - HR: 98 bpm
    - RR: 22/min
    - SpO2: 94% on room air
    - Cardiac: S1S2 present, no murmurs
    - Lungs: Clear to auscultation bilaterally

    Labs:
    - Troponin I: 2.5 ng/mL (elevated)
    - BNP: 450 pg/mL
    - Creatinine: 1.2 mg/dL
    - ECG: ST depression in leads V4-V6

    Assessment: Acute coronary syndrome - NSTEMI

    Plan:
    1. Admit to cardiac care unit
    2. Start dual antiplatelet therapy (Aspirin 325mg, Clopidogrel 75mg)
    3. Heparin drip
    4. Statin therapy (Atorvastatin 80mg)
    5. Cardiology consult for angiography within 24-48 hours
    6. Telemetry monitoring
    7. Serial troponins q6h
    """

    # Create test job
    test_job = create_test_job(
        text=test_text.strip(),
        image_path=args.with_image,
        max_tokens=args.max_tokens,
        mode=args.mode
    )

    print(f"\nTest Input (mode: {args.mode}):")
    print("-" * 40)
    print(test_text.strip()[:500] + "..." if len(test_text) > 500 else test_text.strip())
    print("-" * 40)

    print("\nProcessing...")
    result = handler(test_job)

    print("\nResult:")
    print("-" * 40)
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Mode: {result.get('mode', 'unknown')}")
        print(f"Output:\n{result['output']}")
    print("-" * 40)
    print("\nLocal test completed!")


if __name__ == "__main__":
    main()
