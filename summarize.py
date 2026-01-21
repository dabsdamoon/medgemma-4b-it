"""
Medical Document Summarization using MedGemma-1.5-4B-IT

Uses 4-bit quantization to run on RTX 2080 (8GB VRAM).
"""

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)


def create_dummy_image(size: tuple = (224, 224)) -> Image.Image:
    """Create a blank white image for text-only inference."""
    return Image.new("RGB", size, color=(255, 255, 255))


def load_model(model_id: str = "google/medgemma-1.5-4b-it"):
    """Load MedGemma model with 4-bit quantization for RTX 2080."""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(model_id)

    return model, processor


def summarize_medical_document(
    text: str,
    model,
    processor,
    max_new_tokens: int = 512,
    debug: bool = False,
    image: Image.Image = None,
) -> str:
    """
    Summarize a medical document.

    Args:
        text: The medical document text to summarize.
        model: The loaded MedGemma model.
        processor: The processor for the model.
        max_new_tokens: Maximum tokens to generate.
        debug: Whether to print debug information.
        image: Optional image (uses dummy image if not provided).

    Returns:
        The generated summary.
    """
    if image is None:
        image = create_dummy_image()

    # prompt = (
    #     "You are an expert medical professional. "
    #     "Provide a clear, accurate, and concise summary of the following medical document. "
    #     "Focus on key findings, diagnoses, treatments, and recommendations.\n\n"
    #     f"Medical Document:\n{text}\n\n"
    #     "Summary:"
    # )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful medical assistant."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "How do you differentiate bacterial from viral pneumonia?"}]
        }   
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.float16)

    input_len = inputs["input_ids"].shape[-1]

    import ipdb; ipdb.set_trace()

    if debug:
        print(f"[DEBUG] Input token length: {input_len}")
        input_text = processor.decode(inputs["input_ids"][0], skip_special_tokens=False)
        print(f"[DEBUG] Input text (first 500 chars):\n{input_text[:500]}...")

    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        if debug:
            print(f"[DEBUG] Full generation shape: {generation.shape}")
            print(f"[DEBUG] Full generation token length: {generation.shape[-1]}")
            full_output = processor.decode(generation[0], skip_special_tokens=False)
            print(f"[DEBUG] Full output (with special tokens):\n{full_output}")

        new_tokens = generation[0][input_len:]

        if debug:
            print(f"[DEBUG] New tokens length: {len(new_tokens)}")
            print(f"[DEBUG] New tokens: {new_tokens}")

    summary = processor.decode(new_tokens, skip_special_tokens=True)
    return summary


def main():
    print("Loading MedGemma-1.5-4B-IT with 4-bit quantization...")
    model, processor = load_model()
    print("Model loaded successfully!\n")

    # Example medical document
    sample_document = """
    Patient: John Doe, 65-year-old male
    Date of Visit: January 15, 2026
    Chief Complaint: Chest pain and shortness of breath

    History of Present Illness:
    The patient presents with intermittent chest pain for the past 3 days,
    described as pressure-like sensation in the substernal area, radiating
    to the left arm. Pain is exacerbated by exertion and relieved by rest.
    Associated symptoms include dyspnea on exertion and mild diaphoresis.

    Past Medical History:
    - Hypertension (diagnosed 2018)
    - Type 2 Diabetes Mellitus (diagnosed 2020)
    - Hyperlipidemia

    Current Medications:
    - Lisinopril 10mg daily
    - Metformin 500mg twice daily
    - Atorvastatin 20mg daily

    Physical Examination:
    - BP: 145/92 mmHg
    - HR: 88 bpm, regular
    - RR: 18/min
    - SpO2: 96% on room air
    - Cardiac: Regular rate and rhythm, no murmurs
    - Lungs: Clear to auscultation bilaterally

    Diagnostic Tests:
    - ECG: Sinus rhythm, ST depression in leads V4-V6
    - Troponin I: 0.08 ng/mL (elevated)
    - BNP: 180 pg/mL

    Assessment:
    Acute coronary syndrome - Non-ST elevation myocardial infarction (NSTEMI)

    Plan:
    1. Admit to cardiac care unit
    2. Start dual antiplatelet therapy (Aspirin + Clopidogrel)
    3. Initiate heparin infusion
    4. Cardiology consult for potential coronary angiography
    5. Continue current medications
    6. Serial troponin levels every 6 hours
    """

    print("Summarizing medical document...")
    print("-" * 50)

    summary = summarize_medical_document(sample_document, model, processor, debug=True)

    print("\nSummary:")
    print("-" * 50)
    print(f"'{summary}'")


if __name__ == "__main__":
    main()
