import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)
import io

def create_dummy_image(size: tuple = (224, 224)) -> Image.Image:
    """Create a blank white image for text-only inference."""
    return Image.new("RGB", size, color=(255, 255, 255))

def load_model(model_id: str = "google/medgemma-1.5-4b-it"):
    """Load MedGemma model with 4-bit quantization."""
    print(f"Loading model: {model_id}")
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
    print("Model loaded.")
    return model, processor

def process_request(
    text: str,
    model,
    processor,
    image_bytes: bytes = None,
    max_new_tokens: int = 512,
) -> str:
    """
    Process a request with text and optional image.
    """
    if image_bytes:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    else:
        image = create_dummy_image()

    prompt = (
        "You are an expert medical professional. "
        "Provide a clear, accurate, and concise response to the following inputs. "
        "Focus on key findings, diagnoses, treatments, and recommendations if applicable.\n\n"
        f"Input Text:\n{text}\n\n"
        "Response:"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.float16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        new_tokens = generation[0][input_len:]

    result = processor.decode(new_tokens, skip_special_tokens=True)
    return result
