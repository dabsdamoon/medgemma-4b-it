"""
Model loading and inference for MedGemma.

Handles model initialization with optional quantization
and text/image generation.
"""

import os
import torch
import logging
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)
from .prompts import get_prompt
from .utils import create_dummy_image

logger = logging.getLogger(__name__)

# Global model instances (singleton pattern)
_model = None
_processor = None

MODEL_ID = "google/medgemma-1.5-4b-it"


def load_model(model_id: str = MODEL_ID, force_reload: bool = False):
    """
    Load MedGemma model with optional 4-bit quantization.

    Uses singleton pattern to avoid reloading on every request.
    Quantization controlled by USE_QUANTIZATION env var (default: "true").

    Args:
        model_id: HuggingFace model identifier
        force_reload: If True, reload even if already loaded

    Returns:
        tuple: (model, processor)
    """
    global _model, _processor

    if _model is not None and _processor is not None and not force_reload:
        logger.info("Using cached model instance")
        return _model, _processor

    use_quantization = os.getenv("USE_QUANTIZATION", "true").lower() == "true"

    if use_quantization:
        logger.info(f"Loading model: {model_id} with 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        _model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    else:
        logger.info(f"Loading model: {model_id} in bfloat16 precision...")
        _model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    _processor = AutoProcessor.from_pretrained(model_id)

    logger.info("Model loaded successfully!")
    return _model, _processor


def generate(
    text: str,
    model,
    processor,
    image=None,
    max_new_tokens: int = 512,
    mode: str = "general",
    system_prompt: str | None = None,
) -> str:
    """
    Generate a response using the MedGemma model.

    Args:
        text: Input text/query
        model: Loaded model instance
        processor: Loaded processor instance
        image: Optional PIL Image (uses dummy if None)
        max_new_tokens: Maximum tokens to generate
        mode: Processing mode ("summarize", "general")
        system_prompt: Optional custom prompt (overrides mode default)

    Returns:
        str: Generated response text
    """
    if image is None:
        image = create_dummy_image()

    # Get prompt (custom or default based on mode)
    prompt_text = get_prompt(mode, system_prompt)
    full_prompt = f"{prompt_text}\n\nInput Text:\n{text}\n\nResponse:"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": full_prompt},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)

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
