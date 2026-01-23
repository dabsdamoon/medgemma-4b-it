# RunPod Serverless Dockerfile for MedGemma-1.5-4B-IT
#
# Environment variables:
#   USE_QUANTIZATION: "true" for 4-bit quantization (~4GB VRAM),
#                     "false" for bfloat16 precision (~8GB VRAM, better quality)
#                     Default: "true"

# Base image with CUDA and Python
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache
ENV TOKENIZERS_PARALLELISM=false
ENV USE_QUANTIZATION=true

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY medgemma/ ./medgemma/
COPY handler.py .

# Create cache directory
RUN mkdir -p /app/hf_cache

# Pre-download the model during build (optional but recommended)
# This downloads model weights without loading into GPU (for build time)
# Makes cold starts faster but increases image size (~8GB)
ARG PRELOAD_MODEL=false
ARG HF_TOKEN=""
RUN if [ "$PRELOAD_MODEL" = "true" ]; then \
    echo "Pre-downloading model weights..." && \
    python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('google/medgemma-1.5-4b-it', \
    cache_dir='/app/hf_cache', \
    token='${HF_TOKEN}' or None)"; \
    fi

# Healthcheck - RunPod will ping this
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('CUDA available:', torch.cuda.is_available())" || exit 1

# Run the handler
CMD ["python", "-u", "handler.py"]
