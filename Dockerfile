# RunPod Serverless Dockerfile for MedGemma-1.5-4B-IT
# Base image with CUDA and Python
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY handler.py .

# Create cache directory
RUN mkdir -p /app/hf_cache

# Pre-download the model during build (optional but recommended)
# This makes cold starts faster but increases image size
ARG PRELOAD_MODEL=false
RUN if [ "$PRELOAD_MODEL" = "true" ]; then \
    python -c "from transformers import pipeline; import torch; \
    pipeline('image-text-to-text', model='google/medgemma-1.5-4b-it', \
    torch_dtype=torch.bfloat16, device='cpu')"; \
    fi

# Run the handler
CMD ["python", "-u", "handler.py"]
