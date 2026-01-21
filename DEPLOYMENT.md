# MedGemma RunPod Deployment Guide

This guide walks you through deploying MedGemma-1.5-4B-IT as a serverless API on RunPod.

## Prerequisites

1. [RunPod Account](https://console.runpod.io/)
2. [Docker Hub Account](https://hub.docker.com/) (or other container registry)
3. [HuggingFace Account](https://huggingface.co/) with access to MedGemma model
4. Docker installed locally

## Quick Start

### Step 1: Set Up HuggingFace Token

MedGemma requires accepting the license on HuggingFace.

1. Go to [MedGemma model page](https://huggingface.co/google/medgemma-1.5-4b-it)
2. Accept the license terms
3. Create an access token at [HuggingFace Settings](https://huggingface.co/settings/tokens)

### Step 2: Build and Push Docker Image

```bash
# Login to Docker Hub
docker login

# Build the image
docker build -t yourusername/medgemma-runpod:latest .

# For faster cold starts, pre-download the model (larger image ~15GB)
docker build --build-arg PRELOAD_MODEL=true -t yourusername/medgemma-runpod:latest .

# Push to Docker Hub
docker push yourusername/medgemma-runpod:latest
```

### Step 3: Create RunPod Serverless Endpoint

1. Go to [RunPod Console](https://console.runpod.io/)
2. Navigate to **Serverless** → **Endpoints**
3. Click **New Endpoint**
4. Configure:

| Setting | Value |
|---------|-------|
| Endpoint Name | `medgemma-summarizer` |
| Container Image | `yourusername/medgemma-runpod:latest` |
| GPU Type | L4 (24GB) or RTX 4090 (24GB) |
| Active Workers | 0 (scale to zero) |
| Max Workers | 3 |
| Idle Timeout | 5 seconds |
| Execution Timeout | 300 seconds |

5. Add Environment Variables:

| Variable | Value |
|----------|-------|
| `HF_TOKEN` | Your HuggingFace token |

6. Click **Create Endpoint**

### Step 4: Test Your Endpoint

Get your Endpoint ID and API Key from the RunPod dashboard.

#### Using cURL

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "text": "Patient: John Doe, 65-year-old male\nChief Complaint: Chest pain\nAssessment: NSTEMI\nPlan: Admit to CCU, dual antiplatelet therapy",
      "max_tokens": 256
    }
  }'
```

#### Using Python

```python
import requests

RUNPOD_API_KEY = "your_api_key"
ENDPOINT_ID = "your_endpoint_id"

def summarize_medical_document(text: str, max_tokens: int = 512) -> str:
    response = requests.post(
        f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "input": {
                "text": text,
                "max_tokens": max_tokens
            }
        },
        timeout=300
    )
    result = response.json()

    if "output" in result:
        return result["output"]["summary"]
    elif "error" in result:
        raise Exception(result["error"])
    else:
        raise Exception(f"Unexpected response: {result}")

# Example usage
document = """
Patient: John Doe, 65-year-old male
Date of Visit: January 15, 2026
Chief Complaint: Chest pain and shortness of breath

Assessment: Acute coronary syndrome - NSTEMI

Plan:
1. Admit to cardiac care unit
2. Start dual antiplatelet therapy
3. Cardiology consult for angiography
"""

summary = summarize_medical_document(document)
print(summary)
```

## API Reference

### Request Format

**Endpoint:** `POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync`

**Headers:**
```json
{
  "Authorization": "Bearer YOUR_API_KEY",
  "Content-Type": "application/json"
}
```

**Body:**
```json
{
  "input": {
    "text": "Medical document text...",
    "max_tokens": 512
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | Medical document to summarize |
| `max_tokens` | integer | No | Max tokens to generate (default: 512) |

### Response Format

**Success:**
```json
{
  "id": "job-id",
  "status": "COMPLETED",
  "output": {
    "summary": "Generated summary text..."
  }
}
```

**Error:**
```json
{
  "id": "job-id",
  "status": "FAILED",
  "error": "Error message..."
}
```

## Async Requests (Recommended for Production)

For longer documents, use async requests:

```python
import requests
import time

def summarize_async(text: str) -> str:
    # Start the job
    response = requests.post(
        f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run",
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
        json={"input": {"text": text}}
    )
    job_id = response.json()["id"]

    # Poll for completion
    while True:
        status = requests.get(
            f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{job_id}",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
        ).json()

        if status["status"] == "COMPLETED":
            return status["output"]["summary"]
        elif status["status"] == "FAILED":
            raise Exception(status.get("error", "Job failed"))

        time.sleep(1)
```

## Cost Optimization

### GPU Selection

| GPU | Price/hr | VRAM | Recommendation |
|-----|----------|------|----------------|
| RTX 3090 | $0.22 | 24GB | Budget option |
| RTX 4090 | $0.34 | 24GB | Good balance |
| L4 | $0.44 | 24GB | Reliable, data center |
| L40S | $0.79 | 48GB | If you need more VRAM |

### Scaling Settings

For cost optimization:
- **Active Workers: 0** - Scale to zero when idle
- **Max Workers: 1-3** - Limit concurrent requests
- **Idle Timeout: 5s** - Quick scale-down

For performance:
- **Active Workers: 1** - Always-warm instance
- **Max Workers: 5+** - Handle burst traffic

### Estimated Monthly Costs

| Usage Pattern | Requests/day | Est. Cost/month |
|---------------|--------------|-----------------|
| Light | 10 | ~$5-10 |
| Medium | 100 | ~$30-50 |
| Heavy | 1000 | ~$150-300 |

*Based on L4 GPU, ~30s per request, scale-to-zero enabled*

## Troubleshooting

### Cold Start Latency

First request after idle takes 30-60s (model loading). Solutions:
- Pre-download model in Docker image (`PRELOAD_MODEL=true`)
- Keep 1 active worker (costs more)

### Out of Memory

If you see OOM errors:
- Use a GPU with more VRAM (L40S: 48GB)
- Reduce `max_tokens`
- Enable 8-bit quantization in handler.py

### HuggingFace Authentication

If model download fails:
- Verify HF_TOKEN is set in RunPod environment
- Ensure you accepted the model license
- Check token has read permissions

## File Structure

```
medgemma-4b-it/
├── handler.py          # RunPod serverless handler
├── Dockerfile          # Container build file
├── requirements.txt    # Python dependencies
├── DEPLOYMENT.md       # This guide
├── summarize.py        # Local testing script
└── notes/              # Documentation
```
