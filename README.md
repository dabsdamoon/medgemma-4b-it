# Legacy Medical Document Modernizer

A pipeline that modernizes old medical documents by extracting figures, analyzing them with MedGemma, and generating modern schematic images using Vertex AI.

## Overview

This tool processes legacy medical documents (PDFs or images) through an AI-powered pipeline:

1. **Extract** figures and text from documents
2. **Analyze** medical figures using MedGemma (via RunPod)
3. **Generate** modernized images using Vertex AI (Gemini Flash/Pro)
4. **Evaluate** alignment between original and generated figures

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  PDF/Image      │────▶│  MedGemma       │────▶│  Vertex AI      │
│  Extraction     │     │  Analysis       │     │  Generation     │
│  (Local)        │     │  (RunPod)       │     │  (GCP)          │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                                ┌─────────────────┐
                                                │  Alignment      │
                                                │  Evaluation     │
                                                └─────────────────┘
```

## Project Structure

```
medgemma-4b-it/
├── medgemma/                 # MedGemma prompts and utilities
│   ├── prompts.py           # System prompts for different modes
│   └── utils.py             # Image encoding/hashing utilities
│
├── pipeline/                 # Main pipeline modules
│   ├── config.py            # Pipeline configuration
│   ├── orchestrator.py      # Main pipeline controller
│   ├── ingest/              # PDF extraction, figure detection
│   ├── ocr/                 # OCR backends (Tesseract, EasyOCR, Google Vision)
│   ├── understanding/       # MedGemma figure analysis
│   ├── generation/          # Image generation (Vertex AI)
│   └── evaluation/          # Alignment scoring
│
├── client.py                # RunPod API client for MedGemma
├── handler.py               # RunPod serverless handler (Docker)
├── run_pipeline.py          # CLI entry point
└── Dockerfile               # MedGemma deployment to RunPod
```

## Components

### MedGemma (RunPod)
- Runs on RunPod Serverless GPU infrastructure
- Provides medical figure analysis and captioning
- Supports batch processing for efficiency

### Vertex AI Image Generation
- **Gemini 2.5 Flash** (`vertex_flash`): Faster, cheaper - default option
- **Gemini 3 Pro** (`vertex_pro`): Higher quality output
- Uses img2img conversion to modernize existing figures

### Prompt Sanitizer
- Automatically sanitizes prompts to avoid Vertex AI safety filter blocks
- Replaces sensitive medical terms with clinical alternatives
- Preserves educational context while enabling generation

## Installation

```bash
# Clone and setup
git clone <repo-url>
cd medgemma-4b-it

# Create conda environment
conda create -n medgemma python=3.11
conda activate medgemma

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file with required credentials:

```bash
# RunPod (for MedGemma)
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_ENDPOINT_ID=your_endpoint_id

# GCP (for Vertex AI image generation)
GOOGLE_CLOUD_PROJECT=your_gcp_project
```

## Usage

### Pipeline Modes

| Mode | Description | API Calls |
|------|-------------|-----------|
| `--extract-only` | Extract figures and OCR text only | None |
| `--analyze-only` | Run MedGemma analysis, save JSON | RunPod only |
| `--mode reproducible` | Use cached images for demos | RunPod only |
| `--mode full` | Full pipeline with image generation | RunPod + Vertex AI |
| `--eval` | Run alignment evaluation (optional) | +RunPod (comparison) |

### Examples

**Extract figures from PDF (no API calls):**
```bash
python run_pipeline.py \
  --input document.pdf \
  --output output/extracted \
  --extract-only
```

**Analyze figures with MedGemma:**
```bash
python run_pipeline.py \
  --input medical_figure.png \
  --output output/analysis \
  --analyze-only \
  --verbose
```

**Full pipeline (analysis + generation):**
```bash
python run_pipeline.py \
  --input document.pdf \
  --output output/full \
  --mode full \
  --verbose
```

**Full pipeline with alignment evaluation:**
```bash
python run_pipeline.py \
  --input document.pdf \
  --output output/full \
  --mode full \
  --eval \
  --verbose
```

**Process multiple images:**
```bash
python run_pipeline.py \
  --images fig1.png fig2.png fig3.png \
  --output output/batch \
  --analyze-only
```

### CLI Options

```
Input:
  --input, -i         Path to input file (PDF or image)
  --images            Multiple image files to process
  --cached            Use cached example (for demos)

Output:
  --output, -o        Output directory (default: output/)

Pipeline:
  --mode              Pipeline mode: full | reproducible
  --extract-only      Extract only, skip AI analysis
  --analyze-only      Analyze only, skip image generation
  --no-filter         Process all detected figures
  --eval              Run alignment evaluation (disabled by default)

OCR:
  --ocr               Backend: tesseract | easyocr | google_vision
  --ocr-language      Language code (default: eng)

Generation:
  --variants          Number of image variants (1-4)
  --no-sanitize       Disable prompt sanitization (falls back if blocked)

Evaluation:
  --pass-threshold    Alignment pass threshold (default: 0.80)
  --flag-threshold    Alignment flag threshold (default: 0.65)

Debug:
  --verbose, -v       Enable verbose output
  --save-intermediates Save intermediate results
```

## Output Structure

```
output/
├── ocr_text.txt              # Extracted text from document
├── metadata.json             # Full pipeline metadata
├── analysis_results.json     # MedGemma analysis (analyze-only mode)
└── figures/
    ├── figure_img_0_original.png
    ├── figure_img_0_generated.png
    └── ...
```

## Deployment

### MedGemma on RunPod

```bash
# Build Docker image
docker build -t your-username/medgemma-runpod:latest .

# Push to Docker Hub
docker push your-username/medgemma-runpod:latest

# Deploy on RunPod Serverless
# Use the Docker image URL in RunPod console
```

## License

[Add license information]
