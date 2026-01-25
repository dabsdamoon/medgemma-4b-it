# MedGemma-4B-IT Context

## Project Overview
This project deploys the **Google MedGemma 1.5 4B IT** (Instruction Tuned) model as a serverless API on **RunPod**. It is designed for medical document summarization and general medical question-answering. The system supports multimodal input (text + optional image) but handles text-only requests by generating dummy placeholder images.

## Architecture

### Core Components
*   **`handler.py`**: The entry point for RunPod.
    *   `warmup()`: Loads the model and processor into memory (global state).
    *   `handler(job)`: Processes individual requests. Extracts inputs (`text`, `image`, `mode`), calls the model, and returns the response.
*   **`medgemma/`**:
    *   **`model.py`**: Handles model initialization (`load_model`) and inference (`generate`). Supports **4-bit quantization** via `bitsandbytes` to reduce VRAM usage (controlled by `USE_QUANTIZATION` env var).
    *   **`prompts.py`**: Manages system prompts. Defines default prompts for `"summarize"` and `"general"` modes. Allows runtime overrides via `custom_prompt`.
    *   **`utils.py`**: Utilities for image processing. `decode_image` handles Base64 inputs. `create_dummy_image` generates a white placeholder image for text-only requests (required by the multimodal architecture).
*   **`client.py`**: A Python client for interacting with the deployed RunPod endpoint. Supports synchronous (`process_sync`) and asynchronous (`process_async`) modes.

### Deployment (`Dockerfile`)
*   Base Image: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
*   **Environment Variables**:
    *   `USE_QUANTIZATION`: `"true"` (default) for 4-bit loading (saves VRAM), `"false"` for bfloat16.
    *   `HF_HOME`: Caches Hugging Face models.
*   **Build Arg**: `PRELOAD_MODEL` (boolean) to download weights during image build.

## Usage & Workflows

### Modes
1.  **Summarize** (`mode="summarize"`): Default. Uses a prompt focused on extracting key findings, diagnoses, and treatments.
2.  **General** (`mode="general"`): For open-ended Q&A.
3.  **Custom**: Users can provide a `system_prompt` in the API payload to override defaults.

### Local Development
*   **Testing Handler**: `tests/test_handler_local.py` loads the model locally (requires GPU) and mocks a RunPod job event.
*   **Unit Tests**: `tests/test_local.py` tests individual functions (prompt generation, etc.).

### Remote Interaction
*   **Client**: `python client.py --text "..."` or `python client.py --file doc.md --async`.
*   **Batch Processing**: `tests/test_batch.py` iterates over a folder of markdown files and sends them to the RunPod endpoint.

## Data
*   **`raw/`**: Directory containing raw medical markdown documents (e.g., `EP001.md`, `FF349.md`).
*   **`summaries/`**: (Proposed) Target directory for generated summaries.

## Key Environment Variables
*   **Local/Container**: `USE_QUANTIZATION` (Model loading precision).
*   **Client**: `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`.
