# MedGemma-1.5-4B-IT Web Application

This project provides a web interface for the MedGemma-1.5-4B-IT medical summarization model.

## Prerequisites

- Python 3.10+
- Node.js 18+
- NVIDIA GPU with at least 6GB VRAM (8GB+ recommended) and CUDA drivers installed.

## Setup

### 1. Backend Setup

Navigate to the backend directory and install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

### 2. Frontend Setup

Navigate to the frontend directory and install dependencies:

```bash
cd frontend
npm install
```

## Running the Application

### 1. Start the Backend Server

In a terminal, run:

```bash
cd backend
python main.py
```
The backend will start at `http://0.0.0.0:8000`.
*Note: The first run will download the model (~4GB+), which may take some time.*

### 2. Start the Frontend Server

In a **separate** terminal, run:

```bash
cd frontend
npm run dev
```
The frontend will start at `http://localhost:5173`.

## Usage

1. Open your browser to `http://localhost:5173`.
2. Enter medical text in the text area.
3. (Optional) Upload a medical image.
4. Click "Analyze Document".
