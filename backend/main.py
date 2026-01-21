from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Annotated
from model import load_model, process_request

# Global variables to hold model and processor
model = None
processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global model, processor
    try:
        model, processor = load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        # In a real app, might want to exit or handle this gracefully
    yield
    # Clean up resources if needed
    model = None
    processor = None

app = FastAPI(lifespan=lifespan)

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all. In prod, be specific.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "MedGemma Backend is running"}

@app.post("/process")
async def process_endpoint(
    text: Annotated[str, Form()],
    image: Annotated[UploadFile, File()] = None
):
    global model, processor
    if model is None or processor is None:
        return {"error": "Model not loaded"}

    image_bytes = None
    if image:
        image_bytes = await image.read()
    
    try:
        result = process_request(text, model, processor, image_bytes)
        return {"response": result}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
