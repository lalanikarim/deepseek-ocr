import os
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from dotenv import load_dotenv
from app.services.ocr_service import OCRService

# Load environment variables
load_dotenv()

app = FastAPI(title="DeepSeek OCR Web App")

# Add CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OCR service
ocr_service = OCRService()


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the HTML UI."""
    with open("app/templates/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/ocr", response_class=PlainTextResponse)
async def process_ocr(
    file: UploadFile = File(...),
    operation: str = Form(...)
):
    """
    Process an uploaded image with OCR and required operation.
    
    Args:
        file: The uploaded image file
        operation: Natural language operation to perform (required)
    
    Returns:
        Plain text result
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Validate operation
        if not operation or not operation.strip():
            raise HTTPException(status_code=400, detail="Operation is required")
        
        # Read and open image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data))
        
        # Perform OCR with operation
        result = await ocr_service.perform_ocr(image, operation.strip())
        
        return PlainTextResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

