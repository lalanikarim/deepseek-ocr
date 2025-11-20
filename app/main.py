import os
import base64
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
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


@app.post("/api/ocr")
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
        JSON with text result and annotated image (if detections found)
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
        
        # Ensure image is loaded and get its size
        image.load()  # Force image to load fully
        print(f"[DEBUG] Original image size: {image.size}, mode: {image.mode}")
        
        # Perform OCR with operation and get annotated image
        result, annotated_image = await ocr_service.perform_ocr_with_annotations(image, operation.strip())
        
        # Prepare response
        response_data = {
            "text": result,
            "annotated_image": None
        }
        
        # Convert annotated image to base64 if available
        if annotated_image:
            buffered = BytesIO()
            annotated_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            response_data["annotated_image"] = f"data:image/png;base64,{img_str}"
        
        return JSONResponse(content=response_data)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

