import os
import base64
from io import BytesIO
from typing import Optional
from PIL import Image
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


class OCRService:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = "deepseek-ocr"
        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
        )

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    async def perform_ocr(self, image: Image.Image, operation: str) -> str:
        """Perform OCR on an image using deepseek-ocr model with the given operation."""
        try:
            # Convert image to base64
            img_base64 = self._image_to_base64(image)
            
            # Create message with image and operation text
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": operation
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    }
                ]
            )
            
            # Call the model
            response = await self.llm.ainvoke([message])
            return response.content.strip()
        except Exception as e:
            raise Exception(f"OCR failed: {str(e)}")
