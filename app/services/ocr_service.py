import os
import base64
import re
from io import BytesIO
from typing import Optional, List, Dict
from PIL import Image, ImageDraw, ImageFont
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

    def _parse_detections(self, text: str) -> List[Dict]:
        """Parse detection tags from OCR response."""
        detections = []
        
        # Match ref and det pairs: <|ref|>name<|/ref|><|det|>[[coords]]<|/det|>
        ref_det_regex = r'<\|ref\|>(.*?)<\|/?ref\|><\|det\|>\[(.*?)\]<\|/?det\|>'
        
        for match in re.finditer(ref_det_regex, text, re.DOTALL):
            object_name = match.group(1).strip()
            coords_string = match.group(2).strip()
            
            # Parse all bounding boxes from the coordinates string
            box_regex = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
            
            for box_match in re.finditer(box_regex, coords_string):
                detections.append({
                    'name': object_name,
                    'x1': int(box_match.group(1)),
                    'y1': int(box_match.group(2)),
                    'x2': int(box_match.group(3)),
                    'y2': int(box_match.group(4))
                })
        
        return detections
    
    def _draw_bounding_boxes(self, image: Image.Image, detections: List[Dict]) -> Image.Image:
        """Draw bounding boxes and labels on the image."""
        if not detections:
            return image
        
        # Convert to RGB mode if necessary (PIL ImageDraw requires RGB)
        if image.mode != 'RGB':
            annotated_image = image.convert('RGB')
        else:
            annotated_image = image.copy()
        
        # Log image dimensions for debugging
        img_width, img_height = annotated_image.size
        print(f"[DEBUG] Image dimensions: {img_width}x{img_height}")
        print(f"[DEBUG] Number of detections: {len(detections)}")
        
        # Check if coordinates are normalized (0-999 range)
        # If all coordinates are <= 999, assume they're normalized to 1000x1000 grid
        max_coord = max(
            max(det['x1'], det['x2'], det['y1'], det['y2'])
            for det in detections
        ) if detections else 0
        
        is_normalized = max_coord <= 999
        
        if is_normalized:
            print(f"[DEBUG] Coordinates appear to be normalized (0-999), scaling to image size")
            # Scale factor from 1000x1000 grid to actual image size
            scale_x = img_width / 1000.0
            scale_y = img_height / 1000.0
        else:
            print(f"[DEBUG] Coordinates appear to be absolute pixel coordinates")
            scale_x = 1.0
            scale_y = 1.0
        
        draw = ImageDraw.Draw(annotated_image)
        
        # Colors for different object types
        colors = [
            (102, 126, 234),  # #667eea
            (240, 147, 251),  # #f093fb
            (79, 172, 254),   # #4facfe
            (67, 233, 123),   # #43e97b
            (250, 112, 154),  # #fa709a
            (254, 225, 64),   # #fee140
            (48, 207, 208),   # #30cfd0
            (168, 237, 234),  # #a8edea
        ]
        
        # Map object names to colors
        object_colors = {}
        color_index = 0
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
        
        for det in detections:
            # Get or assign color for this object type
            if det['name'] not in object_colors:
                object_colors[det['name']] = colors[color_index % len(colors)]
                color_index += 1
            
            box_color = object_colors[det['name']]
            
            # Calculate box coordinates (scale if normalized)
            x1 = min(det['x1'], det['x2']) * scale_x
            y1 = min(det['y1'], det['y2']) * scale_y
            x2 = max(det['x1'], det['x2']) * scale_x
            y2 = max(det['y1'], det['y2']) * scale_y
            
            # Convert to integers for drawing
            x1 = int(round(x1))
            y1 = int(round(y1))
            x2 = int(round(x2))
            y2 = int(round(y2))
            
            # Validate coordinates are within image bounds
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(0, min(x2, img_width - 1))
            y2 = max(0, min(y2, img_height - 1))
            
            print(f"[DEBUG] Drawing box for '{det['name']}': original=({det['x1']}, {det['y1']}, {det['x2']}, {det['y2']}), scaled=({x1}, {y1}) to ({x2}, {y2})")
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
            
            # Draw label background
            label_text = det['name']
            # Get text bounding box
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Draw label background rectangle
            label_y = max(0, y1 - text_height - 4)
            draw.rectangle(
                [x1, label_y, x1 + text_width + 8, label_y + text_height + 4],
                fill=box_color
            )
            
            # Draw label text
            draw.text(
                (x1 + 4, label_y + 2),
                label_text,
                fill=(255, 255, 255),
                font=font
            )
        
        return annotated_image
    
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
    
    async def perform_ocr_with_annotations(self, image: Image.Image, operation: str) -> tuple[str, Optional[Image.Image]]:
        """Perform OCR and return annotated image if detections are found."""
        ocr_result = await self.perform_ocr(image, operation)
        detections = self._parse_detections(ocr_result)
        
        annotated_image = None
        if detections:
            annotated_image = self._draw_bounding_boxes(image, detections)
        
        return ocr_result, annotated_image
