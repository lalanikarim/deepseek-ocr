# DeepSeek OCR Web Application

A FastAPI web application with a simple HTML UI for performing OCR on images using deepseek-ocr via langchain-ollama. The application can detect objects in images and draw bounding boxes with labels.

## Features

- Upload image files for OCR processing
- Apply natural language operations to OCR results (required)
- Automatic detection of objects with bounding boxes and labels
- Visual annotation of detected objects on the image
- Simple, clean web interface
- Connects to deepseek-ocr model via Ollama

## Requirements

- Python 3.13 or higher
- Ollama server with deepseek-ocr model installed
- `uv` package manager (recommended) or pip

## Setup

1. Install dependencies using `uv`:
   ```bash
   uv sync
   ```

   Or using pip:
   ```bash
   pip install -e .
   ```

2. Configure environment variables in `.envrc`:
   - `OLLAMA_BASE_URL`: URL of your Ollama server (e.g., `http://localhost:11434`)

   Example `.envrc`:
   ```bash
   export OLLAMA_BASE_URL=http://localhost:11434
   ```

3. Ensure deepseek-ocr model is available on your Ollama server:
   ```bash
   ollama pull deepseek-ocr
   ```

## Running the Application

Start the FastAPI server:
```bash
uv run uvicorn app.main:app --reload
```

Or with pip:
```bash
uvicorn app.main:app --reload
```

The application will be available at `http://localhost:8000`

## Usage

1. Open the web interface in your browser at `http://localhost:8000`
2. Upload an image file
3. Enter a natural language operation in the textarea (required). Examples:
   - `<|grounding|>OCR this image.` - General OCR with layout preservation
   - `<|grounding|>Convert the document to markdown.` - Convert to Markdown format
   - `Free OCR.` - Extract text without layout
   - `Describe this image in detail.` - Get detailed image description
   - `Parse the figure.` - Analyze figures and diagrams
   
   For more prompt examples and advanced usage, see the [DeepSeek-OCR Prompts](#deepseek-ocr-prompts) section below.
4. Click "Process OCR" to process the image
5. View the results:
   - **Text result**: The OCR text output (may include detection tags)
   - **Annotated image**: If detections are found, the image will be displayed with bounding boxes and labels drawn on it

For a comprehensive list of available prompts, see the [DeepSeek-OCR Prompts](#deepseek-ocr-prompts) section below.

## DeepSeek-OCR Prompts

DeepSeek-OCR supports various prompt patterns for different OCR tasks. The following prompts are commonly used and officially supported:

### Common Prompts

- **Document to Markdown Conversion**: Converts the document into a structured Markdown format.
  ```
  <|grounding|>Convert the document to markdown.
  ```

- **General OCR with Layout Preservation**: Performs OCR while maintaining the document's layout and structure.
  ```
  <|grounding|>OCR this image.
  ```

- **Free OCR (Text Extraction without Layout)**: Extracts text without preserving the original layout information.
  ```
  Free OCR.
  ```

- **Figure Parsing**: Analyzes and extracts information from figures, charts, or diagrams within the document.
  ```
  Parse the figure.
  ```

- **Detailed Image Description**: Provides a comprehensive description of the image content.
  ```
  Describe this image in detail.
  ```

- **Text Localization**: Locates and identifies specific text within the image, returning bounding box coordinates.
  ```
  Locate <|ref|>specific text<|/ref|> in the image.
  ```

### Special Tokens

DeepSeek-OCR uses special tokens to control behavior and output format:

- **`<|grounding|>`**: Enables the model to generate outputs with spatial information, such as bounding boxes. Use this token when you need object detection and localization capabilities. The model will return detection tags (`<|det|>`) with coordinate information.

- **`<|ref|>...<|/ref|>`**: Marks specific text spans for localization tasks. Use these tags to locate particular text within an image. The text between these tags will be searched for and its location returned with bounding box coordinates.

- **`<image>`**: This token is automatically handled by the application and langchain-ollama integration. You don't need to include it in your prompts.

### Prompt Tips

- Use `<|grounding|>` prefix when you need bounding box information for detected objects
- For simple text extraction without layout, use "Free OCR." without the grounding token
- Text localization prompts require the `<|ref|>...<|/ref|>` tags around the text you want to find
- You can combine natural language instructions with these special tokens for more specific tasks

## Detection Format

When the OCR model detects objects, it returns them in the following format:
- `<|ref|>ObjectName<|/ref|><|det|>[[x1, y1, x2, y2], ...]<|/det|>`
- Coordinates are normalized to a 1000x1000 grid (0-999)
- Multiple objects of the same type can be detected
- The application automatically scales coordinates to match the actual image size

## API Endpoints

### `GET /`
Serves the HTML web interface.

### `POST /api/ocr`
Processes an image file with OCR and optional object detection.

**Form data:**
- `file`: Image file (required)
- `operation`: Natural language operation to perform (required)

**Returns:** JSON response with:
```json
{
  "text": "OCR result text with detection tags",
  "annotated_image": "data:image/png;base64,..." // null if no detections
}
```

The `annotated_image` field contains a base64-encoded PNG image with bounding boxes and labels drawn on it, if detections were found in the OCR response.

## Project Structure

```
deepseek-ocr/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application and endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   └── ocr_service.py   # OCR service using langchain-ollama
│   │                          # Handles OCR, detection parsing, and image annotation
│   └── templates/
│       └── index.html       # Web UI with image upload and result display
├── pyproject.toml           # Project dependencies and build configuration
├── .gitignore               # Git ignore patterns
├── .envrc                   # Environment variables (not in git)
└── README.md
```

## How It Works

1. **Image Upload**: User uploads an image through the web interface
2. **OCR Processing**: The image is sent to deepseek-ocr model via langchain-ollama with the user's operation
3. **Detection Parsing**: If the response contains detection tags (`<|det|>`), they are parsed to extract bounding box coordinates
4. **Image Annotation**: Bounding boxes and labels are drawn on the image using PIL (Pillow)
5. **Response**: Both the text result and annotated image (if detections found) are returned to the frontend
6. **Display**: The frontend displays the text result and shows the annotated image with bounding boxes

## Notes

- Detection coordinates are normalized to a 1000x1000 grid and automatically scaled to the actual image dimensions
- The operation parameter is required and should be a natural language instruction for what to do with the image
- The application supports multiple detections per object type
- Each object type gets a unique color for its bounding boxes and labels

