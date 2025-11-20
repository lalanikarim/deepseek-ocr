# DeepSeek OCR Web Application

A FastAPI web application with a simple HTML UI for performing OCR on images using deepseek-ocr via langchain-ollama.

## Features

- Upload image files for OCR processing
- Apply natural language operations to OCR results
- Simple, clean web interface
- Connects to deepseek-ocr model via Ollama

## Setup

1. Install dependencies using `uv`:
   ```bash
   uv sync
   ```

2. Configure environment variables in `.envrc`:
   - `OLLAMA_BASE_URL`: URL of your Ollama server (e.g., `http://localhost:11434`)

3. Ensure deepseek-ocr model is available on your Ollama server:
   ```bash
   ollama pull deepseek-ocr
   ```

## Running the Application

Start the FastAPI server:
```bash
uv run uvicorn app.main:app --reload
```

The application will be available at `http://localhost:8000`

## Usage

1. Open the web interface in your browser
2. Upload an image file
3. (Optional) Enter a natural language operation in the textarea (e.g., "Translate to English", "Extract only numbers", "Summarize the text")
4. Click "Process OCR" to extract text and apply the operation
5. View the result in the result area

## API Endpoints

- `GET /`: Serves the HTML web interface
- `POST /api/ocr`: Processes an image file with OCR
  - Form data:
    - `file`: Image file (required)
    - `operation`: Natural language operation (optional)
  - Returns: Plain text result

## Project Structure

```
deepseek-ocr/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── services/
│   │   ├── __init__.py
│   │   └── ocr_service.py   # OCR service using langchain-ollama
│   └── templates/
│       └── index.html       # Web UI
├── pyproject.toml
├── .envrc                   # Environment variables
└── README.md
```

