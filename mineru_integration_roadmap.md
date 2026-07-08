# MinerU Integration Roadmap for PIKE-RAG

## 1. Executive Summary
The current PIKE-RAG document ingestion relies on `PyMuPDF` (fitz) and `python-docx`, which are fast but lack advanced semantic understanding, accurate image/diagram extraction, and layout preservation (such as tables and formulas). **MinerU** is an advanced document parsing engine that outputs structured Markdown/JSON, perfectly suited for the Industrial Knowledge Brain (IKB) SaaS app. 

This roadmap outlines how to replace the current naive parsing pipeline with MinerU's API-driven extraction, system requirements for both local and cloud environments, and Hugging Face deployment strategies.

---

## 2. MinerU Features Relevant to IKB
To build a robust Industrial RAG SaaS app, the following MinerU features will be utilized:
* **Native Parsing:** High-precision extraction of PDF, DOCX, PPTX, and XLSX without hallucinations.
* **Layout Reconstruction:** Converts tables into HTML and formulas into LaTeX, which is critical for industrial SOPs and data sheets.
* **Smart Image Extraction:** Accurately extracts P&ID diagrams, component photos, and schematic charts (unlike PyMuPDF which often extracts raw byte artifacts or background noise).
* **VLM + OCR Dual Engine:** Automatically detects scanned PDFs/garbled text and applies OCR (supports 109 languages).
* **Semantic Chunking Support:** Outputs structured JSON ordered by human-reading flow, allowing us to chunk by paragraphs/sections rather than arbitrary character limits.

---

## 3. System Requirements & Running Locally
MinerU provides multiple inference backends depending on the hardware available. 

### Local CPU-Only Setup (Pipeline Backend)
* **Features Supported:** Full document parsing, OCR, table/formula extraction, and image extraction.
* **Hardware:** Pure CPU. Minimum 16GB RAM (32GB+ recommended), 20GB+ Disk Space (SSD recommended).
* **OS:** Windows (Native Python 3.10-3.12, or WSL2), Linux, macOS (14.0+).
* **Use Case:** Best for running directly on standard developer laptops or cost-effective edge servers.

### Local GPU Setup (Hybrid / VLM Engine)
* **Features Supported:** Highest accuracy parsing utilizing the `MinerU2.5-Pro` vision-language model, superior handling of extremely complex layouts.
* **Hardware:** GPU (Volta architecture or later / Apple Silicon). **Minimum 8GB VRAM**. 16GB RAM.
* **Use Case:** Best for dedicated local parsing servers or high-end workstations.

---

## 4. Hugging Face Deployment Strategy (API Access)
For your IKB SaaS app, you want to deploy MinerU on Hugging Face via Docker and access it via API from `app.py`.

### Hugging Face Space Configuration
* **Hardware Tier:** 
  * *Free Tier (CPU Basic):* You must use the `pipeline` backend. 
  * *GPU Tier (e.g., T4/A10G):* You can utilize the `vlm` or `hybrid` backends for maximum accuracy.
* **Docker Deployment:** MinerU provides an official `mineru-api` FastAPI wrapper. You can create a Hugging Face Docker Space with the following Dockerfile logic:
  ```dockerfile
  FROM opendatalab/mineru:latest
  
  # Set to pipeline for CPU-only spaces, or vlm for GPU spaces
  ENV MINERU_BACKEND="pipeline"
  ENV HF_HOME="/data" # Use persistent volume for model caching
  
  EXPOSE 8000
  CMD ["mineru-api", "--host", "0.0.0.0", "--port", "8000"]
  ```

### API Workflow for PIKE-RAG
1. **PIKE-RAG** sends the user-uploaded file (PDF/DOCX) to the Hugging Face MinerU API endpoint (`POST /file_parse` or via async `POST /tasks`).
2. **MinerU** returns a structured Markdown string, JSON layout data, and base64/URL references to extracted images (P&IDs, etc.).
3. **PIKE-RAG** chunks the Markdown natively, downloads the extracted images to `static_images/`, and embeds the text chunks into the Vector DB.

---

## 5. Integration Plan (Step-by-Step)

### Phase 1: Deploy MinerU API
1. Create a Hugging Face Space (Docker).
2. Use the MinerU base image and expose the FastAPI endpoints.
3. Test the deployment via Postman/cURL by sending a sample industrial PDF.

### Phase 2: Refactor `rag_utils.py`
Modify the `index_document` function in `pikerag/utils/rag_utils.py`:
1. **Remove** PyMuPDF (`fitz`) and `docx` library logic.
2. **Add** a `requests.post()` call to your Hugging Face MinerU API.
3. **Process Response:** Parse the returned Markdown. 
4. **Image Handling:** Extract the returned image bytes from the API response and save them to `static_images/`, injecting the `[Visual Reference Available at: ...]` tag into the corresponding text chunk.

### Phase 3: Implement Semantic Chunking
1. Instead of the current sliding window chunking (`chunk_size = 1000`), iterate over the JSON output provided by MinerU.
2. Group text by logical blocks (Headings, Paragraphs, Tables).
3. Embed these logical blocks using your existing Hugging Face Embeddings Space.

### Phase 4: UI & Pipeline Polish
1. Update `app.py` to handle potential API latency from the parsing step (consider adding a loading spinner or async progress bar in Gradio).
2. Verify that Groq (LLM) accurately references the new Markdown tables and HTML/LaTeX blocks generated by MinerU.
