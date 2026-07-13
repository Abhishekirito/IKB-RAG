# IKB-RAG: Industrial Troubleshooting Assistant

A full-stack, scalable SaaS application utilizing a powerful RAG (Retrieval-Augmented Generation) pipeline designed specifically for industrial engineering. This platform uses **MinerU** for multimodal document parsing (tables, math, layout, diagrams), **Qdrant Cloud** for persistent, high-speed vector storage, and **Groq (Llama 3)** for low-latency, domain-expert troubleshooting.

---

## 🚀 Setup & Installation Instructions

To run this application locally, you need to start both the Python backend and the React frontend.

### 1. Prerequisites
- Python 3.9+
- Node.js & npm
- Valid API Keys for Hugging Face, Groq, Nomic, and Qdrant Cloud.

### 2. Backend Setup (FastAPI & RAG Engine)
Open a terminal in the root folder (`IKB-RAG-main`) and run:
```bash
# Install required Python dependencies
pip install fastapi uvicorn python-multipart requests groq qdrant-client python-dotenv

# Set your environment variables (in your terminal or .env file)
# Example .env file:
HF_TOKEN="your_huggingface_token"
GROQ_API_KEY="your_groq_api_key"
NOMIC_API_KEY="your_nomic_api_key"
QDRANT_URL="https://your-cluster-url.cloud.qdrant.io:6333"
QDRANT_API_KEY="your_qdrant_api_key"
```

Start the backend server:
```bash
python api.py
```
*When the backend starts, it will run a **Diagnostic Checklist** in the terminal to verify active connections to Groq, Qdrant Cloud, and Hugging Face spaces. The API will run on `http://localhost:8002`.*

### 3. Frontend Setup (React/Vite)
Open a **new** terminal window, navigate to the `frontend` folder, and run:
```bash
cd frontend

# Install Node modules
npm install

# Start the Vite Development Server
npm run dev
```
*The frontend will run on `http://localhost:5173`. Open this URL in your browser.*

---

## ⚙️ Core Architecture & Working Functions

This project is divided into a serverless-ready FastAPI backend and a dynamic React frontend with a glassmorphism aesthetic.

### A. Backend API Endpoints (`api.py`)
- **`POST /upload`**: Handles PDF uploads. Passes it to the RAG engine for parsing and indexing, and safely cleans up temporary files to maintain stateless operation.
- **`GET /documents`**: Returns a list of all uploaded PDFs associated with a specific `chat_id`.
- **`GET /documents/{chat_id}/{filename}`**: Serves the raw PDF file directly to the frontend for the built-in document viewer.
- **`GET /images/{chat_id}/{pdf_base}/{filename}`**: Dynamically serves precise industrial diagrams extracted by MinerU.
- **`POST /query`**: Receives a user's question, queries Qdrant Cloud via payload filtering, and streams the generated response.
- **`DELETE /chat/{chat_id}`**: Acts as a garbage collector, permanently wiping all PDFs, extracted images, JSON data, and Qdrant vectors linked to the `chat_id`.
- **`POST /chat/{chat_id}/refresh`**: Forces a re-index of documents for that specific chat session.

### B. The RAG Engine (`pikerag/utils/rag_utils.py`)
- **Structural Element Chunking & Parent Document Retrieval**: Instead of naive text splitting, the engine iterates through MinerU's rich JSON block-by-block. It isolates every single paragraph, table, and image into its own tiny vector to guarantee pixel-perfect search accuracy. When a vector is matched, it uses *Parent Document Retrieval* to dynamically inject the Full Page Context into the LLM prompt, giving the AI massive reasoning power without diluting the search space.
- **Precise Source Citations**: The engine embeds specific PDF page fragments (`#page=X`) into the source links of every vector chunk. When Groq cites a source, clicking it in the frontend seamlessly opens the PDF in a new tab, automatically scrolled to the exact referenced page.
- **Qdrant Cloud Integration**: Utilizes a persistent cloud vector database to ensure scalability. Uses `chat_id` as a highly optimized `keyword` payload index, allowing instantaneous filtering of documents per user session. Features a graceful fallback to a local JSON file if cloud keys are missing.
- **Strict Domain Guardrails**: The Groq LLM prompt is heavily restricted. It is engineered to output Root Cause Analysis (RCA) checklists for equipment failures and is under a strict guardrail to absolutely reject general knowledge, off-topic conversation, or standard programming questions unless specifically related to industrial PLC/SCADA systems.

### C. Frontend Features (`frontend/src/App.jsx`)
- **Glassmorphism UI**: Features a sleek, modern aesthetic with frosted glass sidebars, smooth CSS keyframe animations, and custom gradient status buttons.
- **Dynamic Processing Feed**: When a file is uploaded, an interactive grid UI spawns simulating the MinerU pipeline extraction phases.
- **Robust Markdown Rendering**: Uses `ReactMarkdown` tailored with `target="_blank"` properties so that PDF citation links natively open external tabs without disrupting the user's active chat session.
- **Theme-Matched Modals**: Includes animated, glass-themed confirmation modals (e.g. for permanently deleting chats) to prevent accidental data loss while maintaining visual consistency.

---

## ☁️ Hugging Face Microservices

To keep the application lightweight (zero heavy C++ or GPU dependencies), heavy compute operations are outsourced to Hugging Face Spaces.

### 1. MinerU Parsing API
MinerU handles the layout analysis, OCR, and multimodal extraction.
- **Endpoint**: Configured in `rag_utils.py` pointing to a dedicated HF Space.
- **Flow**: Uploaded PDFs are sent as multipart/form-data. The engine downloads the highly structured `content_list.json` along with base64 encoded diagram images.

### 2. Nomic Atlas Embeddings API
Embedding generation is outsourced to the commercial Nomic Atlas API (`nomic-embed-text-v1.5`).
- **Flow**: Extracted structural elements are sent as a single batched JSON request directly to Nomic's GPU clusters. It instantly returns highly-precise 768-dimensional float arrays, completely bypassing CPU timeouts and avoiding local sentence-transformer overhead.
