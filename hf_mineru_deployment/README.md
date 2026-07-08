---
title: MinerU API (PIKE-RAG)
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# MinerU API Deployment for PIKE-RAG

This Space hosts the **MinerU** FastAPI document parsing engine for the PIKE-RAG application.

## Configuration

- **Backend:** Currently configured to use the `pipeline` backend, which is optimized for CPU-only instances. 
- **Storage:** Configured to use `/data` for `HF_HOME` to cache downloaded models. If you deploy this Space on Hugging Face, make sure to attach a Persistent Storage Volume to `/data` so the models do not re-download on every reboot.
- **GPU Usage:** If you upgrade this space to a GPU (e.g., T4 or A10G), you can edit the `Dockerfile` to change `MINERU_BACKEND` to `"hybrid"` or `"vlm"` for maximum layout recognition accuracy.

## Endpoints

Once deployed, the following endpoints are available to your PIKE-RAG application:

- `POST /file_parse` - Synchronous document parsing (best for small documents).
- `POST /tasks` - Asynchronous parsing task submission (best for large PDFs).

## How to use in PIKE-RAG
Update your `rag_utils.py` to point to the Space URL:
`API_URL = "https://<your-username>-<space-name>.hf.space/file_parse"`
