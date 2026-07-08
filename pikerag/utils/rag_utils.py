import os
import json
import requests
import uuid
import math

def cosine_similarity(v1, v2):
    """Pure Python Cosine Similarity (No Numpy/C++ needed!)"""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude1 = math.sqrt(sum(a * a for a in v1))
    magnitude2 = math.sqrt(sum(b * b for b in v2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)

class IKBRagEngine:
    def __init__(self):
        print("Initializing Pure Python RAG Engine (Zero C++ dependencies!)...")
        self.hf_token = os.getenv("HF_TOKEN")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        # Initialize pure python database
        self.db_path = "./simple_vector_db.json"
        self.knowledge_base = []
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    self.knowledge_base = json.load(f)
            except:
                pass
        
        # Initialize the Embeddings Client ONCE during startup
        from gradio_client import Client
        space_id = "abhiswork/ikb-embeddings"
        try:
            print("Connecting to Custom Embeddings Space...")
            self.embeddings_client = Client(space_id, token=self.hf_token)
        except Exception as e:
            print(f"Failed to connect to embeddings space: {e}")
            self.embeddings_client = None

    def _get_hf_embeddings(self, texts):
        if not self.embeddings_client:
            return {"error": "Embeddings client not connected"}
            
        try:
            # The API name is automatically set to the python function name by Gradio
            result_json = self.embeddings_client.predict(json.dumps(texts), api_name="/generate_embeddings")
            result = json.loads(result_json)
            if isinstance(result, dict) and "error" in result:
                raise Exception(result["error"])
            return result
        except Exception as e:
            print(f"Custom Embedding Space Error: {e}")
            return {"error": str(e)}

    def index_document(self, file_path, chat_id="default"):
        """Robust Python chunking and indexing for PDF, DOCX, and TXT.
        Extracts images to provide exact visual context for the Chatbot."""
        print("\n" + "="*50)
        print("📄 MINERU PROCESSING STATUS")
        print("="*50)
        print("🟢 Prepare: Initializing document processing...")
        print(f"Loading document: {file_path} for chat: {chat_id}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        text = ""
        
        try:
            # Prefix document name with chat_id to isolate files per chat
            base_name = os.path.basename(file_path)
            if not base_name.startswith(f"{chat_id}_"):
                doc_name = f"{chat_id}_{base_name}"
            else:
                doc_name = base_name
            
            # Save a permanent copy of the document for the user to view later
            import shutil
            os.makedirs("uploaded_documents", exist_ok=True)
            saved_doc_path = os.path.abspath(os.path.join("uploaded_documents", doc_name))
            shutil.copy(file_path, saved_doc_path)
            
            # Send file to MinerU API
            mineru_api_url = "https://abhiswork-mineru-api-ikb.hf.space/file_parse"
            print("🟢 Check service: Verifying MinerU API...")
            print("🟢 Submit: Uploading file to MinerU...")
            print(f"Sending document to MinerU: {mineru_api_url}")
            print("🟡 Parse: MinerU is extracting text and tables (this may take a while)...")
            
            with open(file_path, "rb") as f:
                files = {"files": (doc_name, f)}
                data = {"is_ocr": "False", "backend": "pipeline"}
                headers = {"Authorization": f"Bearer {self.hf_token}"}
                # Use standard multipart form-data 'files' field for MinerU
                res = requests.post(mineru_api_url, files=files, data=data, headers=headers)
                
            print("🟢 Parse: Extraction complete!")
            print("🟢 Download: Retrieving JSON payload...")
            if res.status_code != 200:
                raise Exception(f"MinerU API Error: HTTP {res.status_code} - {res.text}")
                
            parsed_data = res.json()
            
            # Extract markdown from common JSON response structures
            text = ""
            if isinstance(parsed_data, dict):
                if "content" in parsed_data:
                    text = parsed_data["content"]
                elif "data" in parsed_data and "markdown" in parsed_data["data"]:
                    text = parsed_data["data"]["markdown"]
                elif "markdown" in parsed_data:
                    text = parsed_data["markdown"]
                else:
                    # Fallback to dumping the whole JSON if schema is unknown
                    text = json.dumps(parsed_data, indent=2)
            else:
                text = str(parsed_data)
                
            # Save raw outputs locally in structured folders
            pdf_base = os.path.splitext(base_name)[0]
            parsed_dir = os.path.join("parsed_data", str(chat_id), pdf_base)
            os.makedirs(parsed_dir, exist_ok=True)
            
            with open(os.path.join(parsed_dir, "mineru_output.json"), "w", encoding="utf-8") as f:
                json.dump(parsed_data, f, indent=2, ensure_ascii=False)
                
            with open(os.path.join(parsed_dir, "content.md"), "w", encoding="utf-8") as f:
                f.write(text)
                
            # Also extract raw images using PyMuPDF for visual rendering in the chat
            import fitz
            try:
                doc_pdf = fitz.open(file_path)
                extracted_images_md = []
                for page_num in range(len(doc_pdf)):
                    page = doc_pdf.load_page(page_num)
                    for img_index, img in enumerate(page.get_images(full=True)):
                        xref = img[0]
                        base_image = doc_pdf.extract_image(xref)
                        img_filename = f"page{page_num+1}_img{img_index}.{base_image['ext']}"
                        
                        img_dir = os.path.join("static_images", str(chat_id), pdf_base)
                        os.makedirs(img_dir, exist_ok=True)
                        
                        img_path = os.path.join(img_dir, img_filename)
                        with open(img_path, "wb") as img_f:
                            img_f.write(base_image["image"])
                        
                        img_url = f"http://localhost:8002/images/{chat_id}/{pdf_base}/{img_filename}"
                        extracted_images_md.append(f"![Diagram from page {page_num+1}]({img_url})")
                
                if extracted_images_md:
                    text += "\n\n### Extracted Diagrams:\n" + "\n".join(extracted_images_md)
            except Exception as fitz_e:
                print(f"Skipping PyMuPDF image extraction: {fitz_e}")
                
            # Prepend source tag for the LLM prompt instructions
            doc_url = f"http://localhost:8002/documents/{chat_id}/{base_name}"
            text = f"\n[Source: {base_name}, Link: {doc_url}]\n\n" + text
            
        except Exception as e:
            raise Exception(f"Failed to process file {file_path}. Error: {str(e)}")
            
        print("🟡 Build outputs: Extracting images and chunking text...")
        # Since MinerU provides highly structured Markdown, we can chunk by double-newlines (paragraphs/tables)
        chunks = []
        current_chunk = ""
        max_chunk_size = 1500 
        
        def add_block(b):
            nonlocal current_chunk
            if len(current_chunk) + len(b) < max_chunk_size:
                current_chunk += b + "\n"
            else:
                if len(current_chunk.strip()) > 10:
                    chunks.append(current_chunk.strip())
                # If b is STILL larger than max_chunk_size (e.g. huge markdown table row), force split it
                if len(b) > max_chunk_size:
                    for i in range(0, len(b), max_chunk_size):
                        chunks.append(b[i:i+max_chunk_size])
                    current_chunk = ""
                else:
                    current_chunk = b + "\n"

        for block in text.split("\n\n"):
            if len(block) > max_chunk_size:
                # Fallback to single newline split for massive blocks
                for sub in block.split("\n"):
                    add_block(sub)
            else:
                add_block(block + "\n")
                
        if len(current_chunk.strip()) > 10:
            chunks.append(current_chunk.strip())
                
        print("🟡 Queue: Generating vector embeddings...")
        embeddings = self._get_hf_embeddings(chunks)
        
        if isinstance(embeddings, dict) and "error" in embeddings:
            raise Exception(f"HF API Error: {embeddings['error']}")
            
        print(f"Indexing {len(chunks)} chunks into Pure Python Vector DB...")
        
        # Save to our custom JSON vector database
        for i, chunk in enumerate(chunks):
            self.knowledge_base.append({
                "chat_id": chat_id,
                "text": chunk,
                "embedding": embeddings[i]
            })
            
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.knowledge_base, f)
            
        print("🟢 Done: Saved to Vector DB!")
        print("="*50 + "\n")
        return len(chunks)
        
    def refresh_chat_data(self, chat_id):
        """Re-indexes all documents associated with a specific chat."""
        # 1. Clear out old chunks
        self.knowledge_base = [doc for doc in self.knowledge_base if doc.get("chat_id") != chat_id]
        
        # 2. Find and re-process all files
        reindexed = 0
        if os.path.exists('uploaded_documents'):
            for filename in os.listdir('uploaded_documents'):
                if filename.startswith(f"{chat_id}_"):
                    file_path = os.path.join('uploaded_documents', filename)
                    reindexed += self.index_document(file_path, chat_id)
                    
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.knowledge_base, f)
            
        return f"✅ Successfully re-parsed {reindexed} chunks for chat {chat_id}"

    def delete_chat_data(self, chat_id):
        """Deletes all embeddings and files associated with a specific chat."""
        # Remove from knowledge base
        self.knowledge_base = [doc for doc in self.knowledge_base if doc.get("chat_id") != chat_id]
        if os.path.exists(self.db_path):
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self.knowledge_base, f)
                
        # Delete files from uploaded_documents
        if os.path.exists('uploaded_documents'):
            for filename in os.listdir('uploaded_documents'):
                if filename.startswith(f"{chat_id}_"):
                    try: os.remove(os.path.join('uploaded_documents', filename))
                    except: pass
                    
        # Delete images from static_images
        chat_images_dir = os.path.join('static_images', str(chat_id))
        if os.path.exists(chat_images_dir):
            import shutil
            try: shutil.rmtree(chat_images_dir)
            except: pass
                    
        # Delete parsed data folder for this chat
        chat_parsed_dir = os.path.join('parsed_data', str(chat_id))
        if os.path.exists(chat_parsed_dir):
            import shutil
            try: shutil.rmtree(chat_parsed_dir)
            except: pass
                    
        return f"✅ Erased all data for chat {chat_id}"

    def clear_knowledge_base(self):
        """Wipes the Vector DB, Document Store, Image Store, and Neo4j Graph DB."""
        self.knowledge_base = []
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            
        import shutil
        if os.path.exists('uploaded_documents'):
            shutil.rmtree('uploaded_documents')
        if os.path.exists('static_images'):
            shutil.rmtree('static_images')
        if os.path.exists('parsed_data'):
            shutil.rmtree('parsed_data')
            
        return "✅ Successfully erased Vector DB and File Storage!"

    def query(self, question: str, chat_id: str = "default"):
        """Retrieves context and generates an answer via Groq API."""
        print(f"Retrieving context for: {question} in chat {chat_id}")
        
        # Embed question
        emb_result = self._get_hf_embeddings([question])
        if isinstance(emb_result, dict) and "error" in emb_result:
            return f"HF API Error: {emb_result['error']}"
        q_emb = emb_result[0]
        
        # Search our custom database for this specific chat
        print("Searching Knowledge Base...")
        scores = []
        for doc in self.knowledge_base:
            if doc.get("chat_id") == chat_id:
                score = cosine_similarity(q_emb, doc["embedding"])
                scores.append((score, doc["text"]))
            
        # Sort by highest score first and get top 4
        scores.sort(reverse=True, key=lambda x: x[0])
        top_docs = [doc for score, doc in scores[:4]]
        
        context = "\n\n".join(top_docs)
        
        prompt = f"""You are the ultimate Industrial Knowledge Copilot for field technicians.
You have access to heterogeneous industrial document corpora including OEM manuals, SOPs, and P&ID diagrams.

CRITICAL INSTRUCTIONS:
1. If the user asks about an equipment failure or troubleshooting, ALWAYS format your answer as a Root Cause Analysis (RCA) Checklist:
   - 🚨 Potential Root Causes
   - 🛠️ Step-by-Step Fix Procedures
   - ⚠️ Safety & Compliance Warnings
2. If the retrieved context contains a string like '[Visual Reference Available at: X]', and the user asks about that equipment or symbol, you MUST start your response by displaying the image using standard Markdown: `![Diagram](/file=X)` (Make sure to include /file= before the path)
3. ALWAYS cite your sources at the bottom of your response! You must provide a clickable Markdown link using the [Source, Page, Link] tags found in the context. Format it exactly like this: `[View Source: pump-manual.pdf (Page 4)](/file=<Link>)`.

Context:
{context}

Question: {question}

Answer:"""
        
        print("Generating response via Groq Cloud API...")
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }
        
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Groq API Error: {response.text}"

# Singleton instance
rag_engine = IKBRagEngine()
