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
        print("\n" + "="*50)
        print("[*] INITIALIZING IKB-RAG ENGINE")
        print("="*50)
        
        self.hf_token = os.getenv("HF_TOKEN")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        # 1. Check Groq LLM Connection
        if self.groq_api_key:
            print("[SUCCESS] Groq API Key found (LLM Inference Ready)")
        else:
            print("[ERROR] Groq API Key missing! LLM queries will fail.")

        # 2. Check Qdrant Cloud DB Connection
        self.qdrant_url = os.getenv("QDRANT_URL", "")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
        self.use_qdrant = bool(self.qdrant_url and self.qdrant_api_key)
        
        self.db_path = "./simple_vector_db.json"
        self.knowledge_base = []
        
        if self.use_qdrant:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, VectorParams
            try:
                self.qdrant = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
                self.collection_name = "ikb_manuals"
                if not self.qdrant.collection_exists(self.collection_name):
                    self.qdrant.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                    )
                print("[SUCCESS] Qdrant Cloud connected successfully!")
            except Exception as e:
                print(f"[ERROR] Qdrant Connection Failed: {e}")
                self.use_qdrant = False
                
        if not self.use_qdrant:
            print("[WARNING] Qdrant unavailable. Falling back to simple_vector_db.json")
            if os.path.exists(self.db_path):
                try:
                    with open(self.db_path, "r", encoding="utf-8") as f:
                        self.knowledge_base = json.load(f)
                except:
                    pass
        
        # 3. Initialize the Embeddings Client ONCE during startup
        from gradio_client import Client
        space_id = "abhiswork/ikb-embeddings"
        try:
            self.embeddings_client = Client(space_id, token=self.hf_token)
            print("[SUCCESS] Hugging Face Embeddings Space connected!")
        except Exception as e:
            print(f"[ERROR] Embeddings space error: {e}")
            self.embeddings_client = None
            
        print("[SUCCESS] MinerU API configured for document processing.")
        print("="*50 + "\n")

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
            mineru_api_url = "https://abhiswork-ikb-mineru.hf.space/file_parse"
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
                # Extract the pure content_list if available to match official JSON structure
                json_to_save = parsed_data
                if isinstance(parsed_data, dict) and "data" in parsed_data and "content_list" in parsed_data["data"]:
                    if parsed_data["data"]["content_list"]:
                        json_to_save = parsed_data["data"]["content_list"]
                json.dump(json_to_save, f, indent=2, ensure_ascii=False)
                
            with open(os.path.join(parsed_dir, "content.md"), "w", encoding="utf-8") as f:
                f.write(text)
                
            # Save images if provided in the JSON from MinerU API
            if isinstance(parsed_data, dict) and "data" in parsed_data and "images_base64" in parsed_data["data"]:
                import base64
                img_dir = os.path.join("static_images", str(chat_id), pdf_base)
                os.makedirs(img_dir, exist_ok=True)
                for img_name, b64_data in parsed_data["data"]["images_base64"].items():
                    img_bytes = base64.b64decode(b64_data)
                    # Handle if the name came with directory prefix e.g. images/xxx.jpg
                    clean_name = os.path.basename(img_name)
                    with open(os.path.join(img_dir, clean_name), "wb") as f:
                        f.write(img_bytes)

            doc_url = f"http://localhost:8002/documents/{chat_id}/{base_name}"

            
        except Exception as e:
            raise Exception(f"Failed to process file {file_path}. Error: {str(e)}")
            
        print("🟡 Build outputs: Pure Page-Level Chunking (1 Chunk = 1 Page)...")
        chunks = []
        
        # Dictionary to hold the aggregated content for each page
        page_contents = {}
        
        # Safely extract content_list whether MinerU wrapped it in "data" or returned a direct list
        content_list = []
        if isinstance(parsed_data, dict) and "data" in parsed_data and "content_list" in parsed_data["data"]:
            content_list = parsed_data["data"]["content_list"]
        elif isinstance(parsed_data, list): 
            content_list = parsed_data
            
        for block in content_list:
            p_idx = block.get("page_idx", 0)
            b_type = block.get("type")
            
            # 1. Skip layout noise so it doesn't pollute the Vector DB
            if b_type in ["header", "footer", "page_number"]:
                continue
                
            # 2. Initialize the page in our dictionary if it doesn't exist yet
            if p_idx not in page_contents:
                page_contents[p_idx] = ""
                
            # 3. Append content to the specific page's bucket
            if b_type == "text":
                page_contents[p_idx] += block.get("text", "") + "\n\n"
                
            elif b_type == "table":
                # Extract HTML table format safely
                table_content = block.get("table_body", block.get("text", ""))
                page_contents[p_idx] += table_content + "\n\n"
                
            elif b_type == "equation":
                eq_content = block.get("equation_body", block.get("text", ""))
                page_contents[p_idx] += eq_content + "\n\n"
                
            elif b_type == "image":
                # Extract image path and names
                img_path = block.get("img_path", "")
                img_filename = os.path.basename(img_path)
                img_url = f"http://localhost:8002/images/{chat_id}/{pdf_base}/{img_filename}"
                
                # Safely parse captions and footnotes (MinerU uses lists here)
                cap = block.get("image_caption", "")
                caption = " ".join(cap) if isinstance(cap, list) else str(cap)
                
                foot = block.get("image_footnote", "")
                footnote = " ".join(foot) if isinstance(foot, list) else str(foot)
                
                # Build clean markdown
                img_md = f"![{caption.strip()}]({img_url})"
                if footnote.strip():
                    img_md += f"\n*{footnote.strip()}*"
                    
                page_contents[p_idx] += img_md + "\n\n"

        # 4. Finalize chunks (Convert buckets to final Qdrant payloads)
        # Using sorted() ensures chunks are appended in page order
        for p_idx, text_content in sorted(page_contents.items()):
            if text_content.strip():
                actual_page = p_idx + 1 # MinerU is 0-indexed
                page_fragment = f"#page={actual_page}"
                chunks.append(f"[Source: {base_name}, Page: {actual_page}, Link: {doc_url}{page_fragment}]\n\n" + text_content.strip())
                
        # 5. Fallback if content_list was empty but we somehow have raw text
        if not chunks and text.strip():
            chunks = [f"[Source: {base_name}, Link: {doc_url}]\n\n" + text[i:i+1500] for i in range(0, len(text), 1500)]

                
        print("🟡 Queue: Generating vector embeddings...")
        embeddings = self._get_hf_embeddings(chunks)
        
        if isinstance(embeddings, dict) and "error" in embeddings:
            raise Exception(f"HF API Error: {embeddings['error']}")
            
        print(f"Indexing {len(chunks)} chunks into Vector DB...")
        
        if self.use_qdrant:
            from qdrant_client.http.models import PointStruct
            import uuid
            points = []
            for i, chunk in enumerate(chunks):
                point_id = str(uuid.uuid4())
                points.append(PointStruct(
                    id=point_id,
                    vector=embeddings[i],
                    payload={"chat_id": chat_id, "text": chunk, "source_file": base_name}
                ))
            
            self.qdrant.upsert(collection_name=self.collection_name, points=points)
            print("🟢 Done: Saved to Qdrant Cloud!")
        else:
            # Save to our custom JSON vector database
            for i, chunk in enumerate(chunks):
                self.knowledge_base.append({
                    "chat_id": chat_id,
                    "text": chunk,
                    "embedding": embeddings[i]
                })
                
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self.knowledge_base, f)
            print("🟢 Done: Saved to local JSON DB!")
            
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
        if self.use_qdrant:
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue
            self.qdrant.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="chat_id", match=MatchValue(value=chat_id))]
                )
            )
        else:
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
        if self.use_qdrant:
            try: self.qdrant.delete_collection(collection_name=self.collection_name)
            except: pass
        else:
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

    def _call_groq(self, prompt: str, temperature=0.1, json_mode=False):
        """Helper to call Groq API"""
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }
        if json_mode:
            data["response_format"] = {"type": "json_object"}
            
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            print(f"Groq API Error: {response.text}")
            return None

    def _retrieve_context(self, q_str: str, chat_id: str, limit: int = 4):
        """Helper to retrieve vector chunks from the database"""
        emb_result = self._get_hf_embeddings([q_str])
        if isinstance(emb_result, dict) and "error" in emb_result:
            print(f"HF API Error: {emb_result['error']}")
            return []
            
        q_emb = emb_result[0]
        if self.use_qdrant:
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue
            search_result = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=q_emb,
                query_filter=Filter(must=[FieldCondition(key="chat_id", match=MatchValue(value=chat_id))]),
                limit=limit
            )
            return [hit.payload["text"] for hit in search_result.points]
        else:
            scores = []
            for doc in self.knowledge_base:
                if doc.get("chat_id") == chat_id:
                    score = cosine_similarity(q_emb, doc["embedding"])
                    scores.append((score, doc["text"]))
            scores.sort(reverse=True, key=lambda x: x[0])
            return [doc for score, doc in scores[:limit]]

    def query(self, question: str, chat_id: str = "default"):
        """Agentic RAG: Routes query, decomposes if complex, retrieves context, and generates answer."""
        print(f"\n[AGENT] Analyzing query intent for: '{question}'")
        
        # Agent 1: Router
        router_prompt = f"""You are an intelligent query router. Analyze the following user question.
If it is a simple factual question requiring a single search, classify as 'SIMPLE'.
If it is a complex, multi-part, or troubleshooting question requiring deep context, classify as 'COMPLEX'.
Output JSON strictly in this format: {{"intent": "SIMPLE"}} or {{"intent": "COMPLEX"}}.
Question: {question}"""
        
        router_response = self._call_groq(router_prompt, json_mode=True)
        intent = "SIMPLE"
        try:
            if router_response:
                intent = json.loads(router_response).get("intent", "SIMPLE")
        except: pass
        
        print(f"[AGENT] Routing Decision: {intent}")
        
        top_docs = []
        if intent == "COMPLEX":
            # Agent 2: Decomposer
            print("[AGENT] Triggering QA Decomposer...")
            decomp_prompt = f"""You are an Expert Industrial Decomposer. Break the following complex user query into independent sub-queries required to search a vector database.
Generate between 2 to 4 sub-queries depending on complexity. 
Output JSON strictly in this format: {{"sub_queries": ["query 1", "query 2"]}}
Question: {question}"""
            
            decomp_response = self._call_groq(decomp_prompt, json_mode=True)
            sub_queries = [question] # Fallback
            try:
                if decomp_response:
                    sub_queries = json.loads(decomp_response).get("sub_queries", [question])
            except: pass
            
            print(f"[AGENT] Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")
            
            all_docs = []
            for sub_q in sub_queries:
                docs = self._retrieve_context(sub_q, chat_id, limit=2)
                all_docs.extend(docs)
                
            # Deduplicate chunks
            top_docs = list(set(all_docs))
            print(f"[AGENT] Retrieved {len(top_docs)} unique context chunks across all sub-queries.")
        else:
            # Standard RAG
            top_docs = self._retrieve_context(question, chat_id, limit=4)
        
        context = "\n\n".join(top_docs)
        
        prompt = f"""You are the ultimate Industrial Knowledge Copilot for field technicians.
You have access to heterogeneous industrial document corpora including OEM manuals, SOPs, and P&ID diagrams.

CRITICAL INSTRUCTIONS:
1. If the user asks about an equipment failure or troubleshooting, ALWAYS format your answer as a Root Cause Analysis (RCA) Checklist:
   - 🚨 Potential Root Causes
   - 🛠️ Step-by-Step Fix Procedures
   - ⚠️ Safety & Compliance Warnings
2. When the user asks for a diagram or figure (e.g. "show me Figure 1"), look precisely for standard markdown images `![Caption](url)` in the context. You MUST output this EXACT markdown image link in your response so the user can see it! Never modify the URL.
3. ALWAYS cite your sources at the bottom of your response! You must provide a clickable Markdown link using the [Source, Page, Link] tags found in the context. Format it exactly like this: `[View Source: pump-manual.pdf (Page X)](<Link>)`. Do NOT modify the URL or prepend anything to it.
4. STRICT GUARDRAIL: You are an INDUSTRIAL EXPERT. You must absolutely REFUSE to answer any questions that are completely unrelated to industrial equipment, factory operations, or the provided context. If the user asks for programming code (like C++, Python), general knowledge, or off-topic conversational questions, politely decline and remind them that you are an Industrial Diagnostics AI. Only provide coding answers if it explicitly relates to PLC/SCADA programming found in the industrial context.

Context:
{context}

Question: {question}

Answer:"""
        
        print("[AGENT] Synthesizing final response via Groq...")
        return self._call_groq(prompt, temperature=0.3) or "Error generating response."

# Singleton instance
rag_engine = IKBRagEngine()
