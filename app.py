import gradio as gr
import os
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from pikerag.utils.rag_utils import rag_engine

def query_ikb(question, history):
    """
    This function receives the user's question, runs it through
    the PIKE-RAG Vector RAG pipeline, and returns the answer.
    """
    try:
        response = rag_engine.query(question)
        return response
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

def process_document(file_path):
    """
    Handles PDF (SOPs) or DOCX/TXT uploads.
    """
    if not file_path:
        return "No file uploaded."
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # Chunk and index into Vector DB
    try:
        doc_name = os.path.basename(file_path)
        num_chunks = rag_engine.index_document(file_path)
        
        msg = f"✅ Document '{doc_name}' processed!\n\n- {num_chunks} text chunks indexed into Pure Python DB."
        return msg
    except Exception as e:
        return f"❌ Error processing document: {str(e)}"

# -------------------------------------------------------------------
# GRADIO UI DEFINITION
# -------------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Base()) as app:
    gr.Markdown("# 🏭 Industrial Knowledge Brain (IKB)")
    gr.Markdown("Powered by PIKE-RAG, Neo4j, and Hugging Face Open-Source Models.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Document Ingestion")
            file_upload = gr.File(label="Upload SOPs, Manuals, or P&ID Diagrams")
            ingest_btn = gr.Button("Process & Index Document", variant="primary")
            ingest_status = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("### Knowledge Base Library")
            kb_viewer = gr.File(label="Stored Industrial Documents", interactive=False, file_count="multiple")
            refresh_btn = gr.Button("Refresh Library")
            
            def list_kb_files():
                import glob, os
                if not os.path.exists("uploaded_documents"):
                    return []
                return glob.glob("uploaded_documents/*")
                
            refresh_btn.click(fn=list_kb_files, outputs=[kb_viewer])
            
            ingest_btn.click(fn=process_document, inputs=[file_upload], outputs=[ingest_status]).then(
                fn=list_kb_files, outputs=[kb_viewer]
            )
            
            # --- ERASE KNOWLEDGE BASE UI ---
            gr.Markdown("---")
            erase_btn = gr.Button("🗑️ Erase Entire Knowledge Base", variant="stop")
            with gr.Group(visible=False) as confirm_group:
                gr.Markdown("⚠️ **Are you absolutely sure? This will permanently wipe your Vector DB, File Storage, and Neo4j Graph DB!**")
                with gr.Row():
                    confirm_yes = gr.Button("Yes, Erase Everything", variant="primary")
                    confirm_no = gr.Button("Cancel")

            def clear_kb_action():
                status_msg = rag_engine.clear_knowledge_base()
                return status_msg, gr.update(visible=False)

            erase_btn.click(lambda: gr.update(visible=True), None, confirm_group)
            confirm_no.click(lambda: gr.update(visible=False), None, confirm_group)
            confirm_yes.click(fn=clear_kb_action, outputs=[ingest_status, confirm_group]).then(fn=list_kb_files, outputs=[kb_viewer])
            
        with gr.Column(scale=2):
            gr.Markdown("### Expert Copilot Chat")
            chatbot = gr.ChatInterface(
                fn=query_ikb,
                chatbot=gr.Chatbot(height=400),
                fill_height=True
            )

if __name__ == "__main__":
    print("Starting IKB Platform...")
    import os
    app.launch(share=False, allowed_paths=[os.path.abspath(".")])
