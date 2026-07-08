import os
import shutil
import tempfile
import subprocess
import gradio as gr
import json

def process_pdf(api_key, pdf_file, backend, enable_ocr, layout_model, max_pages):
    expected_key = os.environ.get("SPACE_API_KEY", "")
    if expected_key and api_key.strip() != expected_key:
        yield "❌ Authentication Failed: Invalid API Key. You are not authorized to use this Space.", "", "", "", None
        return

    if not pdf_file:
        yield "Please upload a PDF file.", "", "", None, None
        return

    # Create a temporary directory for processing
    output_dir = tempfile.mkdtemp()
    file_path = pdf_file.name
    
    # Set the backend environment variable dynamically
    env = os.environ.copy()
    env["MINERU_BACKEND"] = backend
    
    # Construct the MinerU CLI command
    command = ["mineru", "-p", file_path, "-o", output_dir]
    
    if enable_ocr:
        command.append("--is-ocr")
        
    terminal_logs = "Starting conversion...\n"
    yield terminal_logs, "", "", None, None
    
    try:
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1,
            env=env
        )
        
        for line in iter(process.stdout.readline, ''):
            terminal_logs += line
            print(line, end="", flush=True)
            # Yield empty strings for the content during processing
            yield terminal_logs, "", "", None, None
            
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            terminal_logs += f"\nError: MinerU exited with code {return_code}"
            yield terminal_logs, "", "", None, None
            return

        terminal_logs += "\nExtraction Complete! Preparing outputs...\n"
        yield terminal_logs, "", "", None, None
        
        pdf_name = os.path.splitext(os.path.basename(file_path))[0]
        result_dir = os.path.join(output_dir, pdf_name)
        
        md_file_path = None
        json_file_path = None
        
        if os.path.exists(result_dir):
            for file in os.listdir(result_dir):
                if file.endswith(".md"):
                    md_file_path = os.path.join(result_dir, file)
                elif file.endswith(".json"):
                    json_file_path = os.path.join(result_dir, file)
                    
        md_content = ""
        json_content = ""
        
        if md_file_path and os.path.exists(md_file_path):
            with open(md_file_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            terminal_logs += "\n✅ Done"
            
        if json_file_path and os.path.exists(json_file_path):
            with open(json_file_path, "r", encoding="utf-8") as f:
                # Pretty print JSON for the UI
                json_content = json.dumps(json.load(f), indent=2)
                
        if not md_content:
            terminal_logs += "\n❌ Error: Markdown file not generated."
            
        yield terminal_logs, md_content, md_content, json_content, md_file_path

    except Exception as e:
        terminal_logs += f"\nAn error occurred:\n{str(e)}"
        yield terminal_logs, "", "", None, None

def clear_inputs():
    return None, "Upload a file and start conversion.", "", "", None, None

# Define custom dark theme matching the screenshot
theme = gr.themes.Monochrome(
    primary_hue="orange",
    secondary_hue="slate",
    neutral_hue="zinc",
).set(
    body_background_fill="#1a1a1a",
    body_text_color="#e5e5e5",
    background_fill_primary="#262626",
    background_fill_secondary="#1f1f1f",
    border_color_primary="#3f3f46",
    block_background_fill="#262626",
    button_primary_background_fill="#f97316",
    button_primary_background_fill_hover="#ea580c",
    button_primary_text_color="white",
)

css = """
.gradio-container {
    max-width: 100% !important;
}
.row-container {
    align-items: stretch;
}
"""

with gr.Blocks(theme=theme, css=css) as demo:
    with gr.Row(elem_classes="row-container"):
        
        # --- LEFT COLUMN (Controls) ---
        with gr.Column(scale=2):
            api_key = gr.Textbox(
                label="API Key (Required)", 
                type="password", 
                placeholder="Enter your secret key to use this Space...",
                info="This is a private instance. Enter the valid key to proceed."
            )
            
            pdf_input = gr.File(label="Select or paste a file to upload\nPDF, image, DOCX, PPTX, or XLSX", file_types=[".pdf", ".docx", ".pptx", ".xlsx", "image"])
            
            backend = gr.Dropdown(
                choices=["hybrid-engine", "vlm-engine", "pipeline"], 
                value="pipeline", 
                label="Backend",
                info="Exclusive hybrid engine parsing, ultra-high accuracy."
            )
            
            max_pages = gr.Slider(
                minimum=1, 
                maximum=5000, 
                value=100, 
                step=1, 
                label="Max convert pages"
            )
            
            with gr.Accordion("Advanced options", open=False):
                enable_ocr = gr.Checkbox(label="Force OCR", value=False)
                layout_model = gr.Dropdown(
                    choices=["LayoutLMv3", "DocLayout-YOLO"], 
                    value="LayoutLMv3", 
                    label="Layout Model"
                )
            
            with gr.Row():
                submit_btn = gr.Button("Convert", variant="primary", scale=1)
                clear_btn = gr.Button("Clear", variant="secondary", scale=1)
                
            gr.Markdown("### Convert result")
                
            terminal_output = gr.Textbox(
                label="Waiting", 
                lines=10, 
                max_lines=15, 
                autoscroll=True, 
                interactive=False,
                value="Upload a file and start conversion."
            )
            
        # --- MIDDLE COLUMN (PDF Preview) ---
        with gr.Column(scale=3):
            # Gradio File component serves as a previewer when interactive is False for supported types
            pdf_preview = gr.File(label="doc preview", interactive=False)
            
            # Automatically update the preview when a file is uploaded
            pdf_input.change(fn=lambda file: file, inputs=[pdf_input], outputs=[pdf_preview])
            
        # --- RIGHT COLUMN (Outputs) ---
        with gr.Column(scale=4):
            with gr.Tabs():
                with gr.Tab("Markdown rendering"):
                    md_render = gr.Markdown(label="")
                with gr.Tab("Markdown text"):
                    md_text = gr.Code(language="markdown", label="")
                with gr.Tab("JSON Content List"):
                    json_text = gr.Code(language="json", label="")
            
            # Optional direct download link at the bottom of outputs
            md_download = gr.File(label="Download Source Files", visible=False)

    # Event handlers
    submit_btn.click(
        fn=process_pdf,
        inputs=[api_key, pdf_input, backend, enable_ocr, layout_model, max_pages],
        outputs=[terminal_output, md_render, md_text, json_text, md_download]
    )
    
    clear_btn.click(
        fn=clear_inputs,
        inputs=[],
        outputs=[pdf_input, terminal_output, md_render, md_text, json_text, md_download]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
