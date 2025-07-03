import os
import gradio as gr
from ragapp import RAGApp  # Assuming your core RAG logic is in rag.py
from doc import DocumentManager
import config.config as config
import sys
import shutil
import gc
import time

rag_app = RAGApp(db_dir=config.DB_DIR,
        embedding_model_path=config.embedding_model_path,
        llm_model=config.llm_model)

def answer_question(question, use_knowledge=True):
    import time
    start = time.time()
    result = rag_app.answer_question(question, use_knowledge)
    elapsed = time.time() - start
    print(f"LLM response time: {elapsed:.2f} seconds")
    return result, f"{elapsed:.2f} seconds"

def embed_documents(files):
    print(f"Embedding files: {files}")
    status = rag_app.embed_documents(files)
    doc_list = rag_app.list_documents()
    display_list = [os.path.basename(f) for f in doc_list]
    print(f"Documents embedded: doc_list={display_list}")
    return {"choices": display_list, "value": display_list}

def list_documents():
    return rag_app.list_documents()

def delete_documents(selected_docs):
    print(f"Deleting selected documents: {selected_docs}")
    rag_app.delete_documents(selected_docs)
    doc_list = rag_app.list_documents()
    display_list = [os.path.basename(f) for f in doc_list]
    return {"choices": display_list, "value": display_list}

def restart_app():
    import sys
    import os
    import shutil
    import gc
    import time

    global rag_app
    rag_app = None
    gc.collect()
    time.sleep(0.5)
    db_dir = config.DB_DIR
    if os.path.exists(db_dir):
        try:
            shutil.rmtree(db_dir)
            print("Chroma DB directory deleted.")
        except Exception as e:
            print(f"Error deleting Chroma DB directory: {e}")
    print("Restarting app...")
    print("Python executable:", sys.executable)
    print("Args:", [sys.executable] + sys.argv)
    # Use execve for Windows, pass the current environment
    os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)

def change_llm(model_name):
    rag_app.set_llm_model(model_name)
    return f"LLM model changed to: {model_name}"

with gr.Blocks() as demo:
    gr.Markdown("## 💬 Local RAG Assistant")
    with gr.Row():
        question_box = gr.Textbox(placeholder="Type your question here...", show_label=False)
        refer_checkbox = gr.Checkbox(label="Refer to documents", value=True)
        ask_button = gr.Button("Ask")
    answer_box = gr.Textbox(label="Answer", interactive=False)
    time_box = gr.Textbox(label="LLM Response Time", interactive=False)

    ask_button.click(answer_question, inputs=[question_box, refer_checkbox], outputs=[answer_box, time_box])
    question_box.submit(answer_question, inputs=[question_box, refer_checkbox], outputs=[answer_box, time_box])

    gr.Markdown("---")
    with gr.Row():
        file_upload = gr.File(label="Upload Files", file_types=[".txt", ".pdf", ".docx", ".xlsx"], file_count="multiple")
        upload_button = gr.Button("Embed Documents")
    # Set initial choices to current docs
    doc_list = gr.CheckboxGroup(
        choices=[os.path.basename(f) for f in rag_app.list_documents()],
        label="Embedded Documents"
    )
    delete_button = gr.Button("Delete Selected")
    restart_button = gr.Button("Restart & Clean DB")

    # Only return value (list of file names) to update the CheckboxGroup
    upload_button.click(embed_documents, inputs=file_upload, outputs=doc_list)
    delete_button.click(delete_documents, inputs=doc_list, outputs=doc_list)
    restart_button.click(lambda: restart_app(), inputs=[], outputs=[])

    gr.Markdown("---")
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=config.MODEL_OPTIONS,
            value=rag_app.get_llm_model() if rag_app.get_llm_model() in config.MODEL_OPTIONS else config.MODEL_OPTIONS[0],
            label="Select LLM Model"
        )
        model_status = gr.Textbox(label="Model Status", interactive=False)

    model_dropdown.change(change_llm, inputs=model_dropdown, outputs=model_status)

demo.launch()
