import os
import time
import gradio as gr
from ragapp import RAGApp  # Assuming your core RAG logic is in rag.py
from doc import DocumentManager
from chat_history import ChatSessionManager
import config.config as config

rag_app = RAGApp(db_dir=config.DB_DIR,
        embedding_model_path=config.embedding_model_path,
        llm_model=config.llm_model)
session_manager = ChatSessionManager()

CSS = """
/* Semantic tokens (light = default). Palette verified against WCAG 2.2 AA
   (4.5:1 body text, 3:1 large text/UI) rather than eyeballed - see
   craft/color.md in nexu-io/open-design for the source rules. Same values
   as the Flutter frontend's theme.dart for cross-UI consistency. Accent is
   teal, not the indigo #6366f1 that reads as a generic AI-tool default. */
:root {
    --bg: #fafafa;
    --surface: #ffffff;
    --surface-raised: #f4f4f5;
    --fg: #111111;
    --muted: #5c5c5c;
    --border: rgba(0, 0, 0, 0.08);
    --hover: rgba(0, 0, 0, 0.06);
    --selected: rgba(15, 118, 110, 0.12);
    --accent: #0f766e;
    --accent-fg: #ffffff;
    --danger: #dc2626;
    --danger-fg: #ffffff;
}
body.dark {
    --bg: #0f0f0f;
    --surface: #1a1a1a;
    --surface-raised: #242424;
    --fg: #f0f0f0;
    --muted: #a3a3a3;
    --border: rgba(255, 255, 255, 0.08);
    --hover: rgba(255, 255, 255, 0.08);
    --selected: rgba(45, 212, 191, 0.16);
    --accent: #2dd4bf;
    --accent-fg: #0f0f0f;
    --danger: #f87171;
    --danger-fg: #0f0f0f;
}
body, .gradio-container {
    background: var(--bg) !important;
    color: var(--fg) !important;
}
.gradio-container * {
    border-color: var(--border) !important;
}

/* Sidebar */
.sidebar {
    background: var(--surface) !important;
}
.sidebar, .sidebar * {
    color: var(--fg) !important;
}
#new-chat-btn {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--fg) !important;
    border-radius: 12px !important;
    margin-bottom: 12px;
    box-shadow: none !important;
}
#new-chat-btn:hover {
    background: var(--hover) !important;
}
#theme-toggle-btn {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--fg) !important;
    border-radius: 999px !important;
    box-shadow: none !important;
    min-width: 40px !important;
}
#theme-toggle-btn:hover {
    background: var(--hover) !important;
}
#chat-sidebar .wrap {
    flex-direction: column !important;
    align-items: stretch !important;
    gap: 2px;
    background: transparent !important;
}
#chat-sidebar label {
    border-radius: 10px;
    padding: 10px 12px;
    margin: 0 !important;
    background: transparent !important;
    border: 1px solid transparent !important;
}
#chat-sidebar label:hover {
    background: var(--hover) !important;
}
#chat-sidebar label.selected,
#chat-sidebar label.selected span {
    background: var(--selected) !important;
    border: 1px solid transparent !important;
    color: var(--fg) !important;
    font-weight: 500;
}
#chat-sidebar input[type="radio"] {
    display: none;
}
#settings-btn {
    margin-top: auto;
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--fg) !important;
    border-radius: 12px !important;
    box-shadow: none !important;
}
#settings-btn:hover {
    background: var(--hover) !important;
}

/* Header */
#header-row {
    align-items: center;
    justify-content: space-between;
    background: transparent !important;
}
#header-row h2 {
    color: var(--fg) !important;
}
#model-dropdown {
    min-width: 200px;
}
#model-dropdown .wrap, #model-dropdown input {
    background: var(--surface-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: 999px !important;
    color: var(--fg) !important;
}

/* Chat column */
#chat-column {
    max-width: 800px;
    margin: 0 auto;
    width: 100%;
    background: transparent !important;
}
#chatbot {
    min-height: 520px;
    background: transparent !important;
    border: none !important;
}
#chatbot .bubble-wrap {
    background: transparent !important;
}
#chatbot .user-row .flex-wrap.role {
    max-width: 70%;
    margin-left: auto;
}
#chatbot .user-row .user.message,
#chatbot .user-row .message.panel-full-width {
    width: auto !important;
    max-width: none !important;
}
#chatbot .user-row .message.panel-full-width {
    background: var(--surface-raised) !important;
    border-radius: 20px !important;
    border: none !important;
    padding: 10px 16px !important;
}
#chatbot .bot-row .message.panel-full-width,
#chatbot .bot-row .bot.message {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 4px 0 !important;
    max-width: 100%;
}
#chatbot .message-content, #chatbot .prose {
    color: var(--fg) !important;
}

/* Input row */
#input-row {
    background: var(--surface-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: 28px !important;
    padding: 6px 8px !important;
    align-items: center;
}
#input-row textarea, #input-row input {
    background: transparent !important;
    border: none !important;
    color: var(--fg) !important;
    box-shadow: none !important;
}
#ask-btn {
    border-radius: 999px !important;
    background: var(--accent) !important;
    color: var(--accent-fg) !important;
    border: none !important;
    min-width: 44px !important;
}
#ask-btn:hover {
    opacity: 0.85;
}
#refer-checkbox {
    color: var(--muted) !important;
}
#time-caption {
    color: var(--muted) !important;
    font-size: 12px;
    text-align: center;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
#time-caption textarea {
    background: transparent !important;
    border: none !important;
    color: var(--muted) !important;
    text-align: center;
    box-shadow: none !important;
}

/* Settings panel */
#settings-panel {
    position: fixed;
    left: 24px;
    bottom: 80px;
    width: 420px;
    max-height: 85vh;
    overflow-y: auto;
    background: var(--surface-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    padding: 16px;
    z-index: 1000;
    color: var(--fg) !important;
}
#settings-panel, #settings-panel * {
    color: var(--fg) !important;
}
#settings-panel .form,
#settings-panel .block {
    height: auto !important;
    min-height: fit-content !important;
    overflow: visible !important;
}
#upload-btn {
    background: var(--accent) !important;
    color: var(--accent-fg) !important;
    border: none !important;
}
#embed-folder-btn {
    background: var(--accent) !important;
    color: var(--accent-fg) !important;
    border: none !important;
}
#delete-btn, #restart-btn {
    background: transparent !important;
    border: 1px solid var(--danger) !important;
    color: var(--danger) !important;
}
#delete-btn:hover, #restart-btn:hover {
    background: var(--danger) !important;
    color: var(--danger-fg) !important;
}

footer {
    display: none !important;
}
"""

THEME = gr.themes.Base(
    primary_hue="neutral",
    neutral_hue="neutral",
    radius_size="lg",
)

# Gradio toggles a `dark` class on <body> (not <html>) to switch its own
# built-in theme variables - confirmed by inspecting body.className with
# ?__theme=dark vs ?__theme=light. Setting/reading that same class directly
# means the toggle button can flip instantly with no page reload, and the
# choice persists across visits via localStorage. Defaults to dark to match
# this app's established look for anyone without a saved preference yet.
INIT_THEME_JS = """
function() {
    const stored = localStorage.getItem('rag-theme');
    const dark = stored ? stored === 'dark' : true;
    document.body.classList.toggle('dark', dark);
}
"""

TOGGLE_THEME_JS = """
function() {
    const isDark = document.body.classList.toggle('dark');
    localStorage.setItem('rag-theme', isDark ? 'dark' : 'light');
}
"""


def _session_choices(sessions):
    return [(s["title"], s["id"]) for s in sessions]


def _doc_choices():
    return [os.path.basename(f) for f in rag_app.list_documents()]


def on_app_load():
    sessions = session_manager.list_sessions()
    if not sessions:
        session_manager.create_session()
        sessions = session_manager.list_sessions()
    current_id = sessions[0]["id"]
    messages = session_manager.get_messages(current_id)
    doc_choices = _doc_choices()
    return gr.update(choices=_session_choices(sessions), value=current_id), messages, gr.update(choices=doc_choices, value=[])


def on_new_chat():
    new_id = session_manager.create_session()
    sessions = session_manager.list_sessions()
    return gr.update(choices=_session_choices(sessions), value=new_id), []


def on_select_session(session_id):
    return session_manager.get_messages(session_id)


def on_ask(question, use_knowledge, session_id, history):
    if not question or not question.strip():
        return history, question, "", gr.update()

    start = time.time()
    result = rag_app.answer_question(question, use_knowledge)
    elapsed = time.time() - start
    print(f"LLM response time: {elapsed:.2f} seconds")

    session_manager.append_message(session_id, "user", question)
    session_manager.append_message(session_id, "assistant", result)
    new_history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": result},
    ]
    sessions = session_manager.list_sessions()
    return new_history, "", f"{elapsed:.2f} seconds", gr.update(choices=_session_choices(sessions), value=session_id)


def embed_documents(files):
    print(f"Embedding files: {files}")
    rag_app.embed_documents(files)
    display_list = _doc_choices()
    print(f"Documents embedded: doc_list={display_list}")
    # value=[] - nothing pre-selected. Pre-checking every document here previously meant
    # "Delete Selected" would wipe out every OTHER document the instant you unchecked the
    # one you actually wanted to remove, since it started as the only unchecked item.
    return gr.update(choices=display_list, value=[])


def delete_documents(selected_docs):
    print(f"Deleting selected documents: {selected_docs}")
    rag_app.delete_documents(selected_docs)
    display_list = _doc_choices()
    return gr.update(choices=display_list, value=[])


def restart_app():
    import sys
    import shutil
    import gc

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
    os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)


def change_llm(model_name):
    rag_app.set_llm_model(model_name)
    gr.Info(f"LLM model changed to: {model_name}")


def browse_folder():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder_selected = filedialog.askdirectory()
    root.destroy()
    return folder_selected or ""


def embed_folder(folder_path, include_subfolders):
    if not folder_path or not os.path.isdir(folder_path):
        return gr.update(choices=_doc_choices(), value=[]), "Please choose a valid folder first."
    status = rag_app.embed_folder(folder_path, include_subfolders)
    display_list = _doc_choices()
    return gr.update(choices=display_list, value=[]), status


def toggle_settings(is_open):
    new_state = not is_open
    # Refresh the document list from disk each time the panel opens - it previously only
    # ever showed whatever this browser session had last pushed via its own clicks, so a
    # fresh tab (or one left open since before an embed) never saw newly embedded documents.
    doc_choices = _doc_choices()
    return new_state, gr.update(visible=new_state), gr.update(choices=doc_choices, value=[])


def close_settings():
    return False, gr.update(visible=False)


with gr.Blocks(theme=THEME, css=CSS, title="Local RAG Assistant", js=INIT_THEME_JS) as demo:
    with gr.Sidebar(label="Chats", position="left", elem_id="app-sidebar"):
        new_chat_btn = gr.Button("+ New Chat", elem_id="new-chat-btn", variant="primary")
        session_radio = gr.Radio(choices=[], label=None, show_label=False, elem_id="chat-sidebar")
        settings_btn = gr.Button("⚙️ Settings", elem_id="settings-btn")

    with gr.Row(elem_id="header-row"):
        gr.Markdown("## 💬 Local RAG Assistant")
        theme_toggle_btn = gr.Button("🌓", elem_id="theme-toggle-btn", scale=0, min_width=40)
        model_dropdown = gr.Dropdown(
            choices=config.MODEL_OPTIONS,
            value=rag_app.get_llm_model() if rag_app.get_llm_model() in config.MODEL_OPTIONS else config.MODEL_OPTIONS[0],
            show_label=False,
            elem_id="model-dropdown",
            scale=0,
        )
    theme_toggle_btn.click(None, inputs=[], outputs=[], js=TOGGLE_THEME_JS)

    with gr.Column(elem_id="chat-column"):
        chatbot = gr.Chatbot(type="messages", elem_id="chatbot", show_label=False)
        with gr.Row(elem_id="input-row"):
            question_box = gr.Textbox(placeholder="Message the assistant...", show_label=False, container=False, scale=6)
            refer_checkbox = gr.Checkbox(label="Docs", value=True, scale=0, elem_id="refer-checkbox")
            ask_button = gr.Button("➤", variant="primary", scale=0, elem_id="ask-btn")
        time_box = gr.Textbox(show_label=False, interactive=False, elem_id="time-caption", container=False)

    ask_button.click(
        on_ask,
        inputs=[question_box, refer_checkbox, session_radio, chatbot],
        outputs=[chatbot, question_box, time_box, session_radio],
    )
    question_box.submit(
        on_ask,
        inputs=[question_box, refer_checkbox, session_radio, chatbot],
        outputs=[chatbot, question_box, time_box, session_radio],
    )
    new_chat_btn.click(on_new_chat, inputs=[], outputs=[session_radio, chatbot])
    session_radio.change(on_select_session, inputs=[session_radio], outputs=[chatbot])
    model_dropdown.change(change_llm, inputs=model_dropdown, outputs=[])

    settings_open = gr.State(False)
    with gr.Column(visible=False, elem_id="settings-panel") as settings_panel:
        with gr.Row():
            gr.Markdown("### 📄 Documents")
            close_settings_btn = gr.Button("✕", scale=0, min_width=40)

        with gr.Row():
            file_upload = gr.File(label="Upload Files", file_types=[".txt", ".pdf", ".docx", ".xlsx", ".md", ".json"], file_count="multiple")
        upload_button = gr.Button("Embed Uploaded Files", elem_id="upload-btn")

        gr.Markdown("**Embed an entire folder**")
        with gr.Row():
            folder_path_box = gr.Textbox(label="Folder", placeholder="No folder selected", interactive=False, scale=3)
            browse_btn = gr.Button("Browse...", scale=1)
        subfolder_checkbox = gr.Checkbox(label="Include subfolders", value=False)
        embed_folder_btn = gr.Button("Embed Folder", elem_id="embed-folder-btn")
        folder_status = gr.Textbox(label="Status", interactive=False)

        doc_list = gr.CheckboxGroup(choices=_doc_choices(), label="Embedded Documents")
        with gr.Row():
            delete_button = gr.Button("Delete Selected", elem_id="delete-btn")
            restart_button = gr.Button("Restart & Clean DB", elem_id="restart-btn")

        upload_button.click(embed_documents, inputs=file_upload, outputs=doc_list)
        browse_btn.click(browse_folder, inputs=[], outputs=folder_path_box)
        embed_folder_btn.click(embed_folder, inputs=[folder_path_box, subfolder_checkbox], outputs=[doc_list, folder_status])
        delete_button.click(delete_documents, inputs=doc_list, outputs=doc_list)
        restart_button.click(lambda: restart_app(), inputs=[], outputs=[])

    settings_btn.click(toggle_settings, inputs=[settings_open], outputs=[settings_open, settings_panel, doc_list])
    close_settings_btn.click(close_settings, inputs=[], outputs=[settings_open, settings_panel])

    demo.load(on_app_load, inputs=[], outputs=[session_radio, chatbot, doc_list])

demo.launch()
