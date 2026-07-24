
import os
import gradio as gr
from typing import List

from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import shutil

from sentence_transformers import SentenceTransformer


class LocalRAGApp:
    def __init__(self, db_dir="./chroma_db", docs_dir="./docs", model_path=None, llm_model="deepseek-r1:1.5b"):
        self.db_dir = db_dir
        self.docs_dir = docs_dir
        self.vectorstore = None
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_path)
        self.llm = Ollama(model=llm_model)
        self.qa_chain = None
        self.load_or_create_vectorstore()

    def load_or_create_vectorstore(self):
        if os.path.exists(self.db_dir) and os.listdir(self.db_dir):
            self.vectorstore = Chroma(
                embedding_function=self.embedding_model,
                persist_directory=self.db_dir,
            )
        else:
            os.makedirs(self.db_dir, exist_ok=True)
            self.vectorstore = None
        self.update_qa_chain()

    def update_qa_chain(self):
        if self.vectorstore:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever()
            )
        else:
            self.qa_chain = None

    def get_embedded_files(self) -> List[str]:
        return os.listdir(self.docs_dir) if os.path.exists(self.docs_dir) else []

    def embed_documents(self, uploaded_files: List[str]) -> str:
        os.makedirs(self.docs_dir, exist_ok=True)
        loaders = []
        for file in uploaded_files:
            filename = os.path.basename(file.name)
            save_path = os.path.join(self.docs_dir, filename)
            print(f"Saving {filename} to {save_path}")
            shutil.copy(file.name, save_path)
            if filename.endswith(".pdf"):
                loaders.append(PyPDFLoader(save_path))
            elif filename.endswith(".txt"):
                loaders.append(TextLoader(save_path))
        documents = []
        for loader in loaders:
            documents.extend(loader.load())
        if not documents:
            return "No valid documents to embed."
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        if self.vectorstore:
            self.vectorstore.add_documents(chunks)
        else:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                persist_directory=self.db_dir
            )
        self.update_qa_chain()
        return f"{len(documents)} documents embedded successfully."

    def ask(self, question: str, use_docs: bool) -> str:
        if use_docs and self.qa_chain:
            return self.qa_chain.run(question)
        else:
            return self.llm(question)

    def launch_ui(self):
        with gr.Blocks() as demo:
            gr.Markdown("# 📄 Local RAG with DeepSeek")
            with gr.Row():
                file_box = gr.File(file_types=[".pdf", ".txt"], file_count="multiple", label="Upload new files")
                upload_btn = gr.Button("Embed Files")
            upload_status = gr.Textbox(label="Upload Status", interactive=False)

            embedded_files = gr.Textbox(
                label="Currently Embedded Files",
                value="\n".join(self.get_embedded_files()),
                lines=6,
                interactive=False
            )

            with gr.Row():
                query_input = gr.Textbox(label="Ask a question")
                use_docs_chk = gr.Checkbox(label="Refer to documents", value=True)
            answer_output = gr.Textbox(label="Answer")

            upload_btn.click(
                fn=self.embed_documents,
                inputs=[file_box],
                outputs=[upload_status]
            ).then(fn=lambda: "\n".join(self.get_embedded_files()), inputs=[], outputs=[embedded_files])

            gr.Button("Submit Question").click(
                fn=self.ask,
                inputs=[query_input, use_docs_chk],
                outputs=[answer_output]
            )

        demo.launch(share=False)


# Run the app
if __name__ == "__main__":
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    rag_app = LocalRAGApp(
        db_dir="./chroma_db",
        docs_dir="./docs",
        model_path="C:/offline_models/all-MiniLM-L6-v2",  # change this to your model path
        llm_model="deepseek-r1:1.5b"  # change this to your desired LLM model
    )
    rag_app.launch_ui()
