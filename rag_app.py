import os
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms.ollama import Ollama
import gradio as gr
from sentence_transformers import SentenceTransformer

# === Paths ===
DATA_DIR = "./docs"
DB_DIR = "./chroma_db"

os.environ["HF_HUB_OFFLINE"] = "1"

# === Load and split documents ===
loaders = []
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".pdf"):
        loaders.append(PyPDFLoader(os.path.join(DATA_DIR, filename)))
    elif filename.endswith(".txt"):
        loaders.append(TextLoader(os.path.join(DATA_DIR, filename)))

documents = []
for loader in loaders:
    documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# === Local embedding ===
model = SentenceTransformer("C:/offline_models/all-MiniLM-L6-v2")
embedding_model = HuggingFaceEmbeddings(model_name="C:/offline_models/all-MiniLM-L6-v2")

# === Vector store ===
vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=DB_DIR)

# === Local DeepSeek LLM ===
llm = Ollama(model="deepseek-r1:1.5b")

# === RAG chain ===
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# === Gradio UI ===
def ask_question(query):
    return qa.run(query)

gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(label="Ask a question about your documents"),
    outputs=gr.Textbox(label="Answer"),
    title="📄 Local RAG with DeepSeek (Offline)",
    description="Ask questions based on PDFs or TXT files in ./docs folder"
).launch()
