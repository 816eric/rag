from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
try:
    from langchain_community.document_loaders import Docx2txtLoader
except ImportError:
    Docx2txtLoader = None
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from doc import DocumentManager
import config.config as config

import os
import gc
import shutil
import sys
import time

class RAGApp(DocumentManager):    
    def __init__(self, db_dir=config.DB_DIR, embedding_model_path=None, llm_model="deepseek-r1:1.5b"):
        super().__init__()
        self.db_dir = db_dir
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_path)
        self.llm = OllamaLLM(model=llm_model)

        if os.path.exists(self.db_dir) and os.listdir(self.db_dir):
            self.vectorstore = Chroma(persist_directory=self.db_dir, embedding_function=self.embedding_model)
        else:
            self.vectorstore = None

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever() if self.vectorstore else None
        ) if self.vectorstore else None

    def split_documents(self, documents, use_splitter=True, chunk_size=1000, chunk_overlap=100):
        """
        Split documents into chunks or return as-is.
        Args:
            documents (list): List of Document objects.
            use_splitter (bool): Whether to use the splitter.
            chunk_size (int): Chunk size for the splitter.
            chunk_overlap (int): Overlap for the splitter.
        Returns:
            list: List of Document chunks.
        """
        if not use_splitter:
            return documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_documents(documents)
    
    def batch_add_documents(self, chunks, batch_size=5000):
        if self.vectorstore:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                self.vectorstore.add_documents(batch)
        else:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                persist_directory=self.db_dir
            )

    def embed_documents(self, file_paths):
        files_list = self.add_documents(file_paths)
        all_docs = []
        excel_detected = False
        for file_path in files_list:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".txt":
                loader = TextLoader(file_path, encoding='utf8')
                documents = loader.load()
                all_docs.extend(documents)
            elif ext == ".pdf":
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                all_docs.extend(documents)
            elif ext == ".docx" and Docx2txtLoader:
                loader = Docx2txtLoader(file_path)
                documents = loader.load()
                all_docs.extend(documents)
            elif ext == ".docx":
                loader = UnstructuredFileLoader(file_path)
                documents = loader.load()
                all_docs.extend(documents)
            elif ext == ".xlsx":
                # Special handling for Excel: each row is a Document
                import pandas as pd
                from langchain_core.documents import Document
                df = pd.read_excel(file_path)
                for idx, row in df.iterrows():
                    content = " | ".join(str(cell) for cell in row.values)
                    doc = Document(page_content=content, metadata={"row": int(idx), "source": file_path})
                    all_docs.append(doc)
                excel_detected = True
            else:
                print(f"Unsupported file type: {file_path}")
                continue

        # Use splitter for non-Excel, skip for Excel
        if excel_detected:
            chunks = self.split_documents(all_docs, use_splitter=False)
        else:
            chunks = self.split_documents(all_docs, use_splitter=True, chunk_size=1000, chunk_overlap=100)

        self.batch_add_documents(chunks, batch_size=5000)
        self.update_qa_chain()
        print(f"Embedded {len(chunks)} chunks from {len(file_paths)} files.")
        return f"{len(chunks)} chunks embedded successfully."
    
    def update_qa_chain(self):
        if self.vectorstore:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever()
            )
        else:
            self.qa_chain = None

    def answer_question(self, question: str, use_knowledge: bool = True) -> str:
        print(f"Question: {question}, Use Knowledge: {use_knowledge}")
        if use_knowledge and self.qa_chain:
            result = self.qa_chain.invoke({"query": question})
        else:
            result = self.llm.invoke(question)

        if isinstance(result, dict) and "result" in result:
            return result["result"]
        return result
    
    def close_vectorstore(self):
        """Release the Chroma vectorstore so the DB files can be deleted."""
        self.vectorstore = None
        gc.collect()

    def delete_documents(self, files):
        """Delete the Chroma DB directory and release resources."""
        if self.vectorstore:
            self.vectorstore = None
        gc.collect()
        print("Vectorstore refcount:", sys.getrefcount(self.vectorstore))
        time.sleep(0.5)
        if os.path.exists(self.db_dir):
            try:
                shutil.rmtree(self.db_dir)
            except OSError as e:
                print(f"Error deleting Chroma DB directory: {e}")
                return False
        self.delete_documents_manifest(files)
        print("Chroma DB deleted and resources released.")
    
    def set_llm_model(self, model_name):
        self.llm = OllamaLLM(model=model_name)
        self.update_qa_chain()
        print(f"LLM model changed to: {model_name}")

    def get_llm_model(self):
        """
        Returns the current LLM model name.
        
        Returns:
            str: The name of the currently set LLM model.
        """
        return self.llm.model if self.llm else "No LLM model set"
