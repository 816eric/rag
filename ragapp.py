from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
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

    def embed_documents(self, file_paths):
        files_list = self.add_documents(file_paths)
        all_docs = []
        for file_path in files_list:
            loader = TextLoader(file_path, encoding='utf8')
            documents = loader.load()
            all_docs.extend(documents)            

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(all_docs)

        if self.vectorstore:
            self.vectorstore.add_documents(chunks)
        else:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                persist_directory=self.db_dir
            )
        self.update_qa_chain()        
        print(f"Embedded {len(chunks)} chunks from {len(file_paths)} files.")
        return f"{len(documents)} documents embedded successfully."
    
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
