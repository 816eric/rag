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
from langchain_core.prompts import PromptTemplate
from doc import DocumentManager
import config.config as config

import os
import gc

# Small local models (e.g. qwen3:1.7b) default to a trained "I can't access external
# files" disclaimer for phrasing like "can you see X.json?", even when the retriever
# correctly found that file's content in the context below - the default RetrievalQA
# prompt doesn't tell the model that the context IS the file content it has access to.
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are answering questions using excerpts from the user's own documents, "
        "provided below as context. Each excerpt is labeled with its source filename "
        "as [Source: filename]. If the user asks whether you can see or access a "
        "specific file, check whether that filename appears in the context: if it "
        "does, you have access to it - confirm this and summarize its content instead "
        "of saying you can't access files. Only say you don't know if the context "
        "truly has nothing relevant.\n\n{context}\n\nQuestion: {question}\nHelpful Answer:"
    ),
)

class RAGApp(DocumentManager):
    def __init__(self, db_dir=config.DB_DIR, embedding_model_path=None, llm_model="deepseek-r1:1.5b"):
        super().__init__()
        self.db_dir = db_dir
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_path)
        self.llm = OllamaLLM(model=llm_model, base_url=config.OLLAMA_BASE_URL)

        if os.path.exists(self.db_dir) and os.listdir(self.db_dir):
            self.vectorstore = Chroma(persist_directory=self.db_dir, embedding_function=self.embedding_model)
        else:
            self.vectorstore = None

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever() if self.vectorstore else None,
            chain_type_kwargs={"prompt": QA_PROMPT},
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

    SUPPORTED_EXTENSIONS = (".txt", ".md", ".csv", ".json", ".pdf", ".docx", ".xlsx")

    def _load_and_chunk(self, files_list):
        """Load documents from disk paths and split into chunks.

        Each file is loaded independently: a file that fails to read (disconnected
        drive, corrupted file, unsupported quirk) is skipped rather than aborting
        the whole batch, since previously one bad file silently discarded every
        other file's embedding work while the manifest still claimed success.

        Returns (chunks, loaded_paths, failures) where failures is a list of
        (file_path, error_message) for files that could not be loaded.
        """
        all_docs = []
        excel_detected = False
        loaded_paths = []
        failures = []
        for file_path in files_list:
            ext = os.path.splitext(file_path)[1].lower()
            try:
                if ext == ".txt" or ext == ".md" or ext == ".csv" or ext == ".json":
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
                    # Special handling for Excel: each row of every sheet is a Document
                    import pandas as pd
                    from langchain_core.documents import Document
                    sheets = pd.read_excel(file_path, sheet_name=None)
                    for sheet_name, df in sheets.items():
                        for idx, row in df.iterrows():
                            content = " | ".join(str(cell) for cell in row.values)
                            doc = Document(
                                page_content=content,
                                metadata={"row": int(idx), "sheet": sheet_name, "source": file_path},
                            )
                            all_docs.append(doc)
                    excel_detected = True
                else:
                    print(f"Unsupported file type: {file_path}")
                    continue
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
                failures.append((file_path, str(e)))
                continue
            loaded_paths.append(file_path)

        # Use splitter for non-Excel, skip for Excel
        if excel_detected:
            chunks = self.split_documents(all_docs, use_splitter=False)
        else:
            chunks = self.split_documents(all_docs, use_splitter=True, chunk_size=1000, chunk_overlap=100)

        # Prepend the filename to each chunk's embedded text. Without this, questions like
        # "can you see X.json?" or "what files do you have?" never match anything, since the
        # source path only lived in metadata, which similarity search doesn't look at - only
        # page_content is embedded. Content questions worked, file-identity ones silently didn't.
        for chunk in chunks:
            source = chunk.metadata.get("source", "")
            if source:
                chunk.page_content = f"[Source: {os.path.basename(source)}]\n{chunk.page_content}"

        return chunks, loaded_paths, failures

    def embed_documents(self, file_paths):
        files_list = self.add_documents(file_paths)
        chunks, _loaded_paths, failures = self._load_and_chunk(files_list)
        self.batch_add_documents(chunks, batch_size=5000)
        self.update_qa_chain()
        print(f"Embedded {len(chunks)} chunks from {len(file_paths)} files.")
        status = f"{len(chunks)} chunks embedded successfully."
        if failures:
            status += f" {len(failures)} file(s) failed to load: " + ", ".join(os.path.basename(p) for p, _ in failures)
        return status

    def embed_folder(self, folder_path, include_subfolders=False):
        """Scan a local folder (optionally recursive) and embed every supported document found.
        Files are referenced in place rather than copied, since folders may be large."""
        found_paths = []
        if include_subfolders:
            for root, _dirs, filenames in os.walk(folder_path):
                for filename in filenames:
                    if os.path.splitext(filename)[1].lower() in self.SUPPORTED_EXTENSIONS:
                        found_paths.append(os.path.join(root, filename))
        else:
            for filename in os.listdir(folder_path):
                full_path = os.path.join(folder_path, filename)
                if os.path.isfile(full_path) and os.path.splitext(filename)[1].lower() in self.SUPPORTED_EXTENSIONS:
                    found_paths.append(full_path)

        if not found_paths:
            return "No supported documents found in the selected folder."

        chunks, loaded_paths, failures = self._load_and_chunk(found_paths)
        # Only register files that were actually embedded - registering everything found
        # regardless of load outcome is what previously made failed files look "embedded".
        if loaded_paths:
            self.add_external_documents(loaded_paths)
            self.batch_add_documents(chunks, batch_size=5000)
            self.update_qa_chain()
        print(f"Embedded {len(chunks)} chunks from {len(loaded_paths)}/{len(found_paths)} files in folder {folder_path}.")

        status = f"{len(chunks)} chunks embedded from {len(loaded_paths)} of {len(found_paths)} files found."
        if failures:
            status += f" Failed to load: " + ", ".join(os.path.basename(p) for p, _ in failures)
        return status

    def update_qa_chain(self):
        if self.vectorstore:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(),
                chain_type_kwargs={"prompt": QA_PROMPT},
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
        """Remove the selected documents' chunks from the vector store and the manifest.

        Previously this deleted the entire Chroma DB directory via shutil.rmtree, which
        wiped every embedded document (not just the selected ones) and, on Windows, would
        silently fail with WinError 32 because the live process still had the index files
        open - the manifest was then never updated, making the button look like it did
        nothing. Deleting matched chunks through Chroma's own API avoids touching any files
        on disk, so there's nothing to lock, and only the selected documents are removed.
        """
        if self.vectorstore:
            selected_basenames = {os.path.basename(f) for f in files}
            matching_sources = [f for f in self.list_documents() if os.path.basename(f) in selected_basenames]
            for source in matching_sources:
                result = self.vectorstore.get(where={"source": source})
                ids = result.get("ids", [])
                if ids:
                    self.vectorstore.delete(ids=ids)
                    print(f"Deleted {len(ids)} chunks for {source}")
            self.update_qa_chain()
        self.delete_documents_manifest(files)
        print("Selected documents deleted from vector store and manifest.")
    
    def set_llm_model(self, model_name):
        self.llm = OllamaLLM(model=model_name, base_url=config.OLLAMA_BASE_URL)
        self.update_qa_chain()
        print(f"LLM model changed to: {model_name}")

    def get_llm_model(self):
        """
        Returns the current LLM model name.
        
        Returns:
            str: The name of the currently set LLM model.
        """
        return self.llm.model if self.llm else "No LLM model set"
