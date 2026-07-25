import os
import shutil
import tempfile
from dataclasses import dataclass

from fastapi import APIRouter, File, UploadFile

from backend import state
from backend.schemas import (
    BrowseFolderResponse,
    DeleteDocumentsRequest,
    FolderEmbedRequest,
    StatusResponse,
)

router = APIRouter(prefix="/api/documents", tags=["documents"])


@dataclass
class _UploadedFileRef:
    """Adapts a saved-to-disk upload to the `.name` attribute shape
    doc.py's add_documents() expects (it mirrors Gradio's file object)."""
    name: str


def _doc_choices():
    return [os.path.basename(f) for f in state.rag_app.list_documents()]


@router.get("", response_model=list[str])
def list_documents():
    return _doc_choices()


@router.post("/upload", response_model=list[str])
async def upload_documents(files: list[UploadFile] = File(...)):
    tmp_dir = tempfile.mkdtemp()
    refs = []
    for f in files:
        dest = os.path.join(tmp_dir, f.filename)
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        refs.append(_UploadedFileRef(name=dest))

    state.rag_app.embed_documents(refs)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return _doc_choices()


@router.post("/browse", response_model=BrowseFolderResponse)
def browse_folder():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder_selected = filedialog.askdirectory()
    root.destroy()
    return BrowseFolderResponse(folder_path=folder_selected or "")


@router.post("/folder", response_model=StatusResponse)
def embed_folder(body: FolderEmbedRequest):
    if not body.folder_path or not os.path.isdir(body.folder_path):
        return StatusResponse(status="Please choose a valid folder first.")
    status = state.rag_app.embed_folder(body.folder_path, body.include_subfolders)
    return StatusResponse(status=status)


@router.post("/delete", response_model=list[str])
def delete_documents(body: DeleteDocumentsRequest):
    state.rag_app.delete_documents(body.names)
    return _doc_choices()
