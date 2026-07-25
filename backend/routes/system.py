import os
import shutil
import time

from fastapi import APIRouter

from backend import state
from backend.schemas import StatusResponse
import config.config as config

router = APIRouter(prefix="/api/system", tags=["system"])


@router.post("/restart", response_model=StatusResponse)
def restart():
    """Wipe the Chroma DB and re-create the RAG app in-place.

    Same Windows file-lock caveat as the old delete-everything path: the previous
    Chroma client may still hold the index files open briefly after being
    dereferenced, so rmtree can fail with WinError 32. Reported back rather than
    left to crash the server, matching the original app's behavior.
    """
    state.rag_app.close_vectorstore()
    time.sleep(0.5)
    if os.path.exists(config.DB_DIR):
        try:
            shutil.rmtree(config.DB_DIR)
        except OSError as e:
            return StatusResponse(status=f"Error deleting Chroma DB directory: {e}")
    state.reset_rag_app()
    return StatusResponse(status="Chroma DB cleaned and RAG app reset.")
