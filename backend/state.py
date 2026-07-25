import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragapp import RAGApp
from chat_history import ChatSessionManager
import config.config as config

rag_app = RAGApp(
    db_dir=config.DB_DIR,
    embedding_model_path=config.embedding_model_path,
    llm_model=config.llm_model,
)
session_manager = ChatSessionManager()


def reset_rag_app():
    global rag_app
    rag_app = RAGApp(
        db_dir=config.DB_DIR,
        embedding_model_path=config.embedding_model_path,
        llm_model=config.llm_model,
    )
    return rag_app
