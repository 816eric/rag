from fastapi import APIRouter

from backend import state
from backend.schemas import ModelsResponse, SelectModelRequest, StatusResponse
import config.config as config

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=ModelsResponse)
def get_models():
    current = state.rag_app.get_llm_model()
    if current not in config.MODEL_OPTIONS:
        current = config.MODEL_OPTIONS[0]
    return ModelsResponse(current=current, options=config.MODEL_OPTIONS)


@router.post("", response_model=StatusResponse)
def select_model(body: SelectModelRequest):
    state.rag_app.set_llm_model(body.model_name)
    return StatusResponse(status=f"LLM model changed to: {body.model_name}")
