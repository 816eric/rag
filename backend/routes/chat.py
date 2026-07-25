import time

from fastapi import APIRouter

from backend import state
from backend.schemas import AskRequest, AskResponse, Message, SessionSummary

router = APIRouter(prefix="/api/sessions", tags=["chat"])


def _session_summary(s):
    return SessionSummary(id=s["id"], title=s["title"])


@router.get("", response_model=list[SessionSummary])
def list_sessions():
    sessions = state.session_manager.list_sessions()
    if not sessions:
        state.session_manager.create_session()
        sessions = state.session_manager.list_sessions()
    return [_session_summary(s) for s in sessions]


@router.post("", response_model=SessionSummary)
def create_session():
    new_id = state.session_manager.create_session()
    sessions = state.session_manager.list_sessions()
    match = next(s for s in sessions if s["id"] == new_id)
    return _session_summary(match)


@router.get("/{session_id}/messages", response_model=list[Message])
def get_messages(session_id: str):
    return [Message(**m) for m in state.session_manager.get_messages(session_id)]


@router.post("/{session_id}/ask", response_model=AskResponse)
def ask(session_id: str, body: AskRequest):
    start = time.time()
    answer = state.rag_app.answer_question(body.question, body.use_knowledge)
    elapsed = time.time() - start

    state.session_manager.append_message(session_id, "user", body.question)
    state.session_manager.append_message(session_id, "assistant", answer)

    return AskResponse(answer=answer, elapsed_seconds=round(elapsed, 2))
