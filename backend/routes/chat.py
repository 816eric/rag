import json
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

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


@router.post("/{session_id}/ask_stream")
def ask_stream(session_id: str, body: AskRequest):
    """Server-Sent Events stream of response text chunks, for real-time display
    (voice mode, live-typing chat). Each event is `data: {"delta": "..."}\\n\\n`;
    the final event is `data: {"done": true, "elapsed_seconds": ...}\\n\\n`.
    Message persistence happens here, same as the non-streaming /ask, once the
    full answer has been assembled from the chunks.
    """

    def event_generator():
        start = time.time()
        chunks = []
        for chunk in state.rag_app.answer_question_stream(body.question, body.use_knowledge):
            chunks.append(chunk)
            yield f"data: {json.dumps({'delta': chunk})}\n\n"
        elapsed = time.time() - start
        answer = "".join(chunks)

        state.session_manager.append_message(session_id, "user", body.question)
        state.session_manager.append_message(session_id, "assistant", answer)

        yield f"data: {json.dumps({'done': True, 'elapsed_seconds': round(elapsed, 2)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
