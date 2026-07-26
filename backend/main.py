from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import chat, documents, models, system, voice

app = FastAPI(title="Local RAG Assistant API")

# Flutter desktop app runs as a separate process hitting this API over localhost -
# CORS wide open is fine here since this only ever binds to 127.0.0.1 for local use.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(models.router)
app.include_router(system.router)
app.include_router(voice.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
