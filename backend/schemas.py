from typing import Optional
from pydantic import BaseModel


class SessionSummary(BaseModel):
    id: str
    title: str


class Message(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str
    use_knowledge: bool = True


class AskResponse(BaseModel):
    answer: str
    elapsed_seconds: float


class FolderEmbedRequest(BaseModel):
    folder_path: str
    include_subfolders: bool = False


class DeleteDocumentsRequest(BaseModel):
    names: list[str]


class StatusResponse(BaseModel):
    status: str


class BrowseFolderResponse(BaseModel):
    folder_path: str


class ModelsResponse(BaseModel):
    current: str
    options: list[str]


class SelectModelRequest(BaseModel):
    model_name: str
