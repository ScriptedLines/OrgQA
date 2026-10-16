from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    vector_store_connected: bool

class CreateThreadRequest(BaseModel):
    user_id: Optional[str] = None

class ThreadResponse(BaseModel):
    thread_id: str
    user_id: str
    created_at: Union[str, int]
    title: Optional[str] = None

class ThreadHistoryResponse(BaseModel):
    thread_id: str
    user_id: str
    summary: str
    messages: List[Dict[str, Any]]
    updated_at: Union[str, int]

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    stream: Optional[bool] = False

class SourceCitation(BaseModel):
    source: str
    chunk_excerpt: str

class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    sources: List[SourceCitation] = Field(default_factory=list)

class IngestResponse(BaseModel):
    message: str
    total_chunks: int
    files_processed: List[str]

