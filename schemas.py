from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    vector_store_connected: bool

class CreateThreadRequest(BaseModel):
    user_id: Optional[str] = None

class UpdateThreadRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)

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

class AuthLoginRequest(BaseModel):
    email: str
    password: str

class AuthSignUpRequest(BaseModel):
    email: str
    password: str
    redirect_to: Optional[str] = None

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]

class SignUpResponse(BaseModel):
    access_token: Optional[str] = ""
    token_type: str = "bearer"
    user: Optional[Dict[str, Any]] = None
    requires_confirmation: bool = False
    email: Optional[str] = None
    message: Optional[str] = None

class ResendConfirmationRequest(BaseModel):
    email: str
    redirect_to: Optional[str] = None

class UpdateProfileRequest(BaseModel):
    full_name: str
    department: str
    role: Optional[str] = "employee"


