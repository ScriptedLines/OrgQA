import logging
import os
import time
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from supabase import Client, create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("company_qa.api")

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY") or os.environ.get("LANGSMITH_API_KEY")
LANGCHAIN_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "company-qa")

if os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true" or LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if LANGCHAIN_API_KEY:
        os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

missing_vars = [
    var for var, val in [
        ("SUPABASE_URL", SUPABASE_URL),
        ("SUPABASE_SERVICE_KEY", SUPABASE_KEY),
        ("NVIDIA_API_KEY", NVIDIA_API_KEY),
        ("GEMINI_API_KEY / GOOGLE_API_KEY", GEMINI_API_KEY),
    ] if not val
]
if missing_vars:
    error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
    logger.critical(error_msg)
    raise RuntimeError(error_msg)

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024
MAX_TOKEN_THRESHOLD = 6000
SUMMARY_TRIGGER_TOKEN_COUNT = 3000

ALLOWED_EXTENSIONS = {
    ".pdf", ".csv", ".tsv", ".xlsx", ".xls", ".docx", ".doc", ".md", ".markdown"
}

APP_START_TIME = time.time()

supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
auth_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
_primary_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    api_key=GEMINI_API_KEY,
    max_retries=2,
)
if GEMINI_MODEL != "gemini-2.5-flash-lite":
    _backup_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        max_retries=2,
    )
    llm = _primary_llm.with_fallbacks([_backup_llm])
else:
    llm = _primary_llm
embeddings = NVIDIAEmbeddings(
    model="nvidia/nemotron-3-embed-1b",
    api_key=NVIDIA_API_KEY,
)
reranker = NVIDIARerank(
    model="nvidia/llama-nemotron-rerank-vl-1b-v2",
    top_n=5,
    api_key=NVIDIA_API_KEY,
)
vector_store = SupabaseVectorStore(
    client=supabase_client,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

