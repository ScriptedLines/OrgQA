import asyncio
import os
import time
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from config import APP_START_TIME, lifespan, logger, supabase_client
from routers import auth, chat, ingest
from schemas import HealthResponse

app = FastAPI(
    title="Company Policy Q&A API",
    description="Production-ready RAG backend for enterprise policy ingestion, semantic search, and contextual chat.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error processing request to {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please try again later."},
    )

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(ingest.router)

@app.get("/", include_in_schema=False)
async def serve_index():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return JSONResponse(status_code=404, content={"detail": "Frontend not found"})

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    db_ok = False
    if supabase_client is not None:
        try:
            def _ping():
                return supabase_client.table("conversations").select("thread_id").limit(1).execute()
            await asyncio.to_thread(_ping)
            db_ok = True
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")

    return HealthResponse(
        status="healthy" if db_ok else "degraded",
        uptime_seconds=round(time.time() - APP_START_TIME, 2),
        vector_store_connected=db_ok,
    )
