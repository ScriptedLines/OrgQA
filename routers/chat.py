import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.messages import AIMessage, HumanMessage, messages_to_dict
from langchain_core.output_parsers import StrOutputParser
from config import llm, logger, reranker, supabase_client, vector_store
from dependencies import AuthenticatedUser, get_current_user
from prompt import qa_prompt, rag_query_generator_prompt
from schemas import (
    ChatRequest,
    ChatResponse,
    CreateThreadRequest,
    SourceCitation,
    ThreadHistoryResponse,
    ThreadResponse,
    UpdateThreadRequest,
)
from services import _summarize_thread

router = APIRouter(tags=["Conversations & Chat"])

@router.post("/threads", response_model=ThreadResponse, status_code=status.HTTP_201_CREATED)
async def create_thread(
    req: Optional[CreateThreadRequest] = None,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    thread_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    user_id = current_user.id

    def _create():
        return (
            supabase_client.table("conversations")
            .upsert(
                {
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "last_n_messages": [],
                    "all_messages": [],
                    "summary": "",
                    "updated_at": now,
                    "created_at": now,
                }
            )
            .execute()
        )

    await asyncio.to_thread(_create)
    return ThreadResponse(thread_id=thread_id, user_id=user_id, created_at=now)

@router.get("/threads", response_model=List[ThreadResponse])
async def list_threads(current_user: AuthenticatedUser = Depends(get_current_user)):
    def _fetch():
        return (
            supabase_client.table("conversations")
            .select("thread_id, user_id, created_at, all_messages, summary")
            .eq("user_id", current_user.id)
            .order("created_at", desc=True)
            .execute()
        )

    res = await asyncio.to_thread(_fetch)

    def _extract_title(row):
        summary = (row.get("summary") or "").strip()
        if summary and len(summary) > 3:
            return summary.split("\n")[0][:36] + ("..." if len(summary) > 36 else "")
        msgs = row.get("all_messages") or []
        for msg in msgs:
            msg_type = msg.get("type") or (msg.get("data") or {}).get("type")
            if msg_type in ("human", "user"):
                content = (msg.get("content") or (msg.get("data") or {}).get("content") or "").strip()
                if content:
                    return content.split("\n")[0][:36] + ("..." if len(content) > 36 else "")
        return f"Query {row['thread_id'][:6]}"

    return [
        ThreadResponse(
            thread_id=row["thread_id"],
            user_id=row["user_id"],
            created_at=row.get("created_at") or "",
            title=_extract_title(row),
        )
        for row in (res.data or [])
    ]

@router.get("/threads/{thread_id}", response_model=ThreadHistoryResponse)
async def get_thread_history(
    thread_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    def _fetch():
        return (
            supabase_client.table("conversations")
            .select("*")
            .eq("thread_id", thread_id)
            .eq("user_id", current_user.id)
            .execute()
        )

    res = await asyncio.to_thread(_fetch)
    if not res.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread '{thread_id}' not found.",
        )

    record = res.data[0]
    return ThreadHistoryResponse(
        thread_id=record["thread_id"],
        user_id=record["user_id"],
        summary=record.get("summary", ""),
        messages=record.get("all_messages", []),
        updated_at=record.get("updated_at") or "",
    )

@router.delete("/threads/{thread_id}", status_code=status.HTTP_200_OK)
async def delete_thread(
    thread_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    def _delete():
        return (
            supabase_client.table("conversations")
            .delete()
            .eq("thread_id", thread_id)
            .eq("user_id", current_user.id)
            .execute()
        )

    await asyncio.to_thread(_delete)
    return {"message": "Thread deleted", "thread_id": thread_id}

@router.patch("/threads/{thread_id}", response_model=ThreadResponse)
async def update_thread_title(
    thread_id: str,
    req: UpdateThreadRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    now = datetime.now(timezone.utc).isoformat()

    def _update():
        return (
            supabase_client.table("conversations")
            .update({"summary": req.title, "updated_at": now})
            .eq("thread_id", thread_id)
            .eq("user_id", current_user.id)
            .execute()
        )

    res = await asyncio.to_thread(_update)
    if not res.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread '{thread_id}' not found.",
        )
    return ThreadResponse(
        thread_id=thread_id,
        user_id=current_user.id,
        created_at=res.data[0].get("created_at") or now,
        title=req.title,
    )

async def _execute_chat_pipeline(
    thread_id: str,
    query: str,
    current_user: AuthenticatedUser,
    queue: Optional[asyncio.Queue] = None,
) -> ChatResponse:
    user_id = current_user.id
    try:
        summary, last_n_msg, all_msg = await _summarize_thread(user_id, thread_id)

        rag_query_generator_chain = rag_query_generator_prompt | llm | StrOutputParser()
        optimized_query = await rag_query_generator_chain.ainvoke(
            {"summary": summary, "messages": last_n_msg, "query": query},
            config={
                "run_name": "RAG Query Generation",
                "tags": ["rag", "query_expansion"],
                "metadata": {"user_id": user_id, "thread_id": thread_id},
            },
        )

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=vector_store.as_retriever(
                search_kwargs={
                    "k": 50,
                    "score_threshold": 0.30,
                    "filter": {"status": "active"},
                }
            ),
        )
        relevant_chunks = await compression_retriever.ainvoke(
            optimized_query,
            config={
                "run_name": "RAG Document Retrieval",
                "tags": ["rag", "retrieval", "rerank"],
                "metadata": {"user_id": user_id, "thread_id": thread_id},
            },
        )

        if not relevant_chunks:
            fallback_ans = "I don't have enough information in the available policy documents to answer that. Please contact HR or IT for assistance."
            if queue is not None:
                await queue.put({"type": "token", "content": fallback_ans})
                await queue.put({"type": "sources", "sources": []})
                await queue.put({"type": "done", "thread_id": thread_id})
                await queue.put(None)

            new_msgs = [HumanMessage(content=query), AIMessage(content=fallback_ans)]
            last_n_msg.extend(new_msgs)
            all_msg.extend(new_msgs)
            now = datetime.now(timezone.utc).isoformat()

            def _save_fallback():
                return (
                    supabase_client.table("conversations")
                    .upsert(
                        {
                            "user_id": user_id,
                            "thread_id": thread_id,
                            "last_n_messages": messages_to_dict(last_n_msg),
                            "all_messages": messages_to_dict(all_msg),
                            "summary": summary,
                            "updated_at": now,
                        }
                    )
                    .execute()
                )

            await asyncio.to_thread(_save_fallback)
            return ChatResponse(answer=fallback_ans, thread_id=thread_id, sources=[])

        context = "\n\n".join(
            f"[{idx + 1}] Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for idx, doc in enumerate(relevant_chunks)
        )

        parts = []
        if current_user.full_name:
            parts.append(f"Name: {current_user.full_name}")
        if current_user.role:
            parts.append(f"Role: {current_user.role}")
        if current_user.department:
            parts.append(f"Department: {current_user.department}")
        if current_user.office_location:
            parts.append(f"Location: {current_user.office_location}")
        if current_user.employment_type:
            parts.append(f"Employment Type: {current_user.employment_type}")
        user_profile = ", ".join(parts) if parts else "Not specified"

        citations = [
            SourceCitation(
                source=doc.metadata.get("source", "Unknown"),
                chunk_excerpt=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            )
            for doc in relevant_chunks[:5]
        ]

        qa_chain = qa_prompt | llm | StrOutputParser()

        if queue is not None:
            full_chunks = []
            async for chunk in qa_chain.astream(
                {
                    "query": query,
                    "summary": summary,
                    "recent_history": last_n_msg,
                    "context": context,
                    "user_profile": user_profile,
                },
                config={
                    "run_name": "RAG Chat Stream",
                    "tags": ["rag", "chat_generation", "streaming"],
                    "metadata": {
                        "user_id": user_id,
                        "thread_id": thread_id,
                        "user_role": current_user.role,
                        "department": current_user.department,
                    },
                },
            ):
                text_chunk = str(chunk)
                if text_chunk:
                    full_chunks.append(text_chunk)
                    await queue.put({"type": "token", "content": text_chunk})

            full_ans = "".join(full_chunks)
            await queue.put({"type": "sources", "sources": [c.dict() for c in citations]})
            await queue.put({"type": "done", "thread_id": thread_id})
            await queue.put(None)
        else:
            full_ans = await qa_chain.ainvoke(
                {
                    "query": query,
                    "summary": summary,
                    "recent_history": last_n_msg,
                    "context": context,
                    "user_profile": user_profile,
                },
                config={
                    "run_name": "RAG Chat Generation",
                    "tags": ["rag", "chat_generation"],
                    "metadata": {
                        "user_id": user_id,
                        "thread_id": thread_id,
                        "user_role": current_user.role,
                        "department": current_user.department,
                    },
                },
            )

        new_msgs = [HumanMessage(content=query), AIMessage(content=full_ans)]
        last_n_msg.extend(new_msgs)
        all_msg.extend(new_msgs)

        now = datetime.now(timezone.utc).isoformat()
        def _save():
            return (
                supabase_client.table("conversations")
                .upsert(
                    {
                        "user_id": user_id,
                        "thread_id": thread_id,
                        "last_n_messages": messages_to_dict(last_n_msg),
                        "all_messages": messages_to_dict(all_msg),
                        "summary": summary,
                        "updated_at": now,
                    }
                )
                .execute()
            )

        await asyncio.to_thread(_save)
        return ChatResponse(answer=full_ans, thread_id=thread_id, sources=citations)
    except Exception as e:
        if queue is not None:
            await queue.put({"type": "error", "detail": str(e)})
            await queue.put(None)
            logger.error(f"Error executing streaming chat pipeline for thread '{thread_id}': {e}")
            return ChatResponse(answer=f"Error: {e}", thread_id=thread_id, sources=[])
        logger.exception(f"Error executing chat pipeline for thread '{thread_id}': {e}")
        raise

async def _stream_chat_generator(
    thread_id: str,
    req: ChatRequest,
    current_user: AuthenticatedUser,
):
    queue = asyncio.Queue()
    worker_task = asyncio.create_task(
        _execute_chat_pipeline(thread_id, req.query, current_user, queue)
    )

    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"
            if item.get("type") in ("done", "error"):
                break
    except (asyncio.CancelledError, GeneratorExit):
        logger.info(f"Client disconnected from stream for thread '{thread_id}'. Generation and storage continue in background.")
    except Exception as e:
        logger.exception(f"Error in stream generator for thread '{thread_id}': {e}")

@router.post("/threads/{thread_id}/stream")
@router.post("/chat/{thread_id}/stream")
async def stream_chat_endpoint(
    thread_id: str,
    req: ChatRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    await _summarize_thread(current_user.id, thread_id)
    return StreamingResponse(
        _stream_chat_generator(thread_id, req, current_user),
        media_type="text/event-stream",
    )

@router.post("/threads/{thread_id}/chat", responses={200: {"model": ChatResponse}})
@router.post("/chat/{thread_id}", responses={200: {"model": ChatResponse}})
async def chat_endpoint(
    thread_id: str,
    req: ChatRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    await _summarize_thread(current_user.id, thread_id)
    if req.stream:
        return StreamingResponse(
            _stream_chat_generator(thread_id, req, current_user),
            media_type="text/event-stream",
        )

    worker_task = asyncio.create_task(
        _execute_chat_pipeline(thread_id, req.query, current_user, None)
    )
    return await asyncio.shield(worker_task)
