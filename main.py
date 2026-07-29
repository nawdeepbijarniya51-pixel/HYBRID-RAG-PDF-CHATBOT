"""
FastAPI backend for the PDF RAG chatbot — async job-based version.

Endpoints:
  GET    /health                 -> quick check that required API keys are set
  GET    /config                 -> real model configuration (embedding/LLM/rerank models in use)
  POST   /upload                 -> start processing a PDF, returns {job_id} immediately
  GET    /upload/status/{job_id} -> poll REAL ingestion progress (parsing/chunking/embedding/storing/done)
  POST   /chat                   -> start answering a question, returns {job_id} immediately
  GET    /chat/status/{job_id}   -> poll REAL pipeline progress (routing/branching/retrieving/reranking/answering/done)
  GET    /sessions/{id}/history  -> fetch chat history for a session
  DELETE /sessions/{id}          -> drop a session's in-memory state

Design notes:
- /upload and /chat both hand their real work off to a background thread and
  return a job_id immediately, instead of blocking the HTTP request. The
  frontend polls /upload/status or /chat/status to show REAL stage/progress
  instead of a fake timed animation.
- JOBS / SESSIONS / RETRIEVER_CACHE are plain in-memory dicts — fine for a
  single-process student project; won't survive a restart and won't work
  correctly across multiple worker processes.
"""

import os
import shutil
import tempfile
import threading
import traceback
import uuid
from typing import Dict, List

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage

import rag_pipeline as rag

app = FastAPI(title="PDF RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Kept outside the 'files' folder so uvicorn --reload's file-watcher never
# sees new uploads land mid-request and restart the whole process.
UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

RETRIEVER_CACHE: Dict[str, "rag.RetrieverBundle"] = {}   # pdf_hash -> RetrieverBundle
SESSIONS: Dict[str, dict] = {}                            # session_id -> {"pdf_hash","filename","chat_history"}
JOBS: Dict[str, dict] = {}                                # job_id -> status dict (see _new_job)


def _new_job(job_type: str) -> str:
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "job_id": job_id,
        "type": job_type,
        "stage": "queued",
        "progress": 0,
        "message": "Queued",
        "result": None,
        "error": None,
    }
    return job_id


def _update_job(job_id: str, **kwargs):
    if job_id in JOBS:
        JOBS[job_id].update(kwargs)


class ChatRequest(BaseModel):
    session_id: str
    message: str


class HistoryMessage(BaseModel):
    role: str
    content: str


@app.get("/health")
def health():
    missing = rag.check_env_vars()
    return {
        "status": "ok" if not missing else "missing_env_vars",
        "missing_env_vars": missing,
    }


@app.get("/config")
def get_config():
    """Real model identifiers actually in use, for the UI's Model configuration panel."""
    return rag.get_model_config()


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    missing = rag.check_env_vars()
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Server missing required env vars: {', '.join(missing)}",
        )

    # Save synchronously (UploadFile's underlying file may not survive into
    # the background thread), then hand real processing off to a worker.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    job_id = _new_job("upload")
    filename = file.filename

    def worker():
        try:
            content_hash = rag.pdf_hash(tmp_path)
            collection_name = f"rag_{content_hash}"

            stored_path = os.path.join(UPLOAD_DIR, f"{content_hash}.pdf")
            if not os.path.exists(stored_path):
                shutil.move(tmp_path, stored_path)
            elif os.path.exists(tmp_path):
                os.remove(tmp_path)

            def on_progress(stage, pct, msg):
                # rag_pipeline reports its own internal "done" once the vector
                # store is ready, but the job as a whole isn't done yet — the
                # session still needs to be created below. Letting "done"
                # through here lets a poller grab the job right as it reports
                # done, before `result` is set, resulting in an undefined
                # session_id on the frontend. Remap it to a non-terminal stage
                # so only the real final _update_job() call (after session
                # creation) can mark the job done.
                if stage == "done":
                    stage, pct = "storing", min(pct, 99)
                _update_job(job_id, stage=stage, progress=pct, message=msg)

            vector_store, num_pages, num_chunks = rag.get_or_create_vector_store(
                stored_path, collection_name, on_progress=on_progress
            )

            if content_hash not in RETRIEVER_CACHE:
                RETRIEVER_CACHE[content_hash] = rag.build_retriever(vector_store)

            session_id = str(uuid.uuid4())
            SESSIONS[session_id] = {
                "pdf_hash": content_hash,
                "filename": filename,
                "chat_history": [],
            }

            _update_job(
                job_id,
                stage="done",
                progress=100,
                message="Index ready",
                result={
                    "session_id": session_id,
                    "pdf_hash": content_hash,
                    "filename": filename,
                    "pages": num_pages,
                    "chunks": num_chunks,
                },
            )
        except Exception as exc:
            traceback.print_exc()  # surface the real error in the server console, not just the job dict
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            _update_job(job_id, stage="error", progress=0, message="Failed", error=str(exc))

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}


@app.get("/upload/status/{job_id}")
def upload_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None or job["type"] != "upload":
        raise HTTPException(status_code=404, detail="Unknown upload job_id.")
    return job


@app.post("/chat")
def chat(req: ChatRequest):
    session = SESSIONS.get(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id. Upload a PDF first.")
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty.")

    retriever_bundle = RETRIEVER_CACHE.get(session["pdf_hash"])
    if retriever_bundle is None:
        raise HTTPException(
            status_code=500,
            detail="Retriever not found for this session's document. Try re-uploading.",
        )

    job_id = _new_job("chat")

    def worker():
        try:
            def on_stage(stage, pct, msg):
                # Same reasoning as /upload's on_progress: answer_query's own
                # internal "done" fires before chat_history is appended and
                # before the real result (answer/sources) is set below, so
                # remap it to a non-terminal stage to avoid a poller grabbing
                # an empty result mid-race.
                if stage == "done":
                    stage, pct = "answering", min(pct, 99)
                _update_job(job_id, stage=stage, progress=pct, message=msg)

            answer_text, sources = rag.answer_query(
                user_input=req.message,
                retriever_bundle=retriever_bundle,
                chat_history=session["chat_history"],
                on_stage=on_stage,
            )

            session["chat_history"].append(HumanMessage(content=req.message))
            session["chat_history"].append(AIMessage(content=answer_text))

            _update_job(
                job_id,
                stage="done",
                progress=100,
                message="Answer ready",
                result={"answer": answer_text, "sources": sources},
            )
        except Exception as exc:
            traceback.print_exc()  # surface the real error in the server console, not just the job dict
            _update_job(job_id, stage="error", progress=0, message="Failed", error=str(exc))

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}


@app.get("/chat/status/{job_id}")
def chat_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None or job["type"] != "chat":
        raise HTTPException(status_code=404, detail="Unknown chat job_id.")
    return job


@app.get("/sessions/{session_id}/history", response_model=List[HistoryMessage])
def get_history(session_id: str):
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id.")
    history = []
    for msg in session["chat_history"]:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        history.append(HistoryMessage(role=role, content=msg.content))
    return history


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Unknown session_id.")
    del SESSIONS[session_id]
    return {"status": "deleted"}