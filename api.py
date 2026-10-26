from __future__ import annotations
import threading
import uuid
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, HttpUrl
from youtube_rag import PlaylistRAG, Settings

ROOT = Path(__file__).parent
app = FastAPI(title="YouTube Playlist RAG API", version="1.0.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

class IndexRequest(BaseModel):
    playlist_url: HttpUrl
    language: str = Field(default="en", min_length=2, max_length=10)
    replace: bool = False

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=800)

@dataclass
class IndexJob:
    status: str = "queued"; current: int = 0; total: int = 0; current_title: str = "Waiting to start"
    indexed_videos: int = 0; indexed_chunks: int = 0; skipped_videos: int = 0
    failures: list[str] = field(default_factory=list); error: str | None = None

jobs: dict[str, IndexJob] = {}; jobs_lock = threading.Lock()

@lru_cache(maxsize=1)
def rag() -> PlaylistRAG:
    return PlaylistRAG(Settings.from_env())

@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")

@app.get("/api/health")
def health() -> dict[str, str]: return {"status": "ok"}

@app.post("/api/index", status_code=202)
def start_indexing(request: IndexRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    job_id = str(uuid.uuid4())
    with jobs_lock: jobs[job_id] = IndexJob()
    background_tasks.add_task(run_indexing, job_id, str(request.playlist_url), request.language, request.replace)
    return {"job_id": job_id}

@app.get("/api/index/{job_id}")
def indexing_status(job_id: str) -> dict:
    with jobs_lock:
        if job_id not in jobs: raise HTTPException(status_code=404, detail="Unknown indexing job")
        return asdict(jobs[job_id])

@app.post("/api/ask")
def ask(request: AskRequest) -> dict:
    try: result = rag().ask(request.question.strip())
    except Exception as exc: raise HTTPException(status_code=503, detail=f"Search is unavailable: {exc}") from exc
    return {"answer": result.answer, "citations": [asdict(citation) for citation in result.citations]}

def run_indexing(job_id: str, playlist_url: str, language: str, replace: bool) -> None:
    with jobs_lock: jobs[job_id].status = "running"
    def progress(current: int, total: int, title: str) -> None:
        with jobs_lock:
            job = jobs[job_id]; job.current, job.total, job.current_title = current, total, title
    try:
        report = rag().index_playlist(playlist_url, language, replace, progress)
        with jobs_lock:
            job = jobs[job_id]; job.status = "completed"; job.indexed_videos = report.indexed_videos; job.indexed_chunks = report.indexed_chunks; job.skipped_videos = report.skipped_videos; job.failures = report.failures
    except Exception as exc:
        with jobs_lock: jobs[job_id].status, jobs[job_id].error = "failed", str(exc)
