"""Playlist ingestion and timestamp-aware retrieval for YouTube RAG."""

from __future__ import annotations

import os
import re
import uuid
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import yt_dlp
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PayloadSchemaType, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from youtube_transcript_api import YouTubeTranscriptApi

CHUNK_SECONDS = 70
CHUNK_OVERLAP_SECONDS = 12
COLLECTION_NAME = "youtube_playlist_rag"
EMBEDDING_MODEL = "BAAI/bge-m3"
TRANSCRIPT_DIR = Path("transcripts")
AUDIO_DIR = Path("audio")
# Node.js is present on the user's machine. Modern YouTube extraction needs a
# JavaScript runtime for some player formats; explicitly telling yt-dlp to use
# it avoids the "No supported JavaScript runtime" warning and failed downloads.
YTDLP_OPTIONS = {"js_runtimes": {"node": {}}}
ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class Settings:
    qdrant_url: str
    qdrant_api_key: str
    groq_api_key: str
    groq_model: str = "openai/gpt-oss-120b"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        missing = [name for name in ("QDRANT_ENDPOINT", "QDRANT_API", "GROQ_KEY") if not os.getenv(name)]
        if missing:
            raise ValueError("Missing environment settings: " + ", ".join(missing) + ". Add them to .env.")
        return cls(os.environ["QDRANT_ENDPOINT"], os.environ["QDRANT_API"], os.environ["GROQ_KEY"], os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))


@dataclass(frozen=True)
class Video:
    id: str
    title: str
    url: str


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Citation:
    title: str
    start: float
    end: float
    url: str
    excerpt: str
    score: float


@dataclass(frozen=True)
class Answer:
    answer: str
    citations: list[Citation]


@dataclass(frozen=True)
class IndexReport:
    indexed_videos: int
    indexed_chunks: int
    skipped_videos: int
    failures: list[str]


class PlaylistRAG:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(COLLECTION_NAME):
            self.client.create_collection(collection_name=COLLECTION_NAME, vectors_config=VectorParams(size=self.embedder.get_sentence_embedding_dimension(), distance=Distance.COSINE))
        # Qdrant Cloud requires an explicit payload index before filtering by video_id.
        # Without this, its first "already indexed?" check returns HTTP 400.
        self.client.create_payload_index(COLLECTION_NAME, field_name="video_id", field_schema=PayloadSchemaType.KEYWORD, wait=True)

    def index_playlist(self, playlist_url: str, language: str, replace: bool, progress: ProgressCallback | None = None) -> IndexReport:
        videos = list(list_playlist_videos(playlist_url))
        if not videos:
            raise ValueError("No public videos were found in this playlist.")
        indexed_videos = indexed_chunks = skipped = 0
        failures: list[str] = []
        for position, video in enumerate(videos, start=1):
            if progress:
                progress(position - 1, len(videos), video.title)
            try:
                if self._video_exists(video.id) and not replace:
                    skipped += 1
                    continue
                chunks = make_chunks(get_transcript(video.id, language))
                if not chunks:
                    raise ValueError("the transcript was empty")
                self._delete_video(video.id)
                self._upsert(video, chunks)
                indexed_videos += 1
                indexed_chunks += len(chunks)
            except Exception as exc:
                failures.append(f"{video.title}: {exc}")
        if progress:
            progress(len(videos), len(videos), "Done")
        return IndexReport(indexed_videos, indexed_chunks, skipped, failures)

    def ask(self, question: str, limit: int = 5) -> Answer:
        query_vector = self.embedder.encode(question, normalize_embeddings=True).tolist()
        points = self.client.query_points(COLLECTION_NAME, query=query_vector, with_payload=True, limit=limit).points
        if not points:
            return Answer("I don’t have any indexed playlist content yet. Index a playlist first.", [])
        citations = [citation_from_point(point) for point in points]
        context = "\n\n".join(f"[Source {index + 1}: {item.title}, {format_seconds(item.start)}–{format_seconds(item.end)}]\n{item.excerpt}" for index, item in enumerate(citations))
        prompt = ("Answer using only the numbered sources below. Be concise and clear. If the sources do not answer the question, say that. Cite claims inline as [Source N].\n\n" f"Question: {question}\n\nSources:\n{context}")
        completion = Groq(api_key=self.settings.groq_api_key).chat.completions.create(model=self.settings.groq_model, temperature=0.1, messages=[{"role": "system", "content": "You are a precise YouTube lecture assistant."}, {"role": "user", "content": prompt}])
        return Answer(completion.choices[0].message.content or "No answer returned.", citations)

    def _video_exists(self, video_id: str) -> bool:
        result = self.client.scroll(COLLECTION_NAME, scroll_filter=Filter(must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]), limit=1, with_payload=False)
        return bool(result[0])

    def _delete_video(self, video_id: str) -> None:
        self.client.delete(COLLECTION_NAME, points_selector=Filter(must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]))

    def _upsert(self, video: Video, chunks: list[Segment]) -> None:
        vectors = self.embedder.encode([chunk.text for chunk in chunks], batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        points = [PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video.id}:{chunk.start:.3f}:{chunk.end:.3f}")), vector=vector.tolist(), payload={"video_id": video.id, "title": video.title, "video_url": video.url, "start": chunk.start, "end": chunk.end, "text": chunk.text}) for chunk, vector in zip(chunks, vectors)]
        self.client.upsert(COLLECTION_NAME, points=points, wait=True)


def list_playlist_videos(url: str) -> Iterable[Video]:
    options = {**YTDLP_OPTIONS, "quiet": True, "extract_flat": True, "skip_download": True, "ignoreerrors": True}
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)
    for entry in (info or {}).get("entries", []):
        if entry and entry.get("id"):
            video_id = entry["id"]
            yield Video(video_id, entry.get("title") or video_id, f"https://www.youtube.com/watch?v={video_id}")


def get_transcript(video_id: str, language: str) -> list[Segment]:
    cached = TRANSCRIPT_DIR / f"{video_id}.json"
    if cached.exists():
        return load_cached_transcript(cached)
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=[language, "en"])
        segments = [Segment(float(item.start), float(item.start + item.duration), text) for item in transcript if (text := clean_text(item.text))]
    except Exception:
        # Captions are often unavailable on lectures. In that case, make our own
        # timestamped transcript from the audio, just like the reference project.
        segments = transcribe_audio(video_id, language)
    if not segments:
        raise ValueError("no usable captions or speech could be found")
    save_transcript(cached, segments)
    return segments


def load_cached_transcript(path: Path) -> list[Segment]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_segments = data.get("segments", data.get("segements", []))
    return [Segment(float(item["start"]), float(item["end"]), clean_text(item["text"])) for item in raw_segments if clean_text(item.get("text", ""))]


def save_transcript(path: Path, segments: list[Segment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = {"segments": [{"start": item.start, "end": item.end, "text": item.text} for item in segments]}
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)


def transcribe_audio(video_id: str, language: str) -> list[Segment]:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    existing_audio = next(AUDIO_DIR.glob(f"{video_id}.*"), None)
    if existing_audio is None:
        options = {**YTDLP_OPTIONS, "format": "bestaudio/best", "outtmpl": str(AUDIO_DIR / "%(id)s.%(ext)s"), "quiet": True, "noplaylist": True}
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
        existing_audio = next(AUDIO_DIR.glob(f"{video_id}.*"), None)
    if existing_audio is None:
        raise ValueError("audio download failed")
    model = WhisperModel("small", device="cpu", compute_type="int8")
    iterator, _ = model.transcribe(str(existing_audio), language=language, beam_size=5, condition_on_previous_text=False, vad_filter=True)
    return [Segment(item.start, item.end, clean_text(item.text)) for item in iterator if clean_text(item.text)]


def make_chunks(segments: list[Segment]) -> list[Segment]:
    chunks: list[Segment] = []
    start_index = 0
    while start_index < len(segments):
        window_start = segments[start_index].start
        window_end = window_start + CHUNK_SECONDS
        selected = [segment for segment in segments[start_index:] if segment.start < window_end]
        if not selected:
            start_index += 1
            continue
        chunks.append(Segment(selected[0].start, selected[-1].end, " ".join(segment.text for segment in selected)))
        next_start = window_end - CHUNK_OVERLAP_SECONDS
        while start_index < len(segments) and segments[start_index].start < next_start:
            start_index += 1
    return chunks


def citation_from_point(point) -> Citation:
    payload = point.payload
    start = float(payload["start"])
    return Citation(payload["title"], start, float(payload["end"]), f"{payload['video_url']}&t={int(start)}s", payload["text"], float(point.score))


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("&amp;", "&")).strip()


def format_seconds(seconds: float) -> str:
    seconds = max(0, round(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"
