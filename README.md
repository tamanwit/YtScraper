# YouTube Playlist RAG

Search a whole YouTube playlist, get an answer grounded in captions, and open the exact moment that supports it.

Demo Video Link : https://drive.google.com/file/d/10fnfuvjTc8oAMObsKz5DfuXyPzeVISVH/view?usp=sharing

## What it does

- Imports every public video from a playlist.
- Retrieves YouTube captions while preserving timestamps.
- Splits captions into overlapping, time-bounded passages and stores them in Qdrant.
- Uses semantic search plus Groq to answer only from retrieved passages.
- Returns one-click YouTube links that begin at the relevant second.
- Skips already-indexed videos by default; use **Replace already-indexed videos** to refresh them.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Add this to `.env` (never commit this file):

   ```env
   QDRANT_ENDPOINT=https://your-cluster.qdrant.io
   QDRANT_API=your-qdrant-api-key
   GROQ_KEY=your-groq-api-key
   # Optional:
   GROQ_MODEL=openai/gpt-oss-120b
   ```

4. Start the app: `uvicorn api:app --reload`
5. Open http://127.0.0.1:8000 in your browser.

## Note

The app uses a video’s public YouTube captions. Videos without captions are reported while the rest of the playlist continues to index. Change `COLLECTION_NAME` in `youtube_rag.py` if you need separate libraries for separate subjects.
