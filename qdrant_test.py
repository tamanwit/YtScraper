import json
import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from groq import Groq

load_dotenv()

COLLECTION_NAME = "youtube_lectures"


# 1. Load chunks
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("Chunks loaded:", len(chunks))


# 2. Load embedding model
model = SentenceTransformer("BAAI/bge-m3")

print("Embedding model loaded")


# 3. Create embeddings
texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(
    texts,
    batch_size=16,
    show_progress_bar=True
)

print("Embeddings created:", len(embeddings))


# 4. Connect to Qdrant Cloud
client = QdrantClient(
    url=os.getenv("QDRANT_ENDPOINT"),
    api_key=os.getenv("QDRANT_API"),
)

print("Qdrant connected")


# 5. Create collection
if not client.collection_exists(COLLECTION_NAME):

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=1024,
            distance=Distance.COSINE
        )
    )

    print("Collection created")

else:
    print("Collection already exists")


# 6. Create Qdrant points
points = []

for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):

    point = PointStruct(
        id=i,
        vector=embedding.tolist(),
        payload={
            "video_id": chunk["video_id"],
            "start": chunk["start"],
            "end": chunk["end"],
            "text": chunk["text"]
        }
    )
    points.append(point)

# 7. Upload
client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)
print("Uploaded:", len(points), "points")


# searching
query_text = "videos will come daily or not ?"
query_embedding = model.encode(
    query_text,
    batch_size=16,
    show_progress_bar=True
)
context = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_embedding.tolist(),
    with_payload=True,
    limit=3
)

# convert context.points into proper string context
llm_context = ""
for result in context.points:
    llm_context += (
        f"Start: {result.payload['start']}\n"
        f"End: {result.payload['end']}\n"
        f"Text: {result.payload['text']}\n"
        f"\n---\n\n"
    )

# print(llm_context)



# passing the context and query to the LLM
groq_client = Groq(api_key=os.getenv("GROQ_KEY"))

system_message = {
    "role": "system",
    "content": "Answer the user's question only using the provided lecture context."
}
user_message = {
    "role": "user",
    "content" : f"""
        user query  : {query_text}, to answer this question use this context : {llm_context}
"""
}
messages = [system_message, user_message]
completion = groq_client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=messages,
    temperature=0.2
)

print(completion.choices[0].message.content)