import json
from sentence_transformers import SentenceTransformer

print("Loading BGE-M3...")

model = SentenceTransformer("BAAI/bge-m3")

print("Model loaded!")

with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [chunk["text"] for chunk in chunks]

print(f"Embedding {len(texts)} chunks...")

embeddings = model.encode(
    texts,
    batch_size=16,
    show_progress_bar=True
)

print("Embedding complete!")
print("Number of embeddings:", len(embeddings))
print("Embedding dimension:", len(embeddings[0]))