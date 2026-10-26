import json


CHUNK_SECONDS = 75
OVERLAP_SECONDS = 15


def load_transcript(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_chunks(data):
    segments = data["segements"]
    chunks = []

    start_index = 0

    while start_index < len(segments):

        chunk_start = segments[start_index]["start"]
        chunk_end = chunk_start + CHUNK_SECONDS

        chunk_segments = []

        for segment in segments[start_index:]:
            if segment["start"] >= chunk_end:
                break

            chunk_segments.append(segment)

        if not chunk_segments:
            start_index += 1
            continue

        text = " ".join(
            segment["text"]
            for segment in chunk_segments
        )

        chunks.append({
            "video_id": data["video_id"],
            "start": chunk_segments[0]["start"],
            "end": chunk_segments[-1]["end"],
            "text": text
        })

        # Move forward while keeping overlap
        next_start_time = chunk_end - OVERLAP_SECONDS

        while (
            start_index < len(segments)
            and segments[start_index]["start"] < next_start_time
        ):
            start_index += 1

    return chunks


data = load_transcript("transcripts/S8tpzqrwnHk.json")

chunks = create_chunks(data)
with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print("Chunks saved to chunks.json")

print("Total chunks:", len(chunks))

for chunk in chunks[:3]:
    print("\n--- CHUNK ---")
    print("Start:", chunk["start"])
    print("End:", chunk["end"])
    print("Text:", chunk["text"])