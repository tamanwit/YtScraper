import os
import json
from faster_whisper import WhisperModel

audio_file = "audio/S8tpzqrwnHk.webm"
os.makedirs("transcripts", exist_ok=True)
video_id = os.path.splitext(os.path.basename(audio_file))[0]
output_file = f"transcripts/{video_id}.json"


model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

segments, info = model.transcribe(
    audio_file, beam_size=5,
    language="en", condition_on_previous_text=False, vad_filter=True
)

transcript = []
for segment in segments:
    transcript.append({
        "start" : segment.start,
        "end" : segment.end,
        "text" : segment.text.strip()
    })

data = {
    "video_id": video_id,
    "language": info.language,
    "language_probability": info.language_probability,
    "segements" : transcript
}

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    

print(f"Transcript saved to: {output_file}")
print(f"Total segments: {len(transcript)}")
