import whisperx
import torch
from dotenv import load_dotenv
import os

# --- LOAD .ENV VARIABLES ---
load_dotenv()
your_token = os.getenv("HUGGINGFACE_TOKEN")

if your_token is None:
    raise ValueError("Hugging Face token not found! Please set it in your .env file.")

# --- SETTINGS ---
audio_file = "your_audio_file.wav"           # <--- Replace with your audio file
output_txt = "transcription_with_speakers.txt"

# --- DEVICE SETUP ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- LOAD MODELS ---
print("Loading WhisperX model...")
model = whisperx.load_model("small", device, compute_type="float32")

# print("Transcribing audio...")
# result = model.transcribe(audio_file)
print("Transcribing and translating Spanish audio to English...")
result = model.transcribe(audio_file, language="es", task="translate")

print("Loading speaker diarization model...")
diarize_model = whisperx.DiarizationPipeline(
    use_auth_token=your_token, 
    device=device
)

print("Diarizing audio...")
diarize_segments = diarize_model(audio_file)

# --- ALIGN TRANSCRIPTION WITH SPEAKERS ---
print("Aligning transcription with speaker labels...")
aligned_result = whisperx.align_whisper_with_diarization(result, diarize_segments)

# --- SAVE TO TEXT FILE ---
print(f"Saving output to {output_txt}...")
with open(output_txt, "w", encoding="utf-8") as f:
    for segment in aligned_result["segments"]:
        speaker = segment.get("speaker", "Unknown")
        text = segment["text"].strip()
        f.write(f"[{speaker}]: {text}\n")

print("✅ Done! File saved.")
