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
audio_file = "Podcast_4_Interview_with_Quique.mp3"       # <--- Replace with your audio file
output_txt = "transcription_with_speakers.txt"
batch_size = 16  # reduce if low on GPU mem
compute_type = "float32"  # use "float16" if on GPU with enough memory

# --- DEVICE SETUP ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- LOAD WHISPERX MODEL ---
print("Loading WhisperX model...")
model = whisperx.load_model("small", device, compute_type=compute_type)

# --- TRANSCRIBE AUDIO ---
print("Transcribing audio...")
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)

print(result["segments"])  # before alignment

# --- ALIGN TRANSCRIPTION ---
print("Loading alignment model...")
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)

print("Aligning words...")
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

print(result["segments"])  # after alignment

# --- LOAD DIARIZATION MODEL ---
print("Loading speaker diarization model...")
diarize_model = whisperx.DiarizationPipeline(
    use_auth_token=your_token,
    device=device
)

# --- DIARIZE AUDIO ---
print("Diarizing audio...")
diarize_segments = diarize_model(audio)

# --- ASSIGN SPEAKERS ---
print("Assigning speaker labels to words...")
result = whisperx.assign_word_speakers(diarize_segments, result)

# --- SAVE TO TEXT FILE ---
print(f"Saving output to {output_txt}...")
with open(output_txt, "w", encoding="utf-8") as f:
    for segment in result["segments"]:
        speaker = segment.get("speaker", "Unknown")
        text = segment["text"].strip()
        f.write(f"[{speaker}]: {text}\n")

print("Done! File saved successfully! Hurray!!!")