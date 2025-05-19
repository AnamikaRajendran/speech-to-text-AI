import whisperx
import torch
from dotenv import load_dotenv
import os
import time

# --- LOAD .ENV VARIABLES ---
load_dotenv()
your_token = os.getenv("HUGGINGFACE_TOKEN")

if your_token is None:
    raise ValueError("Hugging Face token not found! Please set it in your .env file.")

# --- SETTINGS ---
audio_file = "Podcast_4_Interview_with_Quique.mp3"  # <--- Your Spanish audio file here
output_txt_spanish = "transcription_spanish.txt"
output_txt_english = "transcription_english.txt"
batch_size = 16
compute_type = "float32"

# --- DEVICE SETUP ---
device = "cuda" if torch.cuda.is_available() else "cpu"

start_time = time.time()

# --- LOAD AUDIO ---
print("Loading audio...")
audio = whisperx.load_audio(audio_file)

# --- LOAD WHISPERX MODEL ---
print("Loading WhisperX model...")
model = whisperx.load_model("small", device, compute_type=compute_type)

# --- TRANSCRIBE TO SPANISH ---
print("Transcribing Spanish audio...")
result_spanish = model.transcribe(audio, batch_size=batch_size, language="es")

# --- TRANSLATE TO ENGLISH ---
print("Translating Spanish audio to English...")
result_english = model.transcribe(audio, batch_size=batch_size, language="es", task="translate")

# --- ALIGN SPANISH TRANSCRIPTION ---
print("Loading alignment model...")
model_a, metadata = whisperx.load_align_model(language_code="es", device=device)

print("Aligning Spanish transcription...")
result_spanish = whisperx.align(result_spanish["segments"], model_a, metadata, audio, device, return_char_alignments=False)

# --- ALIGN ENGLISH TRANSLATION (still original audio) ---
print("Aligning English translation...")
# Even if it's translated, alignment happens against original Spanish audio
result_english = whisperx.align(result_english["segments"], model_a, metadata, audio, device, return_char_alignments=False)

# --- LOAD DIARIZATION MODEL ---
print("Loading speaker diarization model...")
diarize_model = whisperx.DiarizationPipeline(
    use_auth_token=your_token,
    device=device
)

# --- PERFORM DIARIZATION ---
print("Diarizing audio...")
diarize_segments = diarize_model(audio)

# --- ASSIGN SPEAKERS ---
print("Assigning speakers to Spanish segments...")
result_spanish = whisperx.assign_word_speakers(diarize_segments, result_spanish)

print("Assigning speakers to English segments...")
result_english = whisperx.assign_word_speakers(diarize_segments, result_english)

# --- SAVE SPANISH OUTPUT ---
print(f"Saving Spanish transcription to {output_txt_spanish}...")
with open(output_txt_spanish, "w", encoding="utf-8") as f:
    for segment in result_spanish["segments"]:
        speaker = segment.get("speaker", "Unknown")
        text = segment["text"].strip()
        f.write(f"[{speaker}]: {text}\n")

# --- SAVE ENGLISH OUTPUT ---
print(f"Saving English transcription to {output_txt_english}...")
with open(output_txt_english, "w", encoding="utf-8") as f:
    for segment in result_english["segments"]:
        speaker = segment.get("speaker", "Unknown")
        text = segment["text"].strip()
        f.write(f"[{speaker}]: {text}\n")

end_time = time.time()
print("Done! Both Spanish and English transcriptions saved with speaker labels. Hurray!!")
print(f"Total execution time: {end_time - start_time:.2f} seconds.")