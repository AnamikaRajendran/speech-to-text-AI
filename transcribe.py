import whisper

# Load the Whisper model
model = whisper.load_model("medium")  # Change to "large" for better accuracy

# Path to your Spanish audio file (update with the actual file path)
audio_path = "WhatsApp Ptt 2025-02-17 at 11.57.50 AM (online-audio-converter.com).mp3"

# Transcribe and translate the audio
result = model.transcribe(audio_path, task="translate")

# Print the translated English text
print("Translated English text:")
print(result["text"])
