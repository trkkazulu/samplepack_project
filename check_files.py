import os

audio_files = [
    "./audio_samples/bass samples - Omni Bass Take 10.wav",
    "./audio_samples/bass samples - Omni Bass.wav",
    "./audio_samples/bass samples - Omni Bass Take 8.wav"
]

for file_path in audio_files:
    if os.path.exists(file_path):
        print(f"Found: {file_path}")
    else:
        print(f"File not found: {file_path}")

