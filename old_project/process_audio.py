import librosa
import numpy as np
import os

AUDIO_PATH = "./audio_samples/"
PROCESSED_PATH = "./processed_samples/"

if not os.path.exists(PROCESSED_PATH):
    os.makedirs(PROCESSED_PATH)

def process_audio(file):
    y, sr = librosa.load(file, sr=48000)
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=22050)
    sr_resampled = 22050
    n_fft = 254
    hop_length = 256
    win_length = 254
    mel_spec = librosa.feature.melspectrogram(y=y_resampled, sr=sr_resampled, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=128)

    # Added lines
    print(f"Raw mel_spec from {file} all zeros: {np.all(mel_spec == 0)}")
    print(f"Raw mel_spec from {file} min: {np.min(mel_spec)}")
    print(f"Raw mel_spec from {file} max: {np.max(mel_spec)}")

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    #Added lines
    print(f"dB mel_spec from {file} all zeros: {np.all(mel_spec_db == 0)}")
    print(f"dB mel_spec from {file} min: {np.min(mel_spec_db)}")
    print(f"dB mel_spec from {file} max: {np.max(mel_spec_db)}")

    return mel_spec_db

for file in os.listdir(AUDIO_PATH):
    if file.endswith(".wav"):
        mel_spec = process_audio(AUDIO_PATH + file)
        np.save(PROCESSED_PATH + file.replace(".wav", ".npy"), mel_spec)
        print(f"Processed: {file}")
