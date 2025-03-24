import numpy as np
import tensorflow as tf
import os
import scipy.io.wavfile as wav
import librosa

MODEL_PATH = './ai_bass_synth_model.h5'
PROCESSED_DATA_PATH = './processed_samples/'

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

for npy_file in os.listdir(PROCESSED_DATA_PATH):
    if npy_file.endswith('.npy'):
        input_data = np.load(os.path.join(PROCESSED_DATA_PATH, npy_file))
        print(f"Shape of input data from {npy_file} before padding/truncation: {input_data.shape}")

        if input_data.shape[1] < 233:
            padding = 233 - input_data.shape[1]
            input_data = np.pad(input_data, ((0, 0), (0, padding)), mode='constant')
        elif input_data.shape[1] > 233:
            input_data = input_data[:, :233]

        input_data = input_data.reshape((-1, 128, 233, 1))
        print(f"Shape of input data from {npy_file} after padding/truncation: {input_data.shape}")

        generated_audio = model.predict(input_data)
        generated_audio = np.squeeze(generated_audio)
        generated_audio = generated_audio.T

        # Added lines
        print(f"Model output from {npy_file} all zeros: {np.all(generated_audio == 0)}")
        print(f"Model output from {npy_file} min: {np.min(generated_audio)}")
        print(f"Model output from {npy_file} max: {np.max(generated_audio)}")

        n_fft = 254
        hop_length = 256
        win_length = 254

        if np.all(generated_audio == 0):
            print(f"Warning: Spectrogram from {npy_file} is all zeros. Skipping.")
            continue

        try:
            reconstructed_audio = librosa.istft(
                generated_audio,
                hop_length=hop_length,
                win_length=win_length,
                n_fft=n_fft
            )
        except Exception as e:
            print(f"Error reconstructing audio from {npy_file}: {e}")
            continue

        reconstructed_audio = np.clip(reconstructed_audio, -1, 1)

        if len(reconstructed_audio) == 0 or np.all(reconstructed_audio == 0):
            print(f"Warning: Reconstructed audio from {npy_file} is empty or all zeros. Skipping.")
            continue

        output_wav_path = os.path.join(PROCESSED_DATA_PATH, f"{os.path.splitext(npy_file)[0]}_generated.wav")
        wav.write(output_wav_path, 22050, (reconstructed_audio * 32767).astype(np.int16))

        print(f"Generated WAV file saved to {output_wav_path}")
