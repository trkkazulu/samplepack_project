import numpy as np
import librosa
import librosa.display
import soundfile as sf
from tensorflow.keras.models import load_model

# 1. Load the Trained Models
vae = load_model("bass_vae_model.keras")
encoder = load_model("bass_vae_encoder.keras")
decoder = load_model("bass_vae_decoder.keras")

# 2. Define Parameters (Must match training!)
num_samples = 5  # Number of audio samples to generate
latent_dim = 32  # Latent space dimension
sequence_length = 10 # Sequence Length Used
num_features = 55 # Number of Features
input_shape = (sequence_length, num_features) # The input shape

# 3. Griffin-Lim Parameters
n_fft = 2048  # Number of FFT points (adjust as needed)
hop_length = 512  # Hop length (adjust as needed)
sr = 22050       # Sample rate
n_mels = 40

def decode_and_invert(latent_vector, decoder, sr, n_fft, hop_length, sequence_length, num_features, n_mels):
    """Decodes a latent vector and reconstructs the audio using Griffin-Lim."""

    #Sample audio shape.
    input_audio_shape = (1, sequence_length, num_features)

    # Expand to fit batch Size
    latent_vector = np.expand_dims(latent_vector, axis=0)

    # Decode the latent vector to generate audio features
    generated_features = decoder.predict(latent_vector)

    # Reshape (remove scaling)
    generated_features_reshaped = generated_features.reshape((sequence_length, num_features))

    # Extract MFCCs from generated_features
    generated_mfccs = generated_features_reshaped[:,:n_mels]

    # Inverse the Mel scale
    mel_spectrogram = librosa.feature.inverse.mfcc_to_mel(generated_mfccs.T, sr=sr, n_fft=n_fft)

    # Power to audio
    audio_recon = librosa.feature.inverse.mel_to_audio(mel_spectrogram, sr=sr, n_fft=n_fft)

    return audio_recon

# 4. Generate New Audio Samples
for i in range(num_samples):
    # Sample from the latent space
    random_latent_vector = np.random.normal(size=(latent_dim,))

    # Decode and Invert
    audio_recon = decode_and_invert(random_latent_vector, decoder, sr, n_fft, hop_length, sequence_length, num_features, n_mels)

    # Scale audio
    audio_recon = np.int16(audio_recon / np.max(np.abs(audio_recon))*32767)

    # Define output file
    output_file = f"generated_bass_{i+1}.wav"
    print(f"Generating: {output_file}")

    # Save the generated audio
    sf.write(output_file, audio_recon, sr)
