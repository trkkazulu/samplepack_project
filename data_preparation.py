import librosa
import numpy as np
import pandas as pd
import os

def extract_features(y, sr):
    """Extracts MFCCs and other features from an audio segment."""
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)[0]

        # Flatten the features
        mfccs = np.mean(mfccs.T, axis=0)
        chroma = np.mean(chroma.T, axis=0)
        spectral_centroid = np.mean(spectral_centroid)
        zcr = np.mean(zcr)
        spectral_contrast = np.mean(spectral_contrast)

        features = np.concatenate((mfccs, chroma, [spectral_centroid], [zcr], [spectral_contrast]))
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def process_audio_file(file_path, segment_length=0.5, sr=22050):  # Reduced segment length
    """Processes a single audio file, segmenting it and extracting features."""
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=segment_length*6) #Increased Duration

        #Ensure to use the same samplerate when loading and extracting features!
        duration = librosa.get_duration(y=y, sr=sr)
        num_segments = int(duration / segment_length)

        features_list = []

        for i in range(num_segments):
            start = i * segment_length
            end = (i + 1) * segment_length
            segment = y[int(start * sr):int(end * sr)]

            features = extract_features(segment, sr)
            if features is not None:
                features_list.append(features)

        return features_list

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []


def add_noise(data, noise_factor=0.01):
    """Adds random noise to the audio data."""
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Limit the values to between -1 and 1
    return np.clip(augmented_data, -1, 1)

def scale_minmax(data, min=-1, max=1):
    """Scales the data to the range of min and max."""
    data_std = (data - np.min(data)) / (np.max(data) - np.min(data))
    data_scaled = data_std * (max - min) + min
    return data_scaled

def create_sequences(data, sequence_length=10):
    """Creates sequences of feature vectors from the data."""
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

# Example usage (replace with your actual file paths and CSV)
csv_file = "metadata.csv"  # <---- UPDATE THIS PATH
root_dir = "bass_recordings"  #Define root Dir here
df = pd.read_csv(csv_file)

all_features = []

for index, row in df.iterrows():
    file_path = os.path.join(root_dir, row['file_path']) #Correct Path
    features = process_audio_file(file_path)

    all_features.extend(features)

# Convert to NumPy arrays
X = np.array(all_features)  # Features

# 1. Add Noise Augmentation
X_noisy = np.array([add_noise(x) for x in X])

# 2. Scale MinMax
X_scaled = np.array([scale_minmax(x, min=-1, max=1) for x in X_noisy], dtype=np.float32) #Cast to float32

# 3. Create Sequences
sequence_length = 10  # Adjust as needed
X_sequenced = create_sequences(X_scaled, sequence_length)

print("Shape of X (Original Features):", X.shape)
print("Shape of X_noisy (Noisy Features):", X_noisy.shape)
print("Shape of X_scaled (Scaled Features):", X_scaled.shape)
print("Shape of X_sequenced (Sequenced Features):", X_sequenced.shape)

# Save the sequenced features for later use
np.save("X_sequenced.npy", X_sequenced)

print("Sequenced features (X_sequenced) saved to X_sequenced.npy")
