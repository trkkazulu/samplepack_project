import librosa
import numpy as np
import pandas as pd
import os

def extract_features(y, sr):  # Modified to take audio data directly
    """Extracts MFCCs and other features from an audio segment."""
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # Reduced MFCC count for simplicity
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]

        # Flatten the features
        mfccs = np.mean(mfccs.T, axis=0)
        chroma = np.mean(chroma.T, axis=0)
        spectral_centroid = np.mean(spectral_centroid)
        zcr = np.mean(zcr)

        features = np.concatenate((mfccs, chroma, [spectral_centroid], [zcr]))
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def process_audio_file(file_path, tempo, key, scale, style, technique, segment_length=1, sr=22050):  # Added sr default
    """Processes a single audio file, segmenting it and extracting features."""
    try:
        # Construct the full path to the audio file
        full_file_path = os.path.join("bass_recordings", file_path)  # <--- Corrected path

        y, sr = librosa.load(full_file_path, sr=sr, duration=segment_length*3)  # Load with specified SR, limit to 3 sec
        duration = librosa.get_duration(y=y, sr=sr)
        num_segments = int(duration / segment_length)

        features_list = []
        labels_list = []

        for i in range(num_segments):
            start = i * segment_length
            end = (i + 1) * segment_length
            segment = y[int(start * sr):int(end * sr)]  # Segment in samples

            features = extract_features(segment, sr)
            if features is not None:
                features_list.append(features)
                labels_list.append([tempo, key, scale, style, technique])  # Store labels

        return features_list, labels_list

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return [], []

# Example usage (replace with your actual file paths and CSV)
csv_file = "metadata.csv"  # <---- CSV SHOULD CONTAIN RELATIVE PATHS FROM bass_recordings
df = pd.read_csv(csv_file)

all_features = []
all_labels = []

for index, row in df.iterrows():
    file_path = row['file_path']
    tempo = row['tempo']
    key = row['key']
    scale = row['scale']
    style = row['style']
    technique = row['technique']

    features, labels = process_audio_file(file_path, tempo, key, scale, style, technique)

    all_features.extend(features)
    all_labels.extend(labels)

# Convert to NumPy arrays
X = np.array(all_features)  # Features
y = np.array(all_labels)  # Labels

print("Shape of X (Features):", X.shape)
print("Shape of y (Labels):", y.shape)

# No splitting for now
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Example split

#print("Shape of X_train:", X_train.shape)
#print("Shape of X_test:", X_test.shape)
#print("Shape of y_train:", y_train.shape)
#print("Shape of y_test:", y_test.shape)

# Save the features and labels for later use
np.save("X.npy", X)
np.save("y.npy", y)

print("Features (X) and labels (y) saved to X.npy and y.npy")
