import librosa
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

def extract_features(y, sr):  # This is identical to your data_preparation
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

def prepare_new_audio(audio_path, segment_length=1, sr=22050): #Same values as data_prep
    """Loads a new audio file, segments it, and extracts features."""
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=segment_length*3)  # Load with specified SR, limit to 3 sec
        duration = librosa.get_duration(y=y, sr=sr)
        num_segments = int(duration / segment_length)
        features_list = []

        for i in range(num_segments):
            start = i * segment_length
            end = (i + 1) * segment_length
            segment = y[int(start * sr):int(end * sr)]  # Segment in samples

            features = extract_features(segment, sr)
            if features is not None:
                features_list.append(features)

        return np.array(features_list)  # Returns a NumPy array

    except Exception as e:
        print(f"Error preparing audio: {e}")
        return None

# Load the trained model
model = load_model("bass_style_model.keras")  # Update if you saved it with .h5

# Load a new audio file
new_audio_path = "/Users/jair-rohmparkerwells/Dev/samplepack_project/old_project/audio_samples/stick.wav"  # <--- UPDATE THIS PATH
new_audio_features = prepare_new_audio(new_audio_path)

if new_audio_features is not None and len(new_audio_features) > 0: #Check for data
    # Make predictions
    predictions = model.predict(new_audio_features)

    # Get the predicted class for each segment
    predicted_classes = np.argmax(predictions, axis=1) # Find index of max value
    print("Predictions shape", predictions.shape)

    # Convert predicted classes back to style names (if you one-hot encoded them)
    # This assumes you have the original style names in the same order as the one-hot encoding
    # Get the style names in the same order
    style_names = pd.get_dummies(pd.Series(np.load("y.npy")[:,3])).columns.tolist() #Unique style names
    predicted_styles = [style_names[i] for i in predicted_classes]  # Convert indices to names

    # Print the predicted styles for each segment
    print("Predicted Styles:")
    for i, style in enumerate(predicted_styles):
        print(f"Segment {i+1}: {style}")
else:
    print("Could not prepare the new audio for prediction.")
