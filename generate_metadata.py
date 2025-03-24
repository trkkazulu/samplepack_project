import os
import pandas as pd
import librosa

def generate_metadata(root_dir="bass_recordings", csv_file="metadata.csv"):
    """
    Generates a new metadata.csv file by traversing the bass_recordings directory
    and extracting information from each WAV file.
    """

    data = []  # List to store the metadata for each file
    columns = ["file_path", "tempo", "key", "scale", "style", "technique", "duration"]  # Define column names

    for style in os.listdir(root_dir):  # Iterate over subdirectories (funk, jazz, rock, etc.)
        style_path = os.path.join(root_dir, style)

        if not os.path.isdir(style_path):
            continue  # Skip non-directories

        for filename in os.listdir(style_path):
            if filename.lower().endswith(".wav"):
                file_path = os.path.join(style, filename)  # Relative path from root_dir

                # **Extract Metadata from Filename (Adapt to your naming convention)**
                # This is where you'll need to parse the filename to extract tempo, key, scale, and technique
                # The following is just an example; modify it to match your actual filenames
                parts = filename[:-4].split("_")  # Remove ".wav", split by underscores
                if len(parts) >= 2:
                    tempo = 120 #Example Value Replace with Value from the File name
                    key = "C" #Example Value Replace with Value from the File name
                    scale = "Major" #Example Value Replace with Value from the File name
                    technique = "Finger Style" #Example Value Replace with Value from the File name
                else: #Default or Error
                    tempo = "Unknown"
                    key = "Unknown"
                    scale = "Unknown"
                    technique = "Unknown"

                # Extract Audio Duration
                try:
                    y, sr = librosa.load(os.path.join(root_dir, file_path))
                    duration = librosa.get_duration(y=y, sr=sr)
                except Exception as e:
                    print(f"Error loading {file_path} for duration: {e}")
                    duration = "Unknown"

                # Add the metadata to the data list
                data.append([file_path, tempo, key, scale, style, technique, duration])

    # Create a Pandas DataFrame from the data
    df = pd.DataFrame(data, columns=columns)

    # Save the DataFrame to a CSV file
    try:
        df.to_csv(csv_file, index=False)
        print(f"Successfully created '{csv_file}'.")
    except Exception as e:
        print(f"Error saving '{csv_file}': {e}")

if __name__ == "__main__":
    generate_metadata()
