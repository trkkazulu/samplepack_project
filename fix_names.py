import os
import pandas as pd

def fix_names(root_dir="bass_recordings", csv_file="metadata.csv"):
    """
    Renames all files in the 'funk', 'jazz', 'rock', and 'space'
    subdirectories of root_dir and updates the metadata.csv file.
    """

    subdirectories = ["funk", "jazz", "rock", "space"]

    # Load the CSV file into a Pandas DataFrame
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        return

    for subdir in subdirectories:
        subdir_path = os.path.join(root_dir, subdir)

        if not os.path.isdir(subdir_path):
            print(f"Warning: Directory '{subdir_path}' not found.")
            continue

        files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]

        for filename in files:
            if filename.lower().endswith(".wav"):
                old_filepath = os.path.join(subdir_path, filename)

                # 1. Replace spaces with underscores in the original filename
                new_filename_base = filename[:-4].replace(" ", "_")  # Remove .wav, replace spaces

                # 2. Prefix with subdirectory name
                new_filename = f"{subdir}_{new_filename_base}.wav"

                new_filepath = os.path.join(subdir_path, new_filename)

                try:
                    os.rename(old_filepath, new_filepath)
                    print(f"Renamed '{filename}' to '{new_filename}' in '{subdir}'")

                    # **UPDATE CSV HERE**
                    # Find the row in the DataFrame with the old file path
                    relative_old_filepath = os.path.join(subdir, filename)  # Path relative to bass_recordings
                    row_index = df[df['file_path'] == relative_old_filepath].index

                    if len(row_index) > 0:  #Check if a row with the old name exists
                        # Update the 'file_path' column with the new file path
                        relative_new_filepath = os.path.join(subdir, new_filename)  # New relative path
                        df.loc[row_index, 'file_path'] = relative_new_filepath
                        print(f"Updated CSV: '{relative_old_filepath}' to '{relative_new_filepath}'")

                    else:
                        print(f"Warning: Could not find '{relative_old_filepath}' in CSV.")

                except OSError as e:
                    print(f"Error renaming '{filename}': {e}")
            else:
                print(f"Skipping non-wav file: '{filename}' in '{subdir}'")

    # Save the updated DataFrame to the CSV file
    try:
        df.to_csv(csv_file, index=False)  # index=False to avoid writing the index column
        print(f"Successfully updated '{csv_file}'.")
    except Exception as e:
        print(f"Error saving updated CSV: {e}")

if __name__ == "__main__":
    fix_names()
