import os
import yaml

# Path to the audio samples directory
directory = './audio_samples/'

# List to store the audio file information
audio_files = []

# Loop through all files in the directory
for filename in os.listdir(directory):
    # Check if the file is a .wav file
    if filename.endswith('.wav'):
        # Construct the full file path
        file_path = os.path.join(directory, filename)
        
        # Add the file information to the list
        audio_files.append({
            'label': filename,
            'path': file_path
        })

# Define the content for the YAML file
yaml_content = {
    'dataset': {
        'audio_files': audio_files
    },
    'channels': 1,
    'sample_rate': 48000
}

# Write the content to the metadata.yaml file
yaml_file_path = './audio_samples/metadata.yaml'
with open(yaml_file_path, 'w') as yaml_file:
    yaml.dump(yaml_content, yaml_file, default_flow_style=False)

print(f"metadata.yaml file regenerated: {yaml_file_path}")

