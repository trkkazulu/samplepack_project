import numpy as np
import matplotlib.pyplot as plt
import os

PROCESSED_DATA_PATH = './processed_samples/'
OUTPUT_IMAGE_PATH = './spectrogram_images/' # Added output path

if not os.path.exists(OUTPUT_IMAGE_PATH): #added image path creation
    os.makedirs(OUTPUT_IMAGE_PATH)

for npy_file in os.listdir(PROCESSED_DATA_PATH):
    if npy_file.endswith('.npy') and not npy_file.endswith('_generated.npy'):
        full_file_path = os.path.join(PROCESSED_DATA_PATH, npy_file)
        print(f"Full file path: {full_file_path}")
        try:
            mel_spec = np.load(full_file_path)
            plt.figure()
            plt.imshow(mel_spec, aspect='auto', origin='lower')
            plt.title(npy_file)
            plt.colorbar()
            output_image_file = os.path.join(OUTPUT_IMAGE_PATH, f"{os.path.splitext(npy_file)[0]}.png") #create output file name
            plt.savefig(output_image_file) #save the image
            plt.close() # close the figure so that the next image is not overlayed.
            print(f"saved: {output_image_file}") #print that the file was saved.
        except Exception as e:
            print(f"Error loading {npy_file}: {e}")
