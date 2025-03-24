import numpy as np
import tensorflow as tf
import os
import datetime

PROCESSED_DATA_PATH = './processed_samples/'
MODEL_PATH = './ai_bass_synth_model.h5'

data = []
for npy_file in os.listdir(PROCESSED_DATA_PATH):
    if npy_file.endswith('.npy') and not npy_file.endswith('_generated.npy'):
        mel_spec = np.load(os.path.join(PROCESSED_DATA_PATH, npy_file))
        if mel_spec.shape[1] < 233:
            padding = 233 - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, padding)), mode='constant')
        elif mel_spec.shape[1] > 233:
            mel_spec = mel_spec[:, :233]
        data.append(mel_spec)

data = np.array(data)
data = data.reshape((-1, 128, 233, 1))

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(128, 233, 1)),
    tf.keras.layers.Resizing(64, 116),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Resizing(32, 58),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Resizing(64, 116),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Resizing(128, 233),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'), # Added layer.
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'), # Added layer.
    tf.keras.layers.Conv2D(1, (3, 3), activation='linear', padding='same')
])

model.compile(optimizer='adam', loss='mse')

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

print("Training started...")
model.fit(data, data, epochs=50, batch_size=4, callbacks=[tensorboard_callback])
print("Training finished.")

model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
