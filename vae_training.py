import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from vae_model import create_cvae  # Import the VAE model definition

# 1. Load the Data
X_sequenced = np.load("X_sequenced.npy")

# 2. Data Splitting
X_train, X_test, _, _ = train_test_split(X_sequenced, X_sequenced, test_size=0.2, random_state=42) #No labels

# 3. Determine Input Shape and Latent Dimension
print("Shape of X_train", X_train.shape)
input_shape = X_train.shape[1:]  # (sequence_length, num_features)
latent_dim = 32  # Adjust as needed
print("Input Shape", input_shape)

# 4. Create the VAE Model
vae, encoder, decoder = create_cvae(input_shape, latent_dim)

# 5. Configure the Optimizer
optimizer = Adam(learning_rate=0.001) # Tune learning rate if needed

# 6. Define the Loss Function (Reconstruction Loss)
def reconstruction_loss_fn(x, x_decoded_mean):
    """Define loss = reconstruction loss"""
    # Reconstruction loss
    # Flatten both tensors to handle shape differences
    x_flat = tf.reshape(x, [tf.shape(x)[0], -1])
    x_decoded_mean_flat = tf.reshape(x_decoded_mean, [tf.shape(x_decoded_mean)[0], -1])
    reconstruction_loss = tf.reduce_mean(tf.abs(x_flat - x_decoded_mean_flat))   # Mean Absolute Error
    return reconstruction_loss

# 7. Training Loop (Using GradientTape)
epochs = 50  # Adjust as needed
batch_size = 32 #Adjust

@tf.function #Enable autograph for speed
def train_step(x):
    with tf.GradientTape() as tape:
        x_decoded_mean = vae(x) #Forward Pass
        loss = reconstruction_loss_fn(x, x_decoded_mean) #Calculate loss
        loss += sum(vae.losses)  # Add the KL loss calculated in the CVAE class
    gradients = tape.gradient(loss, vae.trainable_variables) #Calculate gradients
    optimizer.apply_gradients(zip(gradients, vae.trainable_variables)) #Apply Gradients

print("Beginning Training")

for epoch in range(epochs):
    for x_batch_train in X_train:
        train_step(tf.expand_dims(x_batch_train, axis=0))  # NO BATCHING
    print("Epoch:", epoch+1, "Loss", loss.numpy()) #Print loss every epoch

print("Finished Training")

# 8. Save the Trained Model
try:
    vae.save("bass_vae_model.keras")  # Or .h5 if needed
    encoder.save("bass_vae_encoder.keras")  # Good to save these separately
    decoder.save("bass_vae_decoder.keras")
    print("Trained VAE model saved as bass_vae_model.keras")
    print("Trained encoder model saved as bass_vae_encoder.keras")
    print("Trained decoder model saved as bass_vae_decoder.keras")
except Exception as e:
    print(f"Error saving the model: {e}")
