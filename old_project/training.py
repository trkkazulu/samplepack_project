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
input_shape = X_train.shape[1:]  # (sequence_length, num_features)
latent_dim = 32  # Adjust as needed

# 4. Create the VAE Model
encoder, decoder, vae = create_cvae(input_shape, latent_dim)

# 5. Configure the Optimizer
optimizer = Adam(learning_rate=0.001) # Tune learning rate if needed

# 6. Define the Loss Function (Reconstruction Loss + KL Divergence)
def vae_loss(x, x_decoded_mean):
    """Define loss = reconstruction loss + KL loss"""
    # Reconstruction loss
    reconstruction_loss = tf.reduce_mean(tf.square(x - x_decoded_mean))
    # KL loss
    kl_loss = tf.reduce_sum(vae.losses)
    return reconstruction_loss + kl_loss

# 7. Training Loop (Using GradientTape)
epochs = 50  # Adjust as needed
batch_size = 32 #Adjust

@tf.function #Enable autograph for speed
def train_step(x):
    with tf.GradientTape() as tape:
        x_decoded_mean = vae(x) #Forward Pass
        loss = vae_loss(x, x_decoded_mean) #Calculate loss

    gradients = tape.gradient(loss, vae.trainable_variables) #Calculate gradients
    optimizer.apply_gradients(zip(gradients, vae.trainable_variables)) #Apply Gradients

#Prepare Dataset
train_dataset = tf.data.Dataset.from_tensor_slices(X_train).batch(batch_size)

print("Beginning Training")

for epoch in range(epochs):
    for step, x_batch_train in enumerate(train_dataset):
        train_step(x_batch_train)
    print("Epoch:", epoch+1, "Loss", loss.numpy()) #Print loss every epoch

print("Finished Training")

# 8. Save the Trained Model
vae.save("bass_vae_model.keras") #Or .h5 if needed
encoder.save("bass_vae_encoder.keras") #Good to save these separately
decoder.save("bass_vae_decoder.keras")
print("Trained VAE model saved as bass_vae_model.keras")
print("Trained encoder model saved as bass_vae_encoder.keras")
print("Trained decoder model saved as bass_vae_decoder.keras")
