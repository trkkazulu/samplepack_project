import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class CVAE(Model):
    def __init__(self, input_shape, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape_val = input_shape
        # Encoder
        self.encoder_inputs = layers.Input(shape=input_shape)
        self.conv1d_1 = layers.Conv1D(32, 3, activation="relu", strides=2, padding="same")
        self.conv1d_2 = layers.Conv1D(64, 3, activation="relu", strides=2, padding="same")
        self.flatten = layers.Flatten()
        self.dense_1 = layers.Dense(16, activation="relu")
        self.z_mean = layers.Dense(latent_dim, name="z_mean")
        self.z_log_var = layers.Dense(latent_dim, name="z_log_var")
        self.sampling = Sampling()

        # Decoder
        self.dense_2 = layers.Dense(192, activation="relu")
        self.reshape = layers.Reshape((3, 64)) #REVERTED CHANGES HERE
        self.conv1d_transpose = layers.Conv1DTranspose(64, 3, activation="relu", strides=2, padding="same")
        self.conv1d_transpose_1 = layers.Conv1DTranspose(32, 3, activation="relu", strides=2, padding="same")
        self.conv1d_3 = layers.Conv1D(input_shape[-1], 3, activation="tanh", padding="same")  # Output same # of channels

    def encode(self, x):
        x = self.conv1d_1(x)
        x = self.conv1d_2(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling([z_mean, z_log_var])
        return z_mean, z_log_var, z

    def decode(self, z):
        x = self.dense_2(z)
        x = self.reshape(x)
        x = self.conv1d_transpose(x)
        x = self.conv1d_transpose_1(x)
        x = self.conv1d_3(x)
        return x

    def call(self, inputs):
        z_mean, z_log_var, z = self.encode(inputs)
        reconstruction = self.decode(z)
        # Add VAE loss
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        self.add_loss(kl_loss)
        return reconstruction

    def get_config(self):
      return {'input_shape': self.input_shape_val, 'latent_dim': self.latent_dim}

def create_cvae(input_shape, latent_dim):
    cvae = CVAE(input_shape, latent_dim)
    return cvae, cvae, cvae
