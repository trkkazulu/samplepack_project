import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_mlp_model(input_dim, num_classes):  #input_dim = number of features
    """Creates a simple Multi-Layer Perceptron (MLP) model."""
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))  # First hidden layer
    model.add(Dense(32, activation='relu'))   # Second hidden layer
    model.add(Dense(num_classes, activation='softmax')) # Output layer. Softmax for classification
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
