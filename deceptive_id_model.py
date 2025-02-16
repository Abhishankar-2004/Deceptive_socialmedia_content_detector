

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define a simple model (for illustration)
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Example of creating dummy data for training
import numpy as np
X_train = np.random.rand(100, 10)  # 100 samples, 10 features
y_train = np.random.randint(2, size=100)  # Binary labels

# Train the model
model.fit(X_train, y_train, epochs=10)

# Now save the model
model.save("deceptive_id_model.keras")
