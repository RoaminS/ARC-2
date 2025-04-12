import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, MultiHeadAttention, LayerNormalization, Reshape
from src.dsl import dsl
import os
import pickle
import random

os.makedirs("models", exist_ok=True)

class GoldenInitializer(tf.keras.initializers.Initializer):
    def __init__(self, seed=42):
        self.seed = seed
        self.phi = 1.618
        self.phi_inv = 1 / self.phi

    def __call__(self, shape, dtype=None):
        np.random.seed(self.seed)
        total_params = np.prod(shape)
        values = (np.sin(np.arange(total_params) * self.phi) + 1) / 2
        return tf.convert_to_tensor(values.reshape(shape), dtype=dtype)

def build_model(input_shape=(30, 30, 1)):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer=GoldenInitializer())(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=GoldenInitializer())(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (2, 2), activation='relu', padding='same', kernel_initializer=GoldenInitializer())(x)
    x = Conv2D(256, (2, 2), activation='relu', padding='same', kernel_initializer=GoldenInitializer())(x)
    x = Reshape((-1, 256))(x)
    x = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = LayerNormalization()(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(128, activation='relu')(x)
    return Model(inputs, outputs)

def generate_tasks(n=100):
    tasks = []
    for _ in range(n):
        grid_size = random.randint(5, 20)
        grid = np.random.randint(0, 10, (grid_size, grid_size))
        primitive = random.choice(dsl.primitives)
        out = dsl.apply(grid, primitive)
        tasks.append((grid, out))
    return tasks

def resize(grid, size=30):
    result = np.zeros((size, size))
    result[:grid.shape[0], :grid.shape[1]] = grid
    return result[..., np.newaxis] / 10.0

def main():
    model = build_model()
    model.compile(optimizer='adam', loss='mse')

    tasks = generate_tasks(1000)
    X = []
    y = []

    for inp, out in tasks:
        X.append(resize(inp))
        y.append(model.predict(np.array([resize(out)]), verbose=0)[0])

    X, y = np.array(X), np.array(y)
    model.fit(X, y, batch_size=32, epochs=5)

    model.save("models/cnn.h5")
    print("[✅] Modèle sauvegardé dans models/cnn.h5")

if __name__ == "__main__":
    main()
