import tensorflow as tf
import keras
from keras import layers, Sequential

def build_model():
    model = Sequential([
            layers.Input(shape=(256, 256, 3)),

            layers.Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),

            layers.Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),

            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
    return model