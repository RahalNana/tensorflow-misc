import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # removes warnings and info printing

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

model = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),

        layers.Conv2D(32, 3, activation='relu'),  # kernel size is assumed square if only one is specified
        # padding : valid - reduces output layer size (default padding), same - maintains layer size
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),  # default 2x2

        layers.Conv2D(128, 3, activation='relu'),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),

        layers.Dense(10)
    ]
)
print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=3e-4),
    metrics=["accuracy"]
)

model.fit(x_train, y_train, batch_size=64, epochs=10)
model.evaluate(x_test, y_test, batch_size=64)

print("COMPLETED")
