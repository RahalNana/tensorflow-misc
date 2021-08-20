import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # removes warnings and info printing

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations, regularizers
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

## MODEL CREATION

inputs = keras.Input(shape=(32, 32, 3))

x = layers.Conv2D(32, 3, padding='same',
                  kernel_regularizer=regularizers.l2(0.01))(inputs)
x = layers.BatchNormalization()(x)
x = activations.relu(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(64, 3, padding='same',
                  kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.BatchNormalization()(x)
x = activations.relu(x)

x = layers.Conv2D(128, 3, padding='same',
                  kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.BatchNormalization()(x)
x = activations.relu(x)

x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu',
                 kernel_regularizer=regularizers.l2(0.01))(x)

x = layers.Dropout(0.5)(x)

outputs = layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=3e-4),
    metrics=["accuracy"]
)

model.fit(x_train, y_train, batch_size=64, epochs=20)
model.evaluate(x_test, y_test, batch_size=64)

print("COMPLETED")
