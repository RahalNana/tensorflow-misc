import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # removes warnings and info printing

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

# x_train = tf.convert_to_tensor(x_train)  # done internally from numpy arrays

# SEQUENTIAL API ###########################################################################
# very convenient, not very flexible

model = keras.Sequential(
    [
        keras.Input(shape=(28 * 28)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10)
    ]
)

################# Alternative definition
model = keras.Sequential()
model.add(keras.Input(shape=(28*28)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu', name='layer2'))
model.add(layers.Dense(10))

# print(model.summary())

################## Debugging options

# model = keras.Model(inputs=model.inputs,
#                     outputs=[model.layers[-1].output])
# model = keras.Model(inputs=model.inputs,
#                     outputs=[model.get_layer('layer2').output])
# feature = model.predict(x_train)
# print(feature.shape)

# model = keras.Model(inputs=model.inputs,
#                     outputs=[layer.output for layer in model.layers])
# features = model.predict(x_train)
# for feature in features:
#     print(feature.shape)

model.compile(  # specifies network configurations
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # from_logits true when no softmax in last layer
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)

model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
