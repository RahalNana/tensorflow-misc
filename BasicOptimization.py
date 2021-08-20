import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # removes warnings and info printing

import numpy as np
import tensorflow as tf

w = tf.Variable(0, dtype=tf.float32)
optimizer = tf.keras.optimizers.Adam(0.1)


def trainStep():
    with tf.GradientTape() as tape:
        cost = w**2-10*w+25
    trainableVariables = [w]
    grads = tape.gradient(cost, trainableVariables)
    optimizer.apply_gradients(zip(grads, trainableVariables))


print(w)

for i in range(1000):
    trainStep()
print(w)
