import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # removes warnings and info printing

import numpy as np
import tensorflow as tf

w = tf.Variable(0, dtype=tf.float32)
optimizer = tf.keras.optimizers.Adam(0.1)

x = np.array([1, -10, 25], dtype=np.float32)


def costFn():
    return x[0]*w**2 + x[1]*w + x[2]


print(w)
for i in range(1000):
    optimizer.minimize(costFn, [w])
print(w)
