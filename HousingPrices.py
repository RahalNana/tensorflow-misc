import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as tfl

DATADIR = "E:/Pycharm Projects/Datasets/HousingPrices.csv"

df = pd.read_csv(DATADIR)

x = df.drop(columns="SalePrice")
y = df[["SalePrice"]]

model = tf.keras.Sequential()

model.add(tfl.Dense(8, activation="relu", input_shape=(8,)))
model.add(tfl.Dense(8, activation="relu"))
model.add(tfl.Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error")

history = model.fit(x, y, validation_split=0.33, epochs=500)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
