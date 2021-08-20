import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
from collections import deque
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

DATADIR = "E:/Pycharm Projects/Datasets/crypto_data"

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = "LTC-USD"


def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


def preprocess_df(df):
    df = df.drop('future', 1)

    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for row in df.values:
        prev_days.append([val for val in row[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), row[-1]])

    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))
    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells
    random.shuffle(sequential_data)

    X = []
    Y = []

    for seq, target in sequential_data:
        X.append(seq)
        Y.append(target)

    return np.array(X), np.array(Y)


main_df = pd.DataFrame()

ratios = ["BTC-USD", "LTC-USD", "ETH-USD", "BCH-USD"]

for ratio in ratios:
    dataset = DATADIR + "/" + ratio + ".csv"
    df = pd.read_csv(dataset, names=["time", "low", "high", "open", "close", "volume"])
    df.rename(columns={"close": ratio + "_close", "volume": ratio + "_volume"}, inplace=True)
    df.set_index("time", inplace=True)
    df = df[[ratio+"_close", ratio+"_volume"]]

    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

main_df.fillna(method="ffill", inplace=True)
main_df.dropna(inplace=True)

main_df["future"] = main_df[RATIO_TO_PREDICT+"_close"].shift(-FUTURE_PERIOD_PREDICT)
main_df["target"] = list(map(classify, main_df[RATIO_TO_PREDICT+"_close"], main_df["future"]))

main_df.dropna(inplace=True)

times = sorted(main_df.index.values)
last_5pct = times[-int(0.05*len(times))]

validation_main_df = main_df[main_df.index >= last_5pct]
main_df = main_df[main_df.index < last_5pct]

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

model = Sequential()

model.add(tfl.LSTM(128, input_shape=(train_x.shape[1:]), activation="tanh", return_sequences=True))
model.add(tfl.Dropout(0.2))
model.add(tfl.BatchNormalization())

model.add(tfl.LSTM(128, return_sequences=True, activation="tanh"))
model.add(tfl.Dropout(0.2))
model.add(tfl.BatchNormalization())

model.add(tfl.LSTM(128, activation="tanh"))
model.add(tfl.Dropout(0.2))
model.add(tfl.BatchNormalization())

model.add(tfl.Dense(32, activation="relu"))
model.add(tfl.Dropout(0.2))

model.add(tfl.Dense(2, activation="softmax"))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=1e-6),
              metrics=["accuracy"])

model.fit(train_x, np.array(train_y), batch_size=64, epochs=10, validation_data=(validation_x, validation_y))
