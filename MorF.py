import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import string
import sys
import os


TEST_SIZE = 1 / 3
EPOCHS = 25
VALIDATION_SPLIT = 1 / 5

VOCAB = set(string.ascii_lowercase)
MAX_LENGTH = 30

UNDEFINED = 0


def main():
    _, data_path = sys.argv

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} does not exist.")

    df = load_data(path=data_path)

    X = prepare_X(df)
    y = np.array(df["gender"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)

    model = get_model()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=3),
        keras.callbacks.TensorBoard("logs"),
    ]

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
    )

    model.evaluate(X_test, y_test)

    model.save("model")


def load_data(path="data.csv"):
    return pd.read_csv(path)


def vectorize(word):
    vector = np.array(
        [to_categorical(ord(c) - ord("a") + 1, len(VOCAB) + 1) for c in word.lower()]
    )

    vector = np.concatenate(
        [
            [to_categorical(UNDEFINED, len(VOCAB) + 1)] * (MAX_LENGTH - len(vector)),
            vector,
        ]
    )

    return vector


def prepare_X(df):
    return np.array([vectorize(word) for word in df["name"]])


def get_model():
    model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=(MAX_LENGTH, len(VOCAB) + 1)),
            keras.layers.LSTM(units=64, return_sequences=True),
            keras.layers.Dropout(0.5),
            keras.layers.LSTM(units=64),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(units=1, activation=keras.activations.sigmoid),
        ]
    )

    model.compile(
        "adam",
        "bce",
        metrics=[
            keras.metrics.BinaryAccuracy(),
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            keras.metrics.AUC(),
        ],
    )

    return model


if __name__ == "__main__":
    main()
