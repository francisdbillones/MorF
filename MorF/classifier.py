import tensorflow.keras as keras
import numpy as np

from typing import List

from vectorizer import WordVectorizer


NONE = object()


class GenderClassifier:
    """
    Just a tiny abstraction layer.
    This exists because I feel it's going to be useful in the future.
    """

    def __init__(self, x: List[str], y: List[int]):
        self.vector_size = max(map(len, x))
        x = WordVectorizer.vectorize(x, self.vector_size)

        self.x = x
        self.y = y

        self.model = GenderClassifier.initialize_model(self.vector_size)

    @staticmethod
    def initialize_model(input_shape: int):
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(input_shape,)),
                keras.layers.Dense(64, activation=keras.activations.relu),
                keras.layers.Dropout(0.1),
                keras.layers.Dense(64, activation=keras.activations.relu),
                keras.layers.Dropout(0.1),
                keras.layers.Dense(3, activation=keras.activations.softmax),
            ]
        )

        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=[keras.metrics.categorical_accuracy],
        )

        return model

    def train(self, epochs=50):
        self.model.fit(self.x, self.y, epochs=epochs)

    def evaluate(self, x: np.array, y: np.array):
        x = WordVectorizer.vectorize(x, self.vector_size)
        self.model.evaluate(x, y)

    def predict(self, name):
        name = WordVectorizer.vectorize_word(name, self.vector_size)
        return self.model.predict([name])

    def predict_all(self, names):
        return self.model.predict(names)
