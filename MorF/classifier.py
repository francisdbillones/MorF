import tensorflow.keras as keras
import numpy as np

from vectorizer import WordVectorizer


NONE = object()


class GenderClassifier:
    """
    Just a tiny abstraction layer.
    This exists because I feel it's going to be useful in the future.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.vector_size = max(map(len, x))
        x = WordVectorizer.vectorize(x, self.vector_size)

        self.x = x
        self.y = y

        self.model = GenderClassifier.initialize_model(self.vector_size)

    @staticmethod
    def initialize_model(input_shape: int):
        model = keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=(input_shape,)),
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
        distribution = self.model.predict(np.array([name]))
        return max(
            np.ndenumerate(distribution), key=lambda enumeration: enumeration[1]
        )[0][1]

    def predict_all(self, names):
        return self.model.predict(names)
