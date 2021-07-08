import tensorflow.keras as keras
import numpy as np

from vectorizer import WordVectorizer


NONE = object()


class GenderClassifier:
    """
    Just a tiny abstraction layer.
    This exists because I feel it's going to be useful in the future.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, input_shape: int = NONE):
        self.vector_size = input_shape
        if input_shape is NONE:
            self.vector_size = max(map(len, x))
        x = WordVectorizer.vectorize_all(x, self.vector_size)

        self.x = x
        self.y = y

        self.model = GenderClassifier.initialize_model(self.vector_size)

    @staticmethod
    def initialize_model(input_shape: int):
        model = keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=(input_shape,)),
                keras.layers.Dense(128, activation=keras.activations.relu),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(128, activation=keras.activations.relu),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(1, activation=keras.activations.sigmoid),
            ]
        )

        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[keras.metrics.binary_accuracy],
        )

        return model

    def train(self, epochs=50):
        self.model.fit(self.x, self.y, epochs=epochs)

    def evaluate(self, x: np.array, y: np.array):
        x = WordVectorizer.vectorize_all(x, self.vector_size)
        self.model.evaluate(x, y)

    def predict(self, name):
        name_vector = WordVectorizer.vectorize(name, self.vector_size)
        prediction = self.model.predict(np.array([name_vector]))
        return round(prediction)

    def predict_all(self, names):
        return [self.predict(name) for name in names]
