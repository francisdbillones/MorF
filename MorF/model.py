import tensorflow.keras as keras
import numpy as np

NONE = object()


class GenderClassifier:
    """
    Just a tiny abstraction layer over the Keras API.
    This exists because I feel it's going to be useful in the future.
    """

    def __init__(self, x, y):
        self.vector_size = max(map(len, x))
        x = self.vectorize(x, self.vector_size)

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
        x = self.vectorize(x, self.vector_size)
        self.model.evaluate(x, y)

    def predict(self, name):
        name = WordVectorizer.vectorize_word(name, self.vector_size)
        return self.model.predict([name])

    def predict_all(self, names):
        return self.model.predict(names)

    def vectorize(self, x, vector_size):
        return WordVectorizer.vectorize(x, shape=vector_size)


class WordVectorizer:
    @staticmethod
    def vectorize(words, shape: int = NONE) -> np.ndarray:

        if shape is NONE:
            shape = max(map(len, words))

        vectors = []

        for word in words:
            vector = WordVectorizer.vectorize_word(word, shape=shape)
            vectors.append(vector)

        return np.array(vectors)

    @staticmethod
    def vectorize_word(word, shape=NONE) -> np.array:
        if shape is NONE:
            shape = len(word)

        vector = np.array([ord(c) for c in word])

        if len(word) < shape:
            vector = WordVectorizer.prepend_zeroes(vector, shape)

        return vector

    @staticmethod
    def prepend_zeroes(vector: np.ndarray, length: int) -> np.ndarray:
        """
        Prepends length - len(vector) zeros to the vector
        """
        return np.concatenate([np.zeros(length - len(vector)), vector])
