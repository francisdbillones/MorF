import tensorflow.keras as keras
import numpy as np

from vectorizer import WordVectorizer


NONE = object()


class GenderClassifier:
    """
    Just a tiny abstraction layer.
    This exists because I feel it's going to be useful in the future.
    """

    VALIDATION_SPLIT = 0.2

    def __init__(self, x: np.ndarray, y: np.ndarray, input_shape: int = NONE):
        self.vector_size = input_shape
        self.vocab = set(c.lower() for word in x for c in word)

        if input_shape is NONE:
            self.vector_size = max(map(len, x))
        x = WordVectorizer.vectorize_all(x, self.vector_size)

        self.x = x
        self.y = y

        self.model = self.initialize_model()

    def initialize_model(self):
        model = keras.Sequential(
            [
                keras.layers.Bidirectional(
                    keras.layers.LSTM(units=64, return_sequences=True),
                    backward_layer=keras.layers.LSTM(
                        units=64, return_sequences=True, go_backwards=True
                    ),
                    input_shape=(self.vector_size, len(self.vocab) + 1),
                ),
                keras.layers.Dropout(0.5),
                keras.layers.Bidirectional(keras.layers.LSTM(units=64)),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(units=1, activation=keras.activations.sigmoid),
            ]
        )

        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[
                keras.metrics.BinaryAccuracy(),
                keras.metrics.Precision(),
                keras.metrics.Recall(),
                keras.metrics.AUC(),
            ],
        )

        return model

    def train(self, epochs=50):
        callbacks = [
            keras.callbacks.EarlyStopping(patience=3),
            keras.callbacks.ModelCheckpoint("checkpoints/checkpoint_model.h5"),
            keras.callbacks.TensorBoard(
                log_dir="logs",
            ),
        ]
        return self.model.fit(
            self.x,
            self.y,
            validation_split=self.VALIDATION_SPLIT,
            epochs=epochs,
            callbacks=callbacks,
        )

    def evaluate(self, x: np.array, y: np.array):
        x = WordVectorizer.vectorize_all(x, self.vector_size)
        self.model.evaluate(x, y)

    def predict(self, name):
        name_vector = WordVectorizer.vectorize(name, self.vector_size)
        prediction = self.model.predict(np.array([name_vector]))[0][0]
        return round(prediction)

    def predict_all(self, names):
        return [self.predict(name) for name in names]
