import numpy as np
import string

from tensorflow.keras.utils import to_categorical

NONE = object()


class WordVectorizer:
    VOCAB = set(string.ascii_lowercase)
    UNDEFINED = to_categorical(0, len(VOCAB) + 1)

    @staticmethod
    def vectorize_all(words, shape: int = NONE) -> np.ndarray:

        if shape is NONE:
            shape = max(map(len, words))

        vectors = [WordVectorizer.vectorize(word, shape=shape) for word in words]

        return np.array(vectors)

    @staticmethod
    def vectorize(word, shape=NONE) -> np.array:
        if shape is NONE:
            shape = len(word)

        vector = np.array(
            [
                to_categorical(ord(c) - ord("a") + 1, len(WordVectorizer.VOCAB) + 1)
                for c in word.lower()
            ]
        )

        if len(word) < shape:
            vector = WordVectorizer.prepend_undefined(vector, shape)

        return vector

    @staticmethod
    def prepend_undefined(vector: np.ndarray, length: int) -> np.ndarray:
        """
        Prepends length - len(vector) zeros to the vector
        """
        return np.concatenate(
            [
                [WordVectorizer.UNDEFINED] * (length - len(vector)),
                vector,
            ]
        )
