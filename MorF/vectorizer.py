import numpy as np

NONE = object()
UNDEFINED = -1


class WordVectorizer:
    @staticmethod
    def vectorize_all(words, shape: int = NONE) -> np.ndarray:

        if shape is NONE:
            shape = max(map(len, words))

        vectors = []

        for word in words:
            vector = WordVectorizer.vectorize(word, shape=shape)
            vectors.append(vector)

        return np.array(vectors)

    @staticmethod
    def vectorize(word, shape=NONE) -> np.array:
        if shape is NONE:
            shape = len(word)

        vector = np.array([ord(c) - ord("a") for c in word.lower()])

        if len(word) < shape:
            vector = WordVectorizer.prepend_undefined(vector, shape)

        return vector

    @staticmethod
    def prepend_undefined(vector: np.ndarray, length: int) -> np.ndarray:
        """
        Prepends length - len(vector) zeros to the vector
        """
        return np.concatenate([np.full((length - len(vector),), UNDEFINED), vector])