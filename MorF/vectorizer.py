import numpy as np

NONE = object()


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
