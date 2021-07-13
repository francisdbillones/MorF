import sys

import tensorflow.keras as keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


from classifier import GenderClassifier
from load_data import load_data

TEST_SIZE = 1 / 3
EPOCHS = 50


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run.py file.csv")
        sys.exit(1)

    directory = sys.argv[1]

    df = load_data(directory)

    x_train, x_test, y_train, y_test = train_test_split(
        np.stack(df["name"].to_numpy()),
        np.stack(df["gender"].to_numpy()),
        test_size=TEST_SIZE,
    )

    classifier = GenderClassifier(
        x_train, y_train, input_shape=max(map(len, df["name"].values))
    )

    print(classifier.model.summary())

    history: keras.callbacks.History = classifier.train(epochs=EPOCHS)

    classifier.evaluate(x_test, np.stack(y_test))

    plot(history)

    print(
        f"Disclaimer: given training data, the model can only handle names up to length {classifier.vector_size}"
    )

    if len(sys.argv) == 3:
        output_filename = sys.argv[2]

        classifier.model.save(output_filename)
        print(f"Saved model to {output_filename}")

    start_interactive_mode(classifier)


def start_interactive_mode(classifier: GenderClassifier):
    categories = ["Male", "Female"]
    while s := input():
        if len(s) > classifier.vector_size:
            print(
                f"Name too long. Keep it less than or equal to {classifier.vector_size}"
            )
            continue
        prediction = classifier.predict(s)
        print(categories[prediction])


def plot(history: keras.callbacks.History):
    pd.DataFrame(history.history).plot()
    plt.show()


if __name__ == "__main__":
    main()
