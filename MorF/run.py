import sys
import os

import tensorflow.keras as keras

import numpy as np

from sklearn.model_selection import train_test_split

from model import GenderClassifier

TEST_SIZE = 1 / 3
EPOCHS = 10


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run.py directory")
        sys.exit(1)

    directory = sys.argv[1]

    evidence, labels, vector_size = load_data(directory)

    x_train, x_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    classifier = GenderClassifier(x_train, y_train)

    classifier.train(epochs=EPOCHS)

    classifier.evaluate(x_test, y_test)

    if len(sys.argv) == 3:
        output_filename = sys.argv[2]

        classifier.model.save(output_filename)
        print(f"Saved model to {output_filename}")

    start_interactive_mode(classifier)


def load_data(directory):
    """
    Male names will be labelled 0,
    while female names will be labelled 1.
    """
    print("Loading data...")
    with open(os.path.join(directory, "male_names.txt")) as reader:
        male_names = [name.strip() for name in reader.readlines()]

    with open(os.path.join(directory, "female_names.txt")) as reader:
        female_names = [name.strip() for name in reader.readlines()]

    with open(os.path.join(directory, "androgynous_names.txt")) as reader:
        andro_names = [name.strip() for name in reader.readlines()]

    vector_size = max(
        max(map(len, male_names)),
        max(map(len, female_names)),
        max(map(len, andro_names)),
    )

    # male-like names are categorized as 0
    male_labels = np.zeros((len(male_names),), dtype=int)

    # female-like names are categorized as 1
    female_labels = np.ones((len(female_names),), dtype=int)

    # androgynous names are categorized as 2
    andro_labels = np.full((len(andro_names),), 2, dtype=int)

    male_labels = keras.utils.to_categorical(male_labels, 3)
    female_labels = keras.utils.to_categorical(female_labels, 3)
    andro_labels = keras.utils.to_categorical(andro_labels, 3)

    return (
        np.concatenate([male_names, female_names, andro_names]),
        np.concatenate([male_labels, female_labels, andro_labels]),
        vector_size,
    )


def start_interactive_mode(classifier: GenderClassifier):
    categories = ["Male", "Female", "Androgynous"]
    while s := input():
        prediction = classifier.predict(s)
        print(categories[prediction])


if __name__ == "__main__":
    main()
