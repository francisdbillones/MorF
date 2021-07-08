import sys

from sklearn.model_selection import train_test_split

from classifier import GenderClassifier
from load_data import load_data

TEST_SIZE = 1 / 3
EPOCHS = 10


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run.py directory")
        sys.exit(1)

    directory = sys.argv[1]

    evidence, labels = load_data(directory)

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


def start_interactive_mode(classifier: GenderClassifier):
    categories = ["Male", "Female", "Androgynous"]
    while s := input():
        prediction = max(
            enumerate(classifier.predict(s)), key=lambda enumeration: enumeration[1]
        )[0]
        print(categories[prediction])


if __name__ == "__main__":
    main()
