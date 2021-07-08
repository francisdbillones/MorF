import pandas as pd
import numpy as np
import tensorflow.keras as keras

import os
import string

ALLOWED_CHARS = set(string.ascii_lowercase)


def load_data(directory: str):
    """
    Male names will be labelled 0,
    female names will be labelled 1,
    and androgynous names will be labelled 2.

    Returns a two-element tuple consisting of
    the features, which is a list of strings,
    and the labels, which is a list of integers.
    """
    filenames = ["male_names.txt", "female_names.txt"]

    names = []

    for filename in filenames:
        with open(os.path.join(directory, filename)) as reader:
            allowed_names = []

            for line in reader.readlines():
                name = line.strip().lower()
                if allowed_name(name):
                    allowed_names.append(name)

            names.append(np.array(allowed_names))

    labels = []

    for category in range(2):
        no_of_labels = len(names[category])
        category_labels = keras.utils.to_categorical(
            np.full((no_of_labels,), category, dtype=int), 2
        )
        labels.append(category_labels)

    names = np.concatenate(names)
    labels = np.concatenate(labels)

    return construct_dataframe(names, labels)


def construct_dataframe(names, labels):
    column_names = ["names", "labels"]

    data = [*zip(names, labels)]

    return pd.DataFrame(data=data, columns=column_names)


def allowed_name(name):
    return all(c.lower() in ALLOWED_CHARS for c in name)
