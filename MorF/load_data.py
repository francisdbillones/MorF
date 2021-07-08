import pandas as pd
import numpy as np
import tensorflow.keras as keras

import os


def load_data(directory: str):
    """
    Male names will be labelled 0,
    female names will be labelled 1,
    and androgynous names will be labelled 2.

    Returns a two-element tuple consisting of
    the features, which is a list of strings,
    and the labels, which is a list of integers.
    """
    with open(os.path.join(directory, "male_names.txt")) as reader:
        male_names = np.array([name.strip().lower() for name in reader.readlines()])

    with open(os.path.join(directory, "female_names.txt")) as reader:
        female_names = np.array([name.strip().lower() for name in reader.readlines()])

    with open(os.path.join(directory, "androgynous_names.txt")) as reader:
        andro_names = np.array([name.strip().lower() for name in reader.readlines()])

    names = np.concatenate([male_names, female_names, andro_names])

    # male-like names are categorized as 0
    male_labels = np.zeros((len(male_names),), dtype=int)

    # female-like names are categorized as 1
    female_labels = np.ones((len(female_names),), dtype=int)

    # androgynous names are categorized as 2
    andro_labels = np.full((len(andro_names),), 2, dtype=int)

    male_labels = keras.utils.to_categorical(male_labels, 3)
    female_labels = keras.utils.to_categorical(female_labels, 3)
    andro_labels = keras.utils.to_categorical(andro_labels, 3)

    labels = np.concatenate([male_labels, female_labels, andro_labels])

    return construct_dataframe(names, labels)


def construct_dataframe(names, labels):
    column_names = ["names", "labels"]

    data = [*zip(names, labels)]

    return pd.DataFrame(data=data, columns=column_names)
