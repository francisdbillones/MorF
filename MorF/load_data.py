import pandas as pd

import string

ALLOWED_CHARS = set(string.ascii_lowercase)


def load_data(path_to_data: str):
    """
    Return a DataFrame in which all the rows are valid.
    """
    df = pd.read_csv(path_to_data)

    # filter names that are allowed
    df = df.loc[df["name"].apply(allowed_name)]

    # filter where the gender value is either 0 or 1
    df = df.loc[df["gender"].isin({0, 1})]

    return df


def allowed_name(name):
    return all(c.lower() in ALLOWED_CHARS for c in name)
