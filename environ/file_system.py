import pandas as pd


def sample_csv(file_path: str, sample_size: int, sample_state: int):  # noqa: E501
    """
        Reads the csv file and returns a dataframe.
    """
    raw_df = pd.read_csv(file_path, header=None)
    raw_df.dropna(inplace=True, axis=1)
    raw_df = raw_df.sample(
        frac=sample_size, random_state=sample_state
    )  # noqa: E501
    raw_df.reset_index()

    return raw_df
