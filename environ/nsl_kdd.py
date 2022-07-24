import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .file_system import sample_csv

# CONSTANTS
BASE_RELATIVE_DATA_PATH = "../DataSets/NSL-KDD/Small_Splits/"
SAMPLE_SIZE = 0.2
RANDOM_SAMPLING_STATE = 200
IP_PROTOCOL_COLUMN = 1
SERVICE_COLUMN = 2
FLAG_COLUMN = 3
CLASSIFICATION_COLUMN = 38
ADJUSTED_CLASSIFICATION_COLUMN = 41


class nsl_kdd_env:
    def __init__(self, reward, child_id):
        """
            Initialize the environment.
        """
        self.reward = reward
        self.child_id = child_id
        self.file_path = self.get_file_path()
        self.indexer = 0

    def setup(self):
        """
            Setup the environment.
            kept seperate from init to allow for cleaner testing.
        """
        self.generate_dataframes()
        self.total_record_count = self.observations.shape[0] - 1

    def get_file_path(self):
        """
            Get the file path for the environment data..
        """
        return BASE_RELATIVE_DATA_PATH + "/KDDTrain+_{}.txt".format(
            self.child_id
        )  # noqa: E501

    def generate_dataframes(self):
        """
            Generates the lookup and observations dataframes.
        """

        raw_df = sample_csv(self.file_path, SAMPLE_SIZE, RANDOM_SAMPLING_STATE)
        self.lookup_df = self.preprocess_data(raw_df)

        self.observations = self.lookup_df.drop(
            [ADJUSTED_CLASSIFICATION_COLUMN],
            axis=1)

    def preprocess_data(self, raw_df):
        """
            Preprocesses the dataframe to be used in the environment.
        """

        one_hot_ip_protocol = pd.get_dummies(raw_df[IP_PROTOCOL_COLUMN])
        label_encoder = LabelEncoder()
        lookup = pd.concat([raw_df, one_hot_ip_protocol], axis=1)
        lookup["protocol"] = label_encoder.fit_transform(
            lookup[SERVICE_COLUMN])  # noqa: E501

        lookup["flag"] = label_encoder.fit_transform(
            lookup[FLAG_COLUMN])  # noqa: E501

        lookup = lookup.drop([
            1,
            SERVICE_COLUMN,
            FLAG_COLUMN
            ], axis=1)

        return lookup

    def reset(self):
        """
            Reset the environment.
        """

        self.indexer = 0
        return self.observations.iloc[[self.indexer]].to_numpy()

    def step(self, action):
        """
            Returns the next record and returns a reward for the given action.
        """
        raw = self.lookup_df.iloc[self.indexer, [CLASSIFICATION_COLUMN]].values

        is_normal_record = str(raw[0]) == "normal"

        # kept for sanity testing only.
        # print(
        #     "raw ", raw,
        #     " action ", bool(action),
        #     " is_normal_record ", bool(is_normal_record)
        #     )

        step_reward = (
            self.reward if bool(action) == is_normal_record
            else -self.reward
        )  # noqa: E501

        self.indexer += 1
        done = self.indexer > self.total_record_count

        obs = (
            self.observations.iloc[[self.indexer]]
            if done is False
            else self.observations.iloc[[self.indexer - 1]]
        ).to_numpy()

        info = str(raw[0])

        return obs, step_reward, done, info

    def get_observation_at_row(self, idx):
        """
            Returns the observation at the given row.
        """

        if self.total_record_count > idx:
            return self.observations.iloc[[idx]]

        return self.observations.iloc[[self.total_record_count - 1]].to_numpy()

    def get_observation_space(self):
        """
            Returns the observation space of the environment.
        """

        return self.observations.shape[1]

    def get_total_record_count(self):
        """
            Returns the total number of records in the environment.
        """

        return self.total_record_count
