from datasets.provided_datasets.DatasetLoader import DatasetLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd


class IonosphereDatasetLoader(DatasetLoader):
    """
    A DataLoader class responsible for loading the Ionosphere dataset
    """

    def __init__(self):
        self._data = None

    def load_data(self):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
        column_names = [f"feature_{i}" for i in range(34)] + ["target"]
        self._data = pd.read_csv(url, header=None, names=column_names)

    def get_preprocessed_features(self):

        # Map target values - good to 1 and bad to 0
        self.data['target'] = self.data['target'].map({'g': 1, 'b': 0})

        features = self.data.drop(columns=['target'])
        target = self.data['target']

        # Standardize the features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        data_preprocessed = pd.DataFrame(features_scaled, columns=features.columns)

        # Add target column to standardized features
        data_preprocessed['target'] = target.values

        return data_preprocessed

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def X(self):
        return self.data.drop(columns=["target"])

    @property
    def y(self) -> pd.Series:
        return self.data[["target"]]
