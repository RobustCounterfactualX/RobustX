import pandas as pd
from sklearn.preprocessing import StandardScaler

from rocelib.datasets.provided_datasets.ExampleDatasetLoader import ExampleDatasetLoader


class IonosphereDatasetLoader(ExampleDatasetLoader):
    """
    A DataLoader class responsible for loading the Ionosphere dataset
    """

    def __init__(self):
        categoricals = []
        numericals = [f"feature_{i}" for i in range(34)]
        super().__init__(categoricals, numericals)

    def load_data(self):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
        column_names = self.numerical + ["target"]
        self._data = pd.read_csv(url, header=None, names=column_names)

    def get_default_preprocessed_features(self):
        # We will map the target variable here for default preprocessing
        self.data['target'] = self.data['target'].map({'g': 1, 'b': 0})

        features = self.X

        # Standardize the features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        data_preprocessed = pd.DataFrame(features_scaled, columns=features.columns)

        # Add target column to standardized features
        return data_preprocessed

    @property
    def X(self):
        return self.data.drop(columns=["target"])

    @property
    def y(self) -> pd.Series:
        return self.data[["target"]]
