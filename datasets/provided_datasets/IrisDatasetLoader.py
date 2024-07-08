from datasets.provided_datasets.DatasetLoader import DatasetLoader
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pandas as pd


class IrisDatasetLoader(DatasetLoader):
    """
    A DataLoader class responsible for loading the Iris dataset
    """

    def __init__(self):
        self._data = None

    def load_data(self):
        iris = load_iris(as_frame=True)
        self.data = iris.frame

    def get_preprocessed_features(self):
        scaler = StandardScaler()
        data_preprocessed = scaler.fit_transform(self.data.drop(columns=["target"]))
        data_preprocessed_df = pd.DataFrame(data_preprocessed, columns=self.data.drop(columns=["target"]).columns)
        return data_preprocessed_df
    
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, value):
        self._data = value

    @property
    def X(self):
        return self.data.drop(columns=["target"])

    @property
    def y(self):
        return self.data[["target"]]
