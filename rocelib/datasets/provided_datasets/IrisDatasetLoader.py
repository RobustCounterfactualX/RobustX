import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from rocelib.datasets.provided_datasets.ExampleDatasetLoader import ExampleDatasetLoader


class IrisDatasetLoader(ExampleDatasetLoader):
    """
    A DataLoader class responsible for loading the Iris dataset
    """

    def __init__(self):
        categoricals = []
        numericals = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                      'petal width (cm)']
        super().__init__(categoricals, numericals)

    def load_data(self):
        iris = load_iris(as_frame=True)
        self.data = iris.frame

    def get_default_preprocessed_features(self):
        scaler = StandardScaler()
        data_preprocessed = scaler.fit_transform(self.data.drop(columns=["target"]))
        data_preprocessed_df = pd.DataFrame(data_preprocessed, columns=self.numerical)
        return data_preprocessed_df

    @property
    def X(self):
        return self.data.drop(columns=["target"])

    @property
    def y(self):
        return self.data[["target"]]
