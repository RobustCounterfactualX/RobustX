import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from rocelib.datasets.provided_datasets.ExampleDatasetLoader import ExampleDatasetLoader


class AdultDatasetLoader(ExampleDatasetLoader):

    def __init__(self):
        categoricals = [
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"
        ]
        numericals = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
        super().__init__(categoricals, numericals, missing_val_cat='?')

    @property
    def X(self) -> pd.DataFrame:
        return self._data.drop(columns=["income"])

    @property
    def y(self) -> pd.Series:
        return self._data[["income"]]

    def load_data(self):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        column_names = [
            "age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
        ]
        self._data = pd.read_csv(url, names=column_names, na_values=" ?", skipinitialspace=True)

    def get_default_preprocessed_features(self) -> pd.DataFrame:

        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical),
                ('cat', categorical_transformer, self.categorical)
            ])

        data = self._data.dropna()
        data_features = data.drop(columns=["income"])
        data_preprocessed = preprocessor.fit_transform(data_features)
        data_preprocessed_df = pd.DataFrame(data_preprocessed.toarray())

        self._data["income"] = self._data["income"].map({"<=50K": 0, ">50K": 1})
        return data_preprocessed_df
