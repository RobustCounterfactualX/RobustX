import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from rocelib.datasets.provided_datasets.ExampleDatasetLoader import ExampleDatasetLoader


class TitanicDatasetLoader(ExampleDatasetLoader):

    def __init__(self):
        categoricals = ["Pclass", "Sex", "Embarked", "Cabin"]
        numericals = ["Age", "SibSp", "Parch", "Fare"]
        super().__init__(categoricals, numericals)

    @property
    def X(self) -> pd.DataFrame:
        return self.data.drop(columns=["Survived"])

    @property
    def y(self) -> pd.Series:
        return self.data["Survived"]

    def load_data(self):
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        self._data = pd.read_csv(url)

    def get_default_preprocessed_features(self) -> pd.DataFrame:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical),
                ('cat', categorical_transformer, self.categorical)
            ])

        # Impute and preprocess the data
        data_features = self._data.drop(columns=["Survived"])

        # Log the presence of NaNs before preprocessing
        print("NaNs before preprocessing:")
        print(data_features.isna().sum())

        data_preprocessed = preprocessor.fit_transform(data_features)

        # Ensure that the output is a dense array
        if isinstance(data_preprocessed, np.ndarray):
            data_preprocessed = pd.DataFrame(data_preprocessed,
                                             columns=self.get_feature_names(preprocessor, self.categorical,
                                                                            self.numerical))

        # Log the presence of NaNs after preprocessing
        print("NaNs after preprocessing:")
        print(pd.DataFrame(data_preprocessed).isna().sum())

        return pd.DataFrame.sparse.from_spmatrix(data_preprocessed)

    def get_feature_names(self, preprocessor, categorical_features, numerical_features):
        categorical_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(
            categorical_features)
        all_feature_names = list(numerical_features) + list(categorical_names)
        return all_feature_names
