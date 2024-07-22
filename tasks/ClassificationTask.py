from datasets.DatasetLoader import DatasetLoader
from datasets.provided_datasets.ExampleDatasetLoader import ExampleDatasetLoader
from models.BaseModel import BaseModel
from recourse_methods.RecourseMethods import generate_counterfactuals_binary_linear_search, compute_nnce
import pandas as pd


class ClassificationTask:

    def __init__(self, model: BaseModel, training_data: DatasetLoader):
        self._model = model
        self._training_data = training_data

    def train(self):
        self._model.train(self._training_data.X, self._training_data.y)

    def default_preprocess(self):
        if isinstance(self._training_data, ExampleDatasetLoader):
            self._training_data.default_preprocess()
        else:
            raise ValueError()

    def get_random_positive_instance(self, neg_value, column_name="target") -> pd.Series:
        pos_instance = self._training_data.get_random_positive_instance(neg_value, column_name=column_name)
        while self._model.predict_single(pos_instance) == neg_value:
            pos_instance = self._training_data.get_random_positive_instance(neg_value, column_name=column_name)
        return pos_instance

    @property
    def model(self):
        return self._model

    @property
    def training_data(self):
        return self._training_data

    def generate_counterfactuals(self, method: str, neg_value, column_name="target", gamma=0.1,
                                 distance_func="euclidean"):

        negatives = self.training_data.get_negative_instances(neg_value, column_name=column_name)

        print("The negative instances are:\n")
        print(negatives)

        cs = []

        for _, negative in negatives.iterrows():

            if method == "nnce":
                ct = compute_nnce(
                    self.model,
                    self.training_data,
                    negative,
                    distance_func
                )
            elif method == "binary-linear-search":
                ct = generate_counterfactuals_binary_linear_search(
                    self.model,
                    negative,
                    self.get_random_positive_instance(
                        neg_value,
                        column_name=column_name
                    ),
                    column_name,
                    gamma
                )
            else:
                raise ValueError("Not a possible counterfactual method")
            cs.append(ct)

        res = pd.concat(cs)
        return res
