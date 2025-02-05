import numpy as np
import pandas as pd
import pytest

from enums.dataset_enums import Dataset
from enums.model_enums import ModelType
from rocelib.recourse_methods.BinaryLinearSearch import BinaryLinearSearch


def test_binary_linear_search_nn(testing_models) -> None:
    ct, _, _ = testing_models.get(Dataset.RECRUITMENT, ModelType.NEURALNET, 10, 7, 1)

    # Use BinaryLinearSearch to generate a recourse for each negative value
    recourse = BinaryLinearSearch(ct)

    res = recourse.generate_for_all(neg_value=0, column_name="HiringDecision")
    print(res)

    assert not res.empty


def test_binary_linear_search_dt(testing_models) -> None:
    ct, _, _ = testing_models.get(Dataset.IONOSPHERE, ModelType.DECISION_TREE)

    recourse = BinaryLinearSearch(ct)
    res = recourse.generate_for_all(neg_value=0, column_name="target")

    assert not res.empty


def test_binary_linear_search_lr(testing_models) -> None:
    ct, _, _ = testing_models.get(Dataset.IONOSPHERE, ModelType.LOGISTIC_REGRESSION)

    def euclidean_copy(x: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
        return np.sqrt(np.sum((x.values - c.values) ** 2))

    recourse = BinaryLinearSearch(ct, custom_distance_func=euclidean_copy)

    res = recourse.generate_for_all(neg_value=0, column_name="target")

    assert not res.empty
