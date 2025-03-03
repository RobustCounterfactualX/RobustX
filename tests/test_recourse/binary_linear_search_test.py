import numpy as np
import pandas as pd


def test_binary_linear_search_nn(testing_models) -> None:
    ct = testing_models.get("recruitment", "recruitment", "pytorch", 10, 7, 1)
    res = ct.generate(["BinaryLinearSearch"], column_name="HiringDecision")

    assert not res["BinaryLinearSearch"][0].empty


def test_binary_linear_search_dt(testing_models) -> None:
    ct = testing_models.get("ionosphere", "ionosphere", "decision tree")

    res = ct.generate(["BinaryLinearSearch"])

    assert not res["BinaryLinearSearch"][0].empty


def test_binary_linear_search_lr(testing_models) -> None:
    ct = testing_models.get("ionosphere", "ionosphere", "logistic regression")

    def euclidean_copy(x: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
        return np.sqrt(np.sum((x.values - c.values) ** 2))
    res = ct.generate(["BinaryLinearSearch"], custom_distance_func=euclidean_copy)

    assert not res["BinaryLinearSearch"][0].empty

