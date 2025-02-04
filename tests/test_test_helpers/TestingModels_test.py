from enums.dataset_enums import Dataset
from enums.model_enums import ModelType
from test_helpers.TestingModels import TestingModels


def test_binary_linear_search_dt() -> None:
    tm = TestingModels()

    ct1, dl1 = tm.get(Dataset.IONOSPHERE, ModelType.DECISION_TREE)
    ct2, dl2 = tm.get(Dataset.IONOSPHERE, ModelType.DECISION_TREE)

    # check that these are the same object (only one model trained)
    assert id(ct1) == id(ct2)
    assert id(dl1) == id(dl2)

