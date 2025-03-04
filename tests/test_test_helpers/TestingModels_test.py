from ..test_helpers.TestingModels import TestingModels


def test_binary_linear_search_dt() -> None:
    tm = TestingModels()

    ct1 = tm.get("ionosphere", "ionosphere", "decision tree")
    ct2 = tm.get("ionosphere", "ionosphere", "decision tree")
    # check that these are the same object (only one model trained)
    assert id(ct1.model) == id(ct2.model)
    assert id(ct1.dataset) == id(ct2.dataset)



