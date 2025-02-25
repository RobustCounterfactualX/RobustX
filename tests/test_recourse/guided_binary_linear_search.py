def test_binary_linear_search_nn(testing_models) -> None:
    ct = testing_models.get("recruitment", "recruitment", "pytorch", 10, 7, 1)
    res = ct.generate(["GuidedBinaryLinearSearch"], column_name="HiringDecision")

    assert not res["GuidedBinaryLinearSearch"][0].empty