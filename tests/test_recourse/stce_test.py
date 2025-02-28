def test_stce(testing_models) -> None:
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
    res = ct.generate(["STCE"])

    assert not res["STCE"][0].empty
