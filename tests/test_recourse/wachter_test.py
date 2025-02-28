def test_wachter(testing_models) -> None:
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)

    res = ct.generate(["Wachter"])

    assert not res["Wachter"][0].empty


