
def test_apas(testing_models) -> None:
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)

    res = ct.generate(["APAS"])

    assert not res["APAS"][0].empty


