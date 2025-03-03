def test_rocoursenet(testing_models) -> None:
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
    res = ct.generate(["RoCourseNet"])

    assert not res["RoCourseNet"][0].empty