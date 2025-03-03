def test_mce_predicts_positive_instances(testing_models):
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
    res = ct.generate(["MCE"])

    assert not res["MCE"][0].empty

