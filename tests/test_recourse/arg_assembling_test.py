def test_arg_assembling_predicts_positive_instances(testing_models):
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
    res = ct.generate(["ArgEnsembling"])

    assert not res["ArgEnsembling"][0].empty