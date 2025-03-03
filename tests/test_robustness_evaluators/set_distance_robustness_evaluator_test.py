def test_set_distance_evaluator(testing_models) -> None:
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
    res = ct.generate(["RNCE"])
    evals = ct.evaluate(["RNCE"], ["SetDistanceRobustnessEvaluator"])
    assert isinstance(evals["RNCE"]["SetDistanceRobustnessEvaluator"], float)
   