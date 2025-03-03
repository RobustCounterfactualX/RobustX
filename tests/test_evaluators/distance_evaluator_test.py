from rocelib.lib.distance_functions.DistanceFunctions import manhattan


def test_distance(testing_models):
    ct = testing_models.get("ionosphere", "ionosphere", "decision tree")
    ct.generate(["BinaryLinearSearch"])
    evals = ct.evaluate(["BinaryLinearSearch"], ["Distance"], distance_func=manhattan)
    avg_dist = evals["BinaryLinearSearch"]["Distance"]
    assert avg_dist > 5
