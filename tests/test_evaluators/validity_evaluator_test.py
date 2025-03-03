from rocelib.lib.distance_functions.DistanceFunctions import manhattan


def test_validity(testing_models):
    # assumes binarylinearsearch has 100% validity
    ct = testing_models.get("ionosphere", "ionosphere", "decision tree")
    ct.generate(["BinaryLinearSearch"])
    evals = ct.evaluate(["BinaryLinearSearch"], ["Validity"], distance_func=manhattan)
    efficacy = evals["BinaryLinearSearch"]["Validity"]
    assert efficacy == 1
