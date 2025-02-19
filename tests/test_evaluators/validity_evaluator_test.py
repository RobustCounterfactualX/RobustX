from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.evaluations.ValidityEvaluator import ValidityEvaluator
from rocelib.lib.distance_functions.DistanceFunctions import manhattan
from rocelib.models.Models import get_sklearn_model
from rocelib.recourse_methods.BinaryLinearSearch import BinaryLinearSearch
from rocelib.tasks.ClassificationTask import ClassificationTask


def test_validity(testing_models):
    # assumes binarylinearsearch has 100% validity
    ct = testing_models.get("ionosphere", "ionosphere", "decision tree")
    ct.generate(["BinaryLinearSearch"])
    evals = ct.evaluate(["BinaryLinearSearch"], ["Validity"], distance_func=manhattan)
    efficacy = evals["BinaryLinearSearch"]["Validity"]
    assert efficacy == 1
