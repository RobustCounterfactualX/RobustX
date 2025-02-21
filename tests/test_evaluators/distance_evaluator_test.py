from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.evaluations.DistanceEvaluator import DistanceEvaluator
from rocelib.lib.distance_functions.DistanceFunctions import manhattan
from rocelib.models.Models import get_sklearn_model
from rocelib.recourse_methods.BinaryLinearSearch import BinaryLinearSearch
from rocelib.tasks.ClassificationTask import ClassificationTask


def test_distance(testing_models):
    ct = testing_models.get("ionosphere", "ionosphere", "decision tree")
    ct.generate(["BinaryLinearSearch"])
    evals = ct.evaluate(["BinaryLinearSearch"], ["Distance"], distance_func=manhattan)
    avg_dist = evals["BinaryLinearSearch"]["Distance"]
    assert avg_dist > 5
