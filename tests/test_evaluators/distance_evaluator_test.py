from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.evaluations.DistanceEvaluator import DistanceEvaluator
from rocelib.lib.distance_functions.DistanceFunctions import manhattan
from rocelib.models.Models import get_sklearn_model
from rocelib.recourse_methods.BinaryLinearSearch import BinaryLinearSearch
from rocelib.tasks.ClassificationTask import ClassificationTask


def test_distance():
    model = get_sklearn_model("decision_tree")
    dl = get_example_dataset("ionosphere")
    trained_model = model.train(dl.X, dl.y)

    ct = ClassificationTask(trained_model, dl)

    dl.default_preprocess()


    recourse = BinaryLinearSearch(ct)

    res = recourse.generate_for_all(neg_value=0, column_name="target", distance_func=manhattan)

    dist_eval = DistanceEvaluator(ct)

    avg_dist = dist_eval.evaluate(res, distance_func=manhattan)

    mean = res["loss"].mean()

    assert avg_dist == mean
