from datasets.ExampleDatasets import get_example_dataset
from evaluations.DistanceEvaluator import DistanceEvaluator
from models.Models import get_sklearn_model
from recourse_methods.BinaryLinearSearch import BinaryLinearSearch
from tasks.ClassificationTask import ClassificationTask


def test_distance():
    model = get_sklearn_model("decision_tree")
    dl = get_example_dataset("ionosphere")

    ct = ClassificationTask(model, dl)

    dl.default_preprocess()
    ct.train()

    recourse = BinaryLinearSearch(ct)

    res = recourse.generate_for_all(neg_value=0, column_name="target", distance_func="l1")

    dist_eval = DistanceEvaluator(ct)

    avg_dist = dist_eval.evaluate(res)

    assert avg_dist
