from datasets.ExampleDatasets import get_example_dataset
from evaluations.ValidityEvaluator import ValidityEvaluator
from lib.distance_functions.DistanceFunctions import manhattan
from models.Models import get_sklearn_model
from recourse_methods.BinaryLinearSearch import BinaryLinearSearch
from tasks.ClassificationTask import ClassificationTask


def test_validity():
    model = get_sklearn_model("decision_tree")
    dl = get_example_dataset("ionosphere")

    ct = ClassificationTask(model, dl)

    dl.default_preprocess()
    ct.train()

    recourse = BinaryLinearSearch(ct)

    res = recourse.generate_for_all(neg_value=0, column_name="target", distance_func=manhattan)

    val_eval = ValidityEvaluator(ct)

    efficacy = val_eval.evaluate(res)

    print(f"Valid: {efficacy}")

    assert efficacy == 1
