from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.evaluations.ValidityEvaluator import ValidityEvaluator
from rocelib.lib.distance_functions.DistanceFunctions import manhattan
from rocelib.models.Models import get_sklearn_model
from rocelib.recourse_methods.BinaryLinearSearch import BinaryLinearSearch
from rocelib.tasks.ClassificationTask import ClassificationTask


def test_validity():
    model = get_sklearn_model("decision_tree")
    dl = get_example_dataset("ionosphere")

    dl.default_preprocess()
    trained_model = model.train(dl.X, dl.y)

    ct = ClassificationTask(trained_model, dl)



    recourse = BinaryLinearSearch(ct)

    res = recourse.generate_for_all(neg_value=0, column_name="target", distance_func=manhattan)

    val_eval = ValidityEvaluator(ct)

    efficacy = val_eval.evaluate(res)

    print(f"Valid: {efficacy}")

    assert efficacy == 1
