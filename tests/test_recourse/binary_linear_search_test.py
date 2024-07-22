from datasets.ExampleDatasets import get_example_dataset
from datasets.custom_datasets.CsvDatasetLoader import CsvDatasetLoader
from models.Models import get_sklearn_model
from models.pytorch_models.SimpleNNModel import SimpleNNModel
from models.sklearn_models.DecisionTreeModel import DecisionTreeModel
from recourse_methods.RecourseMethods import generate_counterfactuals_binary_linear_search
from tasks.ClassificationTask import ClassificationTask


def test_binary_linear_search_nn() -> None:
    model = SimpleNNModel(10, 7, 1)
    dl = CsvDatasetLoader('./assets/recruitment_data.csv', "HiringDecision")

    ct = ClassificationTask(model, dl)

    ct.train()

    for _, negative in ct.training_data.get_negative_instances(0, column_name="HiringDecision").iterrows():
        ce = generate_counterfactuals_binary_linear_search(
            ct.model,
            negative,
            ct.get_random_positive_instance(
                0,
                column_name="HiringDecision"
            ),
            column_name="HiringDecision"
        )
        print(ce)

    assert True


def test_binary_linear_search_dt() -> None:

    model = get_sklearn_model("decision_tree")
    dl = get_example_dataset("ionosphere")

    ct = ClassificationTask(model, dl)

    ct.default_preprocess()
    ct.train()

    for _, negative in ct.training_data.get_negative_instances(0, column_name="target").iterrows():
        ce = generate_counterfactuals_binary_linear_search(
            ct.model,
            negative,
            ct.get_random_positive_instance(
                0,
                column_name="target"
            ),
            column_name="target"
        )
        print(ce)

    assert True

def test_binary_linear_search_lr() -> None:

    model = get_sklearn_model("log_reg")
    dl = get_example_dataset("ionosphere")

    ct = ClassificationTask(model, dl)

    ct.default_preprocess()
    ct.train()

    for _, negative in ct.training_data.get_negative_instances(0, column_name="target").iterrows():
        ce = generate_counterfactuals_binary_linear_search(
            ct.model,
            negative,
            ct.get_random_positive_instance(
                0,
                column_name="target"
            ),
            column_name="target"
        )
        print(ce)

    assert True
