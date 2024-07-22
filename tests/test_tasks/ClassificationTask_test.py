from datasets.ExampleDatasets import get_example_dataset
from datasets.custom_datasets.CsvDatasetLoader import CsvDatasetLoader
from models.Models import get_sklearn_model
from tasks.ClassificationTask import ClassificationTask


def test_classification_problem():
    model = get_sklearn_model("log_reg")
    dl = get_example_dataset("ionosphere")
    problem = ClassificationTask(model=model, training_data=dl)

    problem.default_preprocess()
    problem.train()

    print()
    print()
    print(problem.generate_counterfactuals("binary-linear-search", 0, gamma=1))


def test_classification_problem_2():
    model = get_sklearn_model("log_reg")
    dl = get_example_dataset("titanic")
    problem = ClassificationTask(model=model, training_data=dl)

    problem.default_preprocess()
    problem.train()

    print()
    print()
    print(problem.generate_counterfactuals("binary-linear-search", 0, column_name="Survived", gamma=1))


def test_classification_problem_3():
    model = get_sklearn_model("log_reg")
    dl = CsvDatasetLoader("./assets/recruitment_data.csv", "HiringDecision")
    problem = ClassificationTask(model=model, training_data=dl)

    problem.train()

    print()
    print()
    print(problem.generate_counterfactuals("binary-linear-search", 0, column_name="HiringDecision", gamma=1))
