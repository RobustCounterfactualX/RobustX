from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.tasks.TaskBuilder import TaskBuilder


def test_evaluate_mm_validity_robustness() -> None:
    dl = get_example_dataset("ionosphere")
    ct = TaskBuilder().add_pytorch_model(34, [8], 1, dl).add_pytorch_model(34, [8], 1, dl).add_data(dl).build()

    recourse_methods = ["KDTreeNNCE"]
    ces = ct.generate_mm(recourse_methods)

    evals = ct.evaluate(["KDTreeNNCE"], ["ModelMultiplicityRobustness"])
    assert isinstance(evals["KDTreeNNCE"]["ModelMultiplicityRobustness"], float)

