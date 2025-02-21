from rocelib.tasks.TaskBuilder import TaskBuilder
from rocelib.datasets.ExampleDatasets import get_example_dataset

def test_evaluate_mm_validity_robustness() -> None:
    dl = get_example_dataset("ionosphere")
    ct = TaskBuilder().add_pytorch_model(34, [8], 1, dl).add_pytorch_model(34, [8], 1, dl).add_data(dl).build()

    recourse_methods = ["KDTreeNNCE"]
    ces = ct.generate_mm(recourse_methods)

    #TODO: debug implementation of MM metric so that it works
    evals = ct.evaluate(["KDTreeNNCE"], ["ModelMultiplicityRobustness"])

    print(evals)