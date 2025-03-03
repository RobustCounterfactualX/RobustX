from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.tasks.TaskBuilder import TaskBuilder


def test_mm_milp(testing_models):
    dl = get_example_dataset("ionosphere")
    ct = TaskBuilder().add_pytorch_model(34, [8], 1, dl).add_pytorch_model(34, [8], 1, dl).add_data(dl).build()

    ces = ct.generate(["MMMILP"])

    assert not ces["MMMILP"][0].empty

def test_mmmilp_evaluation(testing_models):
    dl = get_example_dataset("ionosphere")
    ct = TaskBuilder().add_pytorch_model(34, [8], 1, dl).add_pytorch_model(34, [8], 1, dl).add_data(dl).build()

    ces = ct.generate(["MMMILP"])
    evals = ct.evaluate(["MMMILP"], ["Distance"])

    assert evals["MMMILP"]["Distance"]