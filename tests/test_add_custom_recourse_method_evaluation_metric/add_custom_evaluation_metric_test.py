from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.evaluations.robustness_evaluations.Evaluator import Evaluator
from rocelib.evaluations.robustness_evaluations.ModelChangesRobustnessEvaluator import ModelChangesRobustnessEvaluator
from rocelib.tasks.TaskBuilder import TaskBuilder


class NewEvaluator(Evaluator):
    def evaluate(self, recourse_method, **kwargs):
        return -1

class NewMCEvaluator(ModelChangesRobustnessEvaluator):
    def evaluate_single_instance(self, instance, counterfactual, **kwargs):
        return True


def test_metric_gets_added_to_list_of_metrics():
    dl = get_example_dataset("ionosphere")
    # Not using testing models as we will be adding a recourse method to the task so could mess up other tests
    ct = TaskBuilder().add_pytorch_model(34, [8], 1, dl).add_data(dl).build()

    ct.add_evaluation_metric("new_metric", NewEvaluator)
    assert "new_metric" in ct.get_evaluation_metrics()

def test_ce_evaluation_for_new_metric():
    dl = get_example_dataset("ionosphere")
    # Not using testing models as we will be adding a recourse method to the task so could mess up other tests
    ct = TaskBuilder().add_pytorch_model(34, [8], 1, dl).add_data(dl).build()

    ct.add_evaluation_metric("new_metric", NewEvaluator)
    ces = ct.generate(["MCE"])
    evals = ct.evaluate(["MCE"], ["new_metric"])

    expected_metric_result = -1
    assert evals["MCE"]["new_metric"] == expected_metric_result

def test_ce_evaluation_for_new_robustness_metric():
    dl = get_example_dataset("ionosphere")
    # Not using testing models as we will be adding a recourse method to the task so could mess up other tests
    ct = TaskBuilder().add_pytorch_model(34, [8], 1, dl).add_data(dl).build()

    ct.add_evaluation_metric("new_mc_metric", NewMCEvaluator)
    ces = ct.generate(["MCE"])
    evals = ct.evaluate(["MCE"], ["new_mc_metric"])

    #Check all eval results are True
    assert evals["MCE"]["new_mc_metric"] == 1