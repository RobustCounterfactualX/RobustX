import pandas as pd

from rocelib.evaluations.robustness_evaluations.MC_Robustness_Implementations.VaRRobustnessEvaluator import \
    VaRRobustnessEvaluator
from rocelib.tasks.Task import Task


# Mock model class that always returns a fixed prediction
class MockModel:
    def __init__(self, output):
        self.output = output

    def predict_single(self, instance):
        return self.output


# Mock dataset class to simulate a dataset object
class MockDataset:
    def __init__(self, data):
        self.data = data


def test_var_robustness_simple():
    # Create a mock dataset
    data = pd.DataFrame({"feature1": [0.5, 0.2], "feature2": [0.7, 0.1]})

    # Create a mock model that always predicts `1`
    mock_model = MockModel(output=1)

    # Initialize Task with both model and dataset
    mock_task = Task(mock_model, MockDataset(data))

    # Corrected: Modify `_CEs` directly instead of using `CEs`
    mock_task._CEs = {"some_method": [data]}

    # Create retrained models that always predict `1` (same as original)
    retrained_models = [MockModel(output=1), MockModel(output=1)]

    # Initialize robustness evaluator
    var_evaluator = VaRRobustnessEvaluator(mock_task, retrained_models)

    # Test instance 0 (should be robust)
    assert var_evaluator.evaluate_single_instance(0, recourse_method="some_method", desired_output=1) is True

    # Change retrained models to predict `0` (should make robustness fail)
    retrained_models = [MockModel(output=0), MockModel(output=1)]
    var_evaluator = VaRRobustnessEvaluator(mock_task, retrained_models)

    # Test instance 1 (should not be robust)
    assert var_evaluator.evaluate_single_instance(1, recourse_method="some_method", desired_output=1) is False

    print("All tests passed!")


# Run the test
test_var_robustness_simple()