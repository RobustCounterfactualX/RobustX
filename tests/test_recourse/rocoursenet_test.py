from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.evaluations.RobustnessProportionEvaluator import RobustnessProportionEvaluator
from rocelib.models.pytorch_models.SimpleNNModel import SimpleNNModel
from rocelib.recourse_methods.RoCourseNet import RoCourseNet
from rocelib.tasks.ClassificationTask import ClassificationTask
from rocelib.evaluations.ValidityEvaluator import ValidityEvaluator



def test_rocoursenet() -> None:
    # Step 1: Initialize the model and dataset
    model = SimpleNNModel(34, [8], 1)  # Neural network with input size 34, hidden layer [8], and output size 1
    dl = get_example_dataset("ionosphere")  # Load the "ionosphere" dataset

    # Step 2: Set up the classification task
    ct = ClassificationTask(model, dl)

    # Step 3: Preprocess the data and train the model
    dl.default_preprocess()  # Preprocess the dataset (e.g., scaling, normalization)
    ct.train()  # Train the model on the dataset

    recourse = RoCourseNet(ct)

    res = recourse.generate_for_all()

    val = ValidityEvaluator(ct)

    x = val.evaluate(res)

    assert x > 0.05
