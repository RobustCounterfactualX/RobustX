from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.evaluations.RobustnessProportionEvaluator import RobustnessProportionEvaluator
from rocelib.models.pytorch_models.SimpleNNModel import SimpleNNModel
from rocelib.recourse_methods.RoCourseNet import RoCourseNet
from rocelib.tasks.ClassificationTask import ClassificationTask


def test_rocoursenet() -> None:
    # Step 1: Initialize the model and dataset
    model = SimpleNNModel(34, [8], 1)  # Neural network with input size 34, hidden layer [8], and output size 1
    dl = get_example_dataset("ionosphere")  # Load the "ionosphere" dataset

    # Step 2: Set up the classification task
    ct = ClassificationTask(model, dl)

    # Step 3: Preprocess the data and train the model
    dl.default_preprocess()  # Preprocess the dataset (e.g., scaling, normalization)
    ct.train()  # Train the model on the dataset

    # Step 4: Define robustness parameters
    delta = 0.01  # Tolerance for robustness in the feature space

    # Step 5: Initialize the RoCourseNet recourse generator
    recourse = RoCourseNet(ct)

    # Step 6: Select negative instances to test counterfactual generation
    for _, neg in dl.get_negative_instances(neg_value=0).head(10).iterrows():
        # Generate a robust counterfactual for the current negative instance
        res = recourse.generate_for_instance(neg, delta=delta)
        
        # Evaluate the robustness of the generated counterfactual
        assert recourse.intabs.evaluate(res, delta=delta), f"Counterfactual failed robustness evaluation for delta={delta}"

