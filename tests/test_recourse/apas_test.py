import pandas as pd
from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.datasets.custom_datasets.CsvDatasetLoader import CsvDatasetLoader
from rocelib.generators.recourse_methods.KDTreeNNCE import KDTreeNNCE
from rocelib.generators.recourse_methods.Wachter import Wachter
from rocelib.lib.models.pytorch_models.SimpleNNModel import SimpleNNModel
from rocelib.generators.robust_recourse_methods.APAS import APAS
from rocelib.lib.tasks.ClassificationTask import ClassificationTask
import warnings
import torch

from rocelib.robustness_evaluations.ApproximateDeltaRobustnessEvaluator import ApproximateDeltaRobustnessEvaluator
warnings.filterwarnings("ignore")


def test_apas() -> None:
   
    # Load dataset
    # csv_path = "../../examples/test.csv"  # Path to the CSV file
    # target_column = "target"  # Name of the target column

    # # Create an instance of CsvDatasetLoader
    # dl = CsvDatasetLoader(csv=csv_path, target_column=target_column)
    # print(dl.data)

    # Load dataset
    dl = get_example_dataset("ionosphere", seed=42)
    dl.default_preprocess()
    print(dl.data.head())

    # Load model, note some RecourseGenerators may only work with a certain type of model,
    model = SimpleNNModel(34, [8], 1, seed=0)

    # set manual weights and biases
    # weights = {
    #     'fc0_weight': torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, -1.0], [0.0, 3.0]]),  # Weights for first Linear layer (input -> hidden)
    #     'fc0_bias': torch.tensor([0.0, 0.0, 0.0, 0.0]),  # Bias for first Linear layer
    #     'fc1_weight': torch.tensor([[1.0, -1.0, 0.0, 3.0]]),  # Weights for second Linear layer (hidden -> output)
    #     'fc1_bias': torch.tensor([0.0])  # Bias for second Linear layer
    # }

    # # Set the custom weights
    # model.set_weights(weights)

    # Create task
    task = ClassificationTask(model, dl)

    # Train model on dataset
    task.train()
    counterfactual_label = 1

    # retrieves a random instance from the training data that does not produce the specified counterfactual_label value, i.e., a valid instance
    pos_instance = task.get_random_positive_instance(neg_value=counterfactual_label, column_name="target")
    
    # instanciate the robust_recourse_generator method. The APAS method is used to generate robust recourse
    robust_recourse = APAS(task, KDTreeNNCE)

    # generate robust recourse
    delta = 0.05
    robust_evaluator = ApproximateDeltaRobustnessEvaluator(ct=task, alpha=0.999, R=0.995)
    robust_ce = robust_recourse._generation_method(pos_instance, target_column="target", desired_outcome=counterfactual_label, robustness_check=robust_evaluator, delta_max=delta)

    if robust_ce is None:
        print(f"\nNo counterfactual explanation robust to Δ={delta} model changes was found.")
    else:
        print(f"\nA counterfactual explanation robust to Δ={delta} model changes with probability ≥ {round((robust_evaluator.alpha)*100, 4)}% is:\n", robust_ce)
        print("\nwith prediction: ", task.model.predict_single(robust_ce))


if __name__ == "__main__":
    test_apas()