import pandas as pd
from sklearn.model_selection import train_test_split
from robustx.datasets.ExampleDatasets import get_example_dataset
from robustx.datasets.custom_datasets.CsvDatasetLoader import CsvDatasetLoader
from robustx.generators.CE_methods.KDTreeNNCE import KDTreeNNCE
from robustx.generators.CE_methods.Wachter import Wachter
from robustx.lib.models.pytorch_models.SimpleNNModel import SimpleNNModel
from robustx.generators.robust_CE_methods.APAS import APAS
from robustx.lib.tasks.ClassificationTask import ClassificationTask
import warnings
import torch

from robustx.robustness_evaluations.ApproximateDeltaRobustnessEvaluator import ApproximateDeltaRobustnessEvaluator
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


    # create a train-test split
    target_column = "target"
    X_train, X_test, y_train, y_test = train_test_split(dl.data.drop(columns=[target_column]), dl.data[target_column], test_size=0.2, random_state=0)

    accuracy = model.compute_accuracy(X_test.values, y_test.values)
      # Display results
    print("Model accuracy before training: ", accuracy)

    model.train(X_train, y_train, epochs=100)

    accuracy = model.compute_accuracy(X_test.values, y_test.values)
      # Display results
    print("Model accuracy after training: ", accuracy)

  
    # Create task
    task = ClassificationTask(model, dl)

 
    counterfactual_label = 0

    # retrieves a random instance from the training data that does not produce the specified counterfactual_label value, i.e., a valid instance
    pos_instance = X_test[y_test == counterfactual_label].iloc[0]
    if model.predict_single(pos_instance) == counterfactual_label:
        print("The selected instance is valid")
   
    
    # instanciate the robust_recourse_generator method. The APAS method is used to generate robust recourse
    confidence = 0.999
    robust_ce_generator = APAS(task, KDTreeNNCE, confidence)

    # generate robust recourse
    delta = 0.05
    robust_ce = robust_ce_generator._generation_method(pos_instance, target_column="target", desired_outcome=counterfactual_label, delta_max=delta)

    if robust_ce is None:
        print(f"\nNo counterfactual explanation robust to Δ={delta} model changes was found.")
    else:
        print(f"\nA counterfactual explanation robust to Δ={delta} model changes with probability ≥ {round((confidence)*100, 4)}% is:\n", robust_ce)
        print("\nwith prediction: ", task.model.predict_single(robust_ce))


if __name__ == "__main__":
    test_apas()