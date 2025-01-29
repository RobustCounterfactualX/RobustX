import pandas as pd
import torch

from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.datasets.custom_datasets.CsvDatasetLoader import CsvDatasetLoader
from rocelib.models.pytorch_models.TrainablePyTorchModel import TrainablePyTorchModel
from rocelib.recourse_methods.MCER import MCER
from rocelib.robustness_evaluations.DeltaRobustnessEvaluator import DeltaRobustnessEvaluator
from rocelib.tasks.ClassificationTask import ClassificationTask


def test_mcer_generates_all_robust():
    # Create the model instance
    model = TrainablePyTorchModel(input_dim=2, hidden_dim=[2], output_dim=1)

    # Define the weights and biases according to the image provided
    weights = {
        'fc0_weight': torch.tensor([[1.0, 0.0], [0.0, 1.0]]),  # Weights for first Linear layer (input -> hidden)
        'fc0_bias': torch.tensor([0.0, 0.0]),  # Bias for first Linear layer
        'fc1_weight': torch.tensor([[1.0, -1.0]]),  # Weights for second Linear layer (hidden -> output)
        'fc1_bias': torch.tensor([0.0])  # Bias for second Linear layer
    }

    # Set the custom weights
    model.set_weights(weights)

    dl = CsvDatasetLoader('./assets/random_normal_values.csv', "target")
    ct = ClassificationTask(model, dl)

    mcer = MCER(ct)

    opt = DeltaRobustnessEvaluator(ct)

    for _, neg in dl.get_negative_instances(neg_value=0).iterrows():
        ce = mcer.generate_for_instance(neg, delta=0.1)
        robust = opt.evaluate(ce, delta=0.1)
        print("######################################################")
        print("CE was: ", ce)
        print("This CE was" + ("" if robust else " not") + " robust")
        print("######################################################")
        assert robust


def test_mcer_generates_all_robust_custom():
    # Create the model instance
    model = TrainablePyTorchModel(input_dim=34, hidden_dim=[10], output_dim=1)

    dl = get_example_dataset("ionosphere")

    dl.default_preprocess()

    model.train(dl.X, dl.y)

    ct = ClassificationTask(model, dl)



    mcer = MCER(ct)

    opt = DeltaRobustnessEvaluator(ct)
    ces = []
    negs = dl.get_negative_instances(neg_value=0)
    for _, neg in negs.iterrows():
        ce = mcer.generate_for_instance(neg, delta=0.005)
        ces.append(ce)
        if not ce.equals(pd.DataFrame(neg)):
            robust = opt.evaluate(ce, delta=0.005)
            print("######################################################")
            print("CE was: ", ce)
            print("This CE was" + ("" if robust else " not") + " robust")
            print("######################################################")
            assert robust
    print(ces)
