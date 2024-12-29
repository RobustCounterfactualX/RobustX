import pandas as pd
import torch

from RoCELib.datasets.ExampleDatasets import get_example_dataset
from RoCELib.datasets.custom_datasets.CsvDatasetLoader import CsvDatasetLoader
from RoCELib.models.pytorch_models.SimpleNNModel import SimpleNNModel
from RoCELib.recourse_methods.KDTreeNNCE import KDTreeNNCE
from RoCELib.robustness_evaluations.DeltaRobustnessEvaluator import DeltaRobustnessEvaluator
from RoCELib.tasks.ClassificationTask import ClassificationTask


def test_from_example_8_in_paper():
    # Create the model instance
    model = SimpleNNModel(input_dim=2, hidden_dim=[], output_dim=1)

    # Define the weights and biases according to the logistic regression model M_theta (x) = sigmoid(−x1 + x2)
    weights = {
        'fc0_weight': torch.tensor([[-1.0, 1.0]]),  # Weights for first Linear layer (input -> hidden)
        'fc0_bias': torch.tensor([0.0]),  # Bias for first Linear layer
    }

    # Set the custom weights
    model.set_weights(weights)

    # Load dummy dataset and create task
    dl = CsvDatasetLoader('./assets/random_normal_values.csv', "target")
    ct = ClassificationTask(model, dl)

    # Create robustness checker
    opt = DeltaRobustnessEvaluator(ct)

    # Create instance for which to check robustness, this one is not robust
    ce = pd.DataFrame({'x1': [0.7], 'x2': [0.7]})

    robust_check_1 = opt.evaluate(ce, delta=0.1)
    print("######################################################")
    print("CE was: ", ce)
    print("This CE was" + ("" if robust_check_1 else " not") + " robust")
    print("######################################################")

    assert not robust_check_1

    # This one is the original negative instance and so is not robust
    ce = pd.DataFrame({'x1': [0.7], 'x2': [0.5]})

    robust_check_2 = opt.evaluate(ce, delta=0.1)
    print("######################################################")
    print("CE was: ", ce)
    print("This CE was" + ("" if robust_check_2 else " not") + " robust")
    print("######################################################")

    assert not robust_check_2

    # This one is robust
    ce = pd.DataFrame({'x1': [0.7], 'x2': [0.86]})

    robust_check_3 = opt.evaluate(ce, delta=0.1)
    print("######################################################")
    print("CE was: ", ce)
    print("This CE was" + ("" if robust_check_3 else " not") + " robust")
    print("######################################################")

    assert robust_check_3


def test_mix_of_robustness_from_example_7_in_paper():
    # Create the model instance
    model = SimpleNNModel(input_dim=2, hidden_dim=[2], output_dim=1)

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

    kdtree = KDTreeNNCE(ct)

    opt = DeltaRobustnessEvaluator(ct)

    for _, neg in dl.get_negative_instances(neg_value=0).iterrows():
        ce = kdtree.generate_for_instance(neg)
        robust = opt.evaluate(ce, delta=0.05)
        print("######################################################")
        print("CE was: ", ce)
        print("This CE was" + ("" if robust else " not") + " robust")
        print("######################################################")


def test_ionosphere_kdtree_robustness():
    # Instantiate the neural network and the IntervalAbstractionPytorch class
    model = SimpleNNModel(34, [8], 1)
    # dl = CsvDatasetLoader('../assets/recruitment_data.csv', "HiringDecision")
    dl = get_example_dataset("ionosphere")
    ct = ClassificationTask(model, dl)

    dl.default_preprocess()

    ct.train()

    kdtree = KDTreeNNCE(ct)

    opt = DeltaRobustnessEvaluator(ct)

    for _, neg in dl.get_negative_instances(neg_value=0).iterrows():
        ce = kdtree.generate_for_instance(neg)
        robust = opt.evaluate(ce, delta=0.0000000000000000001)
        print("######################################################")
        print("CE was: ", ce)
        print("This CE was" + ("" if robust else " not") + " robust")
        print("######################################################")
