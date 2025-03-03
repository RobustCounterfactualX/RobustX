import torch

from rocelib.datasets.custom_datasets.CsvDatasetLoader import CsvDatasetLoader
from rocelib.evaluations.robustness_evaluations.MC_Robustness_Implementations.DeltaRobustnessEvaluator import \
    DeltaRobustnessEvaluator
from rocelib.models.pytorch_models.TrainablePyTorchModel import TrainablePyTorchModel
from rocelib.recourse_methods.MCER import MCER
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
    trained_model = model.set_weights(weights)

    dl = CsvDatasetLoader('./assets/random_normal_values.csv', "target", 0)
    ct = ClassificationTask(trained_model, dl)

    mcer = MCER(ct)

    opt = DeltaRobustnessEvaluator(ct)

    for _, neg in dl.get_negative_instances().iterrows():
        ce = mcer.generate_for_instance(neg, delta=0.1)
        robust = opt.evaluate_single_instance(ce, delta=0.1)
        print("######################################################")
        print("CE was: ", ce)
        print("This CE was" + ("" if robust else " not") + " robust")
        print("######################################################")
        assert robust


# def test_mcer_generates_all_robust_custom(testing_models):
#     # Create the model instance
#     ct, dl = testing_models.get(Dataset.IONOSPHERE, ModelType.NEURALNET, 34, 10, 1)

#     mcer = MCER(ct)

#     opt = DeltaRobustnessEvaluator(ct)
#     ces = []
#     negs = dl.get_negative_instances(neg_value=0)
#     for _, neg in negs.iterrows():
#         ce = mcer.generate_for_instance(neg, delta=0.005)
#         ces.append(ce)
#         if not ce.equals(pd.DataFrame(neg)):
#             robust = opt.evaluate(ce, delta=0.005)
#             print("######################################################")
#             print("CE was: ", ce)
#             print("This CE was" + ("" if robust else " not") + " robust")
#             print("######################################################")
#             assert robust
#     print(ces)
