import torch

from datasets.ExampleDatasets import get_example_dataset
from models.pytorch_models.SimpleNNModel import SimpleNNModel
from recourse_methods.MCE import MCE
from tasks.ClassificationTask import ClassificationTask


def test_mce_predicts_positive_instances():
    model = SimpleNNModel(34, [8], 1)
    dl = get_example_dataset("ionosphere")

    ct = ClassificationTask(model, dl)

    dl.default_preprocess()
    ct.train()

    recourse = MCE(ct)

    # _, neg = list(dl.get_negative_instances(neg_value=0).iterrows())[0]

    for _, neg in dl.get_negative_instances(neg_value=0).iterrows():

        res = recourse.generate_for_instance(neg)

        if not res.empty:

            prediction = model.predict_single(res)

            assert prediction

# def test_mce_generates_all_robust():
#
#     # Create the model instance
#     model = SimpleNNModel(input_dim=2, hidden_dim=[2], output_dim=1)
#
#     # Define the weights and biases according to the image provided
#     weights = {
#         'fc0_weight': torch.tensor([[1.0, 0.0], [0.0, 1.0]]),  # Weights for first Linear layer (input -> hidden)
#         'fc0_bias': torch.tensor([0.0, 0.0]),  # Bias for first Linear layer
#         'fc1_weight': torch.tensor([[1.0, -1.0]]),  # Weights for second Linear layer (hidden -> output)
#         'fc1_bias': torch.tensor([0.0])  # Bias for second Linear layer
#     }
#
#     # Set the custom weights
#     model.set_weights(weights)
#
#     # dl = CsvDatasetLoader('../assets/recruitment_data.csv', "HiringDecision")
#     dl = CsvDatasetLoader('../assets/random_normal_values.csv', "target")
#     ct = ClassificationTask(model, dl)
#
#     mce = MCE(ct)
#
#     opt = DeltaRobustnessEvaluator(ct)
#
#     for _, neg in dl.get_negative_instances(neg_value=0).iterrows():
#         ce = mce.generate_for_instance(neg)
#         robust = opt.evaluate(ce, delta=0.05)
#         print("######################################################")
#         print("CE was: ", ce)
#         print("This CE was" + ("" if robust else " not") + " robust")
#         print("######################################################")
#         assert robust