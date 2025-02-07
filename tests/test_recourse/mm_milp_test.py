from enums.dataset_enums import Dataset
from enums.model_enums import ModelType
from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.models.pytorch_models.TrainablePyTorchModel import TrainablePyTorchModel
from rocelib.recourse_methods.ModelMultiplicityMILP import ModelMultiplicityMILP
from rocelib.tasks.ClassificationTask import ClassificationTask


# def test_mce_predicts_positive_instances(testing_models):

#     ct1, dl, trained_model1 = testing_models.get(Dataset.IONOSPHERE, ModelType.NEURALNET, 34, 8, 1)
#     ct2, _, trained_model2 = testing_models.get(Dataset.IONOSPHERE, ModelType.NEURALNET, 34, 16, 8, 1)
#     ct3, _, trained_model3 = testing_models.get(Dataset.IONOSPHERE, ModelType.NEURALNET, 34, 16, 8, 4, 1)

#     recourse = ModelMultiplicityMILP(dl, [trained_model1, trained_model2, trained_model3])

#     for _, neg in dl.get_negative_instances(neg_value=0).iterrows():
#         res = recourse.generate_for_instance(neg)

#         # TODO can we just assert not res.empty?
#         if not res.empty:
#             prediction1 = trained_model1.predict_single(res)

#             prediction2 = trained_model2.predict_single(res)

#             prediction3 = trained_model3.predict_single(res)

#             assert prediction1
#             assert prediction2
#             assert prediction3
