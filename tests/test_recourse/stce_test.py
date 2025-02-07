import pandas as pd

from enums.dataset_enums import Dataset
from enums.model_enums import ModelType
from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.evaluations.RobustnessProportionEvaluator import RobustnessProportionEvaluator
from rocelib.models.pytorch_models.TrainablePyTorchModel import TrainablePyTorchModel
from rocelib.recourse_methods.MCE import MCE
from rocelib.recourse_methods.STCE import TrexNN
from rocelib.tasks.ClassificationTask import ClassificationTask


# def test_stce(testing_models) -> None:
#     ct, dl, _ = testing_models.get(Dataset.IONOSPHERE, ModelType.NEURALNET, 34, 8, 1)

#     recourse = TrexNN(ct)
#     re = RobustnessProportionEvaluator(ct)

#     _, neg = list(dl.get_negative_instances(neg_value=0).iterrows())[0]
#     ces = []

#     for _, neg in dl.get_negative_instances(neg_value=0).iterrows():
#         res = recourse.generate_for_instance(neg, delta=0.005)
#         ces.append(res)
#         assert ct.model.predict_single(res)

#     ce_df = pd.concat(ces)
#     print(re.evaluate(ce_df, delta=0.005))
