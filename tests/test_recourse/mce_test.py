from enums.dataset_enums import Dataset
from enums.model_enums import ModelType
from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.models.pytorch_models.TrainablePyTorchModel import TrainablePyTorchModel
from rocelib.recourse_methods.MCE import MCE
from rocelib.tasks.ClassificationTask import ClassificationTask


def test_mce_predicts_positive_instances(testing_models):
    ct, dl, _ = testing_models.get(Dataset.IONOSPHERE, ModelType.NEURALNET, 34, 8, 1)

    recourse = MCE(ct)

    # _, neg = list(dl.get_negative_instances(neg_value=0).iterrows())[0]

    for _, neg in dl.get_negative_instances(neg_value=0).iterrows():

        res = recourse.generate_for_instance(neg)

        if not res.empty:
            prediction = ct.model.predict_single(res)

            assert prediction
