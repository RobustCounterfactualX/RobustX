from enums.dataset_enums import Dataset
from enums.model_enums import ModelType
from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.datasets.custom_datasets.CsvDatasetLoader import CsvDatasetLoader
from rocelib.models.Models import get_sklearn_model
from rocelib.models.pytorch_models.TrainablePyTorchModel import TrainablePyTorchModel
from rocelib.recourse_methods.KDTreeNNCE import KDTreeNNCE
from rocelib.recourse_methods.NNCE import NNCE
from rocelib.tasks.ClassificationTask import ClassificationTask


def test_kdtree_nnce(testing_models) -> None:
    ct, _ = testing_models.get(Dataset.IONOSPHERE, ModelType.LOGISTIC_REGRESSION)

    kdrecourse = KDTreeNNCE(ct)
    res = kdrecourse.generate_for_all()

    assert not res.empty


def test_kdtree_nnce_same_as_nnce(testing_models) -> None:
    ct, dl = testing_models.get(Dataset.RECRUITMENT, ModelType.NEURALNET, 10, 7, 1)

    kdrecourse = KDTreeNNCE(ct)
    nncerecourse = NNCE(ct)
    negs = dl.get_negative_instances(neg_value=0, column_name="HiringDecision")

    for _, neg in negs.iterrows():
        a = kdrecourse.generate_for_instance(neg)
        b = nncerecourse.generate_for_instance(neg)

        assert a.index == b.index
