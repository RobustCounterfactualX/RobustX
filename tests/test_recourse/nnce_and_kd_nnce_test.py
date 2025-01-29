from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.datasets.custom_datasets.CsvDatasetLoader import CsvDatasetLoader
from rocelib.models.Models import get_sklearn_model
from rocelib.models.pytorch_models.TrainablePyTorchModel import TrainablePyTorchModel
from rocelib.recourse_methods.KDTreeNNCE import KDTreeNNCE
from rocelib.recourse_methods.NNCE import NNCE
from rocelib.tasks.ClassificationTask import ClassificationTask


def test_kdtree_nnce() -> None:
    model = get_sklearn_model("log_reg")
    dl = get_example_dataset("ionosphere")
    dl.default_preprocess()
    model.train(dl.X, dl.y)


    ct = ClassificationTask(model, dl)


    kdrecourse = KDTreeNNCE(ct)

    res = kdrecourse.generate_for_all()

    assert not res.empty


def test_kdtree_nnce_same_as_nnce() -> None:
    model = TrainablePyTorchModel(10, [7], 1)
    dl = CsvDatasetLoader('./assets/recruitment_data.csv', "HiringDecision")

    model.train(dl.X, dl.y)

    ct = ClassificationTask(model, dl)


    kdrecourse = KDTreeNNCE(ct)

    nncerecourse = NNCE(ct)
    negs = dl.get_negative_instances(neg_value=0, column_name="HiringDecision")

    for _, neg in negs.iterrows():
        a = kdrecourse.generate_for_instance(neg)
        b = nncerecourse.generate_for_instance(neg)

        assert a.index == b.index
