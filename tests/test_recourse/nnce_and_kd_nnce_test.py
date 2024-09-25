from datasets.ExampleDatasets import get_example_dataset
from datasets.custom_datasets.CsvDatasetLoader import CsvDatasetLoader
from models.Models import get_sklearn_model
from models.pytorch_models.SimpleNNModel import SimpleNNModel
from recourse_methods.KDTreeNNCE import KDTreeNNCE
from recourse_methods.NNCE import NNCE
from tasks.ClassificationTask import ClassificationTask


def test_kdtree_nnce() -> None:
    model = get_sklearn_model("log_reg")
    dl = get_example_dataset("ionosphere")
    dl.default_preprocess()

    ct = ClassificationTask(model, dl)

    ct.train()

    kdrecourse = KDTreeNNCE(ct)

    res = kdrecourse.generate_for_all()

    assert not res.empty


def test_kdtree_nnce_same_as_nnce() -> None:
    model = SimpleNNModel(10, [7], 1)
    dl = CsvDatasetLoader('./assets/recruitment_data.csv', "HiringDecision")

    ct = ClassificationTask(model, dl)

    ct.train()

    kdrecourse = KDTreeNNCE(ct)

    nncerecourse = NNCE(ct)
    negs = dl.get_negative_instances(neg_value=0, column_name="HiringDecision")

    for _, neg in negs.iterrows():
        a = kdrecourse.generate_for_instance(neg)
        b = nncerecourse.generate_for_instance(neg)

        assert a.index == b.index
