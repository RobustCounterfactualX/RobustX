from datasets.custom_datasets.CsvDatasetLoader import CsvDatasetLoader
from models.pytorch_models.SimpleNNModel import SimpleNNModel
from tasks.ClassificationTask import ClassificationTask
from recourse_methods.NNCE import NNCE


def test_compute_nnce() -> None:
    model = SimpleNNModel(10, 7, 1)
    dl = CsvDatasetLoader('./assets/recruitment_data.csv', "HiringDecision")

    ct = ClassificationTask(model, dl)

    ct.train()

    recourse = NNCE(ct)

    res = recourse.generate_for_all(neg_value=0, column_name="HiringDecision")

    assert not res.empty
