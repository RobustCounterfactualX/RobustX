import pandas as pd

from datasets.custom_datasets.CsvDatasetLoader import CsvDatasetLoader
from models.pytorch_models.SimpleNNModel import SimpleNNModel
from recourse_methods.RecourseMethods import compute_nnce
from tasks.ClassificationTask import ClassificationTask


def test_compute_nnce() -> None:
    model = SimpleNNModel(10, 7, 1)
    dl = CsvDatasetLoader('./assets/recruitment_data.csv', "HiringDecision")

    ct = ClassificationTask(model, dl)

    ct.train()

    for _, negative in ct.training_data.get_negative_instances(0, column_name="HiringDecision").iterrows():

        nnce = compute_nnce(ct.model, ct.training_data, negative)

        print(nnce)

    assert True
