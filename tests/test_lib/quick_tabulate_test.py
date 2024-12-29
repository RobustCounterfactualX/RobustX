import pandas as pd
from sklearn.preprocessing import StandardScaler

from RoCELib.datasets.ExampleDatasets import get_example_dataset
from RoCELib.datasets.custom_datasets.CsvDatasetLoader import CsvDatasetLoader
from RoCELib.lib.QuickTabulate import quick_tabulate
from RoCELib.models.pytorch_models.SimpleNNModel import SimpleNNModel
from RoCELib.recourse_methods.STCE import TrexNN
from RoCELib.recourse_methods.BinaryLinearSearch import BinaryLinearSearch
from RoCELib.recourse_methods.MCE import MCE
from RoCELib.recourse_methods.MCER import MCER
from RoCELib.recourse_methods.RNCE import RNCE
from RoCELib.recourse_methods.Wachter import Wachter


def test_quick_tabulate():
    #
    # methods = {"BLS": BinaryLinearSearch, "MCE": MCE, "MCER": MCER, "Wachter": Wachter, "RNCE": RNCE}
    # model = SimpleNNModel(10, [8], 1)
    # dl = CsvDatasetLoader(csv="../assets/standardized_recruitment_data.csv", target_column="HiringDecision")
    # subset = dl.get_negative_instances(column_name="HiringDecision", neg_value=0).head(20)
    #
    # quick_tabulate(dl, model, methods, subset=subset, preprocess=False, neg_value=0, column_name="HiringDecision", delta=0.01)

    methods = {"BLS": BinaryLinearSearch, "MCE": MCE, "MCER": MCER, "Wachter": Wachter, "RNCE": RNCE}
    model = SimpleNNModel(34, [10], 1)
    dl = get_example_dataset("ionosphere")

    quick_tabulate(dl, model, methods, neg_value=0, column_name="target", delta=0.05)

    assert True