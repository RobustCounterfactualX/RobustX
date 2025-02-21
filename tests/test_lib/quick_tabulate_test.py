import pandas as pd
from sklearn.preprocessing import StandardScaler

from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.datasets.custom_datasets.CsvDatasetLoader import CsvDatasetLoader
from rocelib.lib.QuickTabulate import quick_tabulate
from rocelib.models.pytorch_models.TrainablePyTorchModel import TrainablePyTorchModel
from rocelib.recourse_methods.STCE import TrexNN
from rocelib.recourse_methods.BinaryLinearSearch import BinaryLinearSearch
from rocelib.recourse_methods.MCE import MCE
from rocelib.recourse_methods.MCER import MCER
from rocelib.recourse_methods.RNCE import RNCE
from rocelib.recourse_methods.Wachter import Wachter

# NO LONGER RELEVANT WITH NEW GENERATION AND EVALUATION
# def test_quick_tabulate():
#     #
#     # methods = {"BLS": BinaryLinearSearch, "MCE": MCE, "MCER": MCER, "Wachter": Wachter, "RNCE": RNCE}
#     # model = TrainablePyTorchModel(10, [8], 1)
#     # dl = CsvDatasetLoader(csv="../assets/standardized_recruitment_data.csv", target_column="HiringDecision")
#     # subset = dl.get_negative_instances(column_name="HiringDecision", neg_value=0).head(20)
#     #
#     # quick_tabulate(dl, model, methods, subset=subset, preprocess=False, neg_value=0, column_name="HiringDecision", delta=0.01)

#     methods = {"BLS": BinaryLinearSearch, "MCE": MCE, "MCER": MCER, "Wachter": Wachter, "RNCE": RNCE}
#     model = TrainablePyTorchModel(34, [10], 1)
#     dl = get_example_dataset("ionosphere")
#     # dl.default_preprocess()

#     quick_tabulate(dl, model, methods, neg_value=0, column_name="target", delta=0.05)

#     assert True