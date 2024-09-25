from datasets.ExampleDatasets import get_example_dataset
from lib.QuickTabulate import quick_tabulate
from models.pytorch_models.SimpleNNModel import SimpleNNModel
from recourse_methods.STCE import TREX2
from recourse_methods.BinaryLinearSearch import BinaryLinearSearch
from recourse_methods.MCE import MCE
from recourse_methods.MCER import MCER
from recourse_methods.RNCE import RNCE
from recourse_methods.Wachter import Wachter


def test_quick_tabulate():

    # methods = {"BLS": BinaryLinearSearch, "MCE": MCE, "MCER": MCER, "Wachter": Wachter, "RNCE": RNCE}
    # model = SimpleNNModel(34, [10], 1)
    # dl = get_example_dataset("ionosphere")
    #
    # quick_tabulate(dl, model, methods, neg_value=0, column_name="target", delta=0.05)

    # methods = {"BLS": BinaryLinearSearch, "MCE": MCE, "MCER": MCER, "Wachter": Wachter}
    # methods = {"RNCE": RNCE}
    # model = SimpleNNModel(108, [10], 1)
    # dl = get_example_dataset("adult")
    # dl.default_preprocess()
    # subset = dl.get_negative_instances(column_name="income", neg_value=0).head(20)
    #
    # quick_tabulate(dl, model, methods, subset=subset, preprocess=False, neg_value=0, column_name="income", delta=0.05)

    assert True