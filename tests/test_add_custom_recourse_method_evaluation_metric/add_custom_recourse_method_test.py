import pandas as pd

from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.recourse_methods.RecourseGenerator import RecourseGenerator
from rocelib.tasks.TaskBuilder import TaskBuilder


class NewRecourseMethod(RecourseGenerator):
    def _generation_method(self, instance, column_name="target", neg_value=0, **kwargs):
        return pd.DataFrame([[1, 2, 3, 4]], columns=["new_recourse_method_1", "new_recourse_method_2", "new_recourse_method_3", "new_recourse_method_4"])



def test_method_gets_added_to_list_of_recourse_methods():
    dl = get_example_dataset("ionosphere")
    # Not using testing models as we will be adding a recourse method to the task so could mess up other tests
    ct = TaskBuilder().add_pytorch_model(34, [8], 1, dl).add_data(dl).build()

    ct.add_recourse_method("new_recourse_method", NewRecourseMethod)
    assert "new_recourse_method" in ct.get_recourse_methods()

def test_ce_generation_for_new_method():
    dl = get_example_dataset("ionosphere")
    # Not using testing models as we will be adding a recourse method to the task so could mess up other tests
    ct = TaskBuilder().add_pytorch_model(34, [8], 1, dl).add_data(dl).build()

    ct.add_recourse_method("new_recourse_method", NewRecourseMethod)
    ces = ct.generate(["new_recourse_method"])
    reference_df = pd.DataFrame([[1, 2, 3, 4]], columns=["new_recourse_method_1", "new_recourse_method_2", "new_recourse_method_3", "new_recourse_method_4"])
    assert ces["new_recourse_method"][0].eq(reference_df.iloc[0]).all(axis=1).all()
