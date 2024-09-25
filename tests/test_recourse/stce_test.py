import pandas as pd

from datasets.ExampleDatasets import get_example_dataset
from evaluations.RobustnessProportionEvaluator import RobustnessProportionEvaluator
from models.pytorch_models.SimpleNNModel import SimpleNNModel
from recourse_methods.MCE import MCE
from recourse_methods.STCE import TREX2
from tasks.ClassificationTask import ClassificationTask


def test_stce() -> None:
    model = SimpleNNModel(34, [8], 1)
    dl = get_example_dataset("ionosphere")

    ct = ClassificationTask(model, dl)

    dl.default_preprocess()
    ct.train()

    recourse = TREX2(ct)

    re = RobustnessProportionEvaluator(ct)

    _, neg = list(dl.get_negative_instances(neg_value=0).iterrows())[0]
    ces = []
    for _, neg in dl.get_negative_instances(neg_value=0).head(10).iterrows():
        res = recourse.generate_for_instance(neg, delta=0.05)
        ces.append(res)
        assert model.predict_single(res)

    ce_df = pd.concat(ces)
    print(re.evaluate(ce_df, delta=0.05))
