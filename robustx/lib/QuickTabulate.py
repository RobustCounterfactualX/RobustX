import pandas as pd

from robustx.datasets.DatasetLoader import DatasetLoader
from robustx.datasets.provided_datasets.ExampleDatasetLoader import ExampleDatasetLoader
from robustx.evaluations.DistanceEvaluator import DistanceEvaluator
from robustx.evaluations.ManifoldEvaluator import ManifoldEvaluator
from robustx.evaluations.RobustnessProportionEvaluator import RobustnessProportionEvaluator
from robustx.evaluations.ValidityEvaluator import ValidityEvaluator
from robustx.lib.models.BaseModel import BaseModel
from robustx.generators.CEGenerator import CEGenerator
from robustx.lib.tasks.ClassificationTask import ClassificationTask
from typing import Dict
import time
from tabulate import tabulate


def quick_tabulate(dl: DatasetLoader, model: BaseModel, methods: Dict[str, CEGenerator.__class__],
                   subset: pd.DataFrame = None, preprocess=True, **params):
    """
    Generates and prints a table summarizing the performance of different counterfactual explanation generation methods.

    @param dl: DatasetLoader, The dataset loader to preprocess and provide data for the classification task.
    @param model: BaseModel, The model to be trained and evaluated.
    @param methods: Dict[str, RecourseGenerator.__class__], A dictionary where keys are method names and values are
                    classes of CE generation methods to evaluate.
    @param subset: optional DataFrame, subset of instances you would like to generate CEs on
    @param preprocess: optional Boolean, whether you want to preprocess the dataset or not, example datasets only
    @param **params: Additional parameters to be passed to the CE generation methods and evaluators.
    @return: None
    """

    # Preprocess example datasets
    if preprocess and isinstance(dl, ExampleDatasetLoader):
        dl.default_preprocess()

    # Create and train task
    model.train(dl.X, dl.y)
    ct = ClassificationTask(model, dl)

    results = []

    # Instantiate evaluators
    validity_evaluator = ValidityEvaluator(ct)
    distance_evaluator = DistanceEvaluator(ct)
    robustness_evaluator = RobustnessProportionEvaluator(ct)

    for method_name in methods:

        # Instantiate ce_generator method
        ce_generator = methods[method_name](ct)

        # Start timer
        start_time = time.perf_counter()

        # Generate CEs
        if subset is None:
            ces = ce_generator.generate_for_all(**params)
        else:
            ces = ce_generator.generate(subset, **params)

        # End timer
        end_time = time.perf_counter()

        # Add to results
        results.append([method_name, end_time - start_time, validity_evaluator.evaluate(ces, **params),
                        distance_evaluator.evaluate(ces, subset=subset, **params), robustness_evaluator.evaluate(ces, **params),
                        ])

    # Set headers
    headers = ["Method", "Execution Time (s)", "Validity proportion", "Average Distance", "Robustness proportion"]

    # Print results
    print(tabulate(results, headers, tablefmt="grid"))

