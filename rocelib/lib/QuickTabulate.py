import pandas as pd

from rocelib.datasets.DatasetLoader import DatasetLoader
from rocelib.datasets.provided_datasets.ExampleDatasetLoader import ExampleDatasetLoader
from rocelib.evaluations.DistanceEvaluator import DistanceEvaluator
from rocelib.evaluations.ManifoldEvaluator import ManifoldEvaluator
from rocelib.evaluations.RobustnessProportionEvaluator import RobustnessProportionEvaluator
from rocelib.evaluations.ValidityEvaluator import ValidityEvaluator
from rocelib.models.TrainableModel import TrainableModel
from rocelib.models.TrainedModel import TrainedModel

from rocelib.recourse_methods.RecourseGenerator import RecourseGenerator
from rocelib.tasks.ClassificationTask import ClassificationTask
from typing import Dict
import time
from tabulate import tabulate


def quick_tabulate(dl: DatasetLoader, model: TrainableModel, methods: Dict[str, RecourseGenerator.__class__],
                   subset: pd.DataFrame = None, preprocess=True, **params):
    """
    Generates and prints a table summarizing the performance of different recourse generation methods.

    @param dl: DatasetLoader, The dataset loader to preprocess and provide data for the classification task.
    @param model: TrainableModel, The model to be trained and evaluated.
    @param methods: Dict[str, RecourseGenerator.__class__], A dictionary where keys are method names and values are
                    classes of recourse generation methods to evaluate.
    @param subset: optional DataFrame, subset of instances you would like to generate CEs on
    @param preprocess: optional Boolean, whether you want to preprocess the dataset or not, example datasets only
    @param **params: Additional parameters to be passed to the recourse generation methods and evaluators.
    @return: None
    """

    trained_model = model.train(dl.X, dl.y)

    # Create and train task
    ct = ClassificationTask(trained_model, dl)

    results = []

    # Instantiate evaluators
    validity_evaluator = ValidityEvaluator(ct)
    distance_evaluator = DistanceEvaluator(ct)
    robustness_evaluator = RobustnessProportionEvaluator(ct)

    for method_name in methods:

        # Instantiate recourse method
        recourse = methods[method_name](ct)

        # Start timer
        start_time = time.perf_counter()

        # Generate CE
        if subset is None:
            ces = recourse.generate_for_all(**params)
        else:
            ces = recourse.generate(subset, **params)

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

