import pandas as pd

from robustx.evaluations import *
from robustx.robustness_evaluations import ModelChangesRobustnessSetEvaluator
from robustx.lib.tasks.ClassificationTask import ClassificationTask
import time
from tabulate import tabulate
from robustx.generators.CE_methods import *
from robustx.generators.robust_CE_methods import *


METHODS = {"APAS": APAS, "ArgEnsembling": ArgEnsembling, "DiverseRobustCE": DiverseRobustCE, "MCER": MCER,
           "ModelMultiplicityMILP": ModelMultiplicityMILP, "PROPLACE": PROPLACE, "RNCE": RNCE, "ROAR": ROAR,
           "STCE": STCE, "BinaryLinearSearch": BinaryLinearSearch, "GuidedBinaryLinearSearch": GuidedBinaryLinearSearch,
           "NNCE": NNCE, "KDTreeNNCE": KDTreeNNCE, "MCE": MCE, "Wachter": Wachter}
EVALUATIONS = {"Distance": DistanceEvaluator, "Validity": ValidityEvaluator, "Manifold": ManifoldEvaluator,
               "Delta-robustness": RobustnessProportionEvaluator}


def default_benchmark(ct: ClassificationTask, methods, evaluations,
                      subset: pd.DataFrame = None, **params):
    """
    Generates and prints a table summarizing the performance of different counterfactual explanation generation methods.

    @param ct: ClassificationTask.
    @param methods: A list or a set of method names.
    @param evaluations: A list or a set of evaluator names.
    @param subset: optional DataFrame, subset of instances you would like to generate CEs on
    @param **params: Additional parameters to be passed to the CE generation methods and evaluators.
    @return: None
    """

    results = []

    for method_name in methods:

        # Instantiate ce_generator method
        ce_generator = METHODS[method_name](ct)

        # Start timer
        start_time = time.perf_counter()

        # Generate CEs
        if subset is None:
            ces = ce_generator.generate_for_all(**params)
        else:
            ces = ce_generator.generate(subset, **params)

        # End timer
        end_time = time.perf_counter()

        # start evaluation
        eval_results = [method_name, end_time-start_time]
        for eval_name in evaluations:
            ce_evaluator = EVALUATIONS[eval_name](ct)
            eval_results.append(ce_evaluator.evaluate(ces, **params))

        # Add to results
        results.append(eval_results)

    # Set headers
    headers = ["Method", "Execution Time (s)"]
    for eval_name in evaluations:
        headers.append(eval_name)

    # Print results
    print(tabulate(results, headers, tablefmt="grid"))
