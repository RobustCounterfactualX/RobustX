import pandas as pd
import time
import torch
import numpy as np
from tabulate import tabulate  # For better table formatting


from rocelib.tasks.Task import Task
from typing import List, Dict, Any, Union, Tuple
from rocelib.recourse_methods.BinaryLinearSearch import BinaryLinearSearch
from rocelib.recourse_methods.NNCE import NNCE
from rocelib.recourse_methods.KDTreeNNCE import KDTreeNNCE
from rocelib.recourse_methods.MCE import MCE
from rocelib.recourse_methods.Wachter import Wachter
from rocelib.recourse_methods.RNCE import RNCE
from rocelib.recourse_methods.MCER import MCER
from rocelib.recourse_methods.RoCourseNet import RoCourseNet
from rocelib.recourse_methods.STCE import TrexNN
from rocelib.recourse_methods.GuidedBinaryLinearSearch import GuidedBinaryLinearSearch
# from rocelib.recourse_methods.ModelMultiplicityMILP import ModelMultiplicityMILP
from rocelib.recourse_methods.APAS import APAS
# from rocelib.recourse_methods.ArgEnsembling import ArgEnsembling
from rocelib.recourse_methods.DiverseRobustCE import DiverseRobustCE

from rocelib.evaluations.ManifoldEvaluator import ManifoldEvaluator
from rocelib.evaluations.DistanceEvaluator import DistanceEvaluator
from rocelib.evaluations.ValidityEvaluator import ValidityEvaluator
from rocelib.evaluations.robustness_evaluations.MC_Robustness_Implementations.DeltaRobustnessEvaluator import DeltaRobustnessEvaluator
from rocelib.evaluations.RobustnessProportionEvaluator import RobustnessProportionEvaluator

# from robustx.generators.robust_CE_methods.ArgEnsembling import ArgEnsembling
# from robustx.generators.robust_CE_methods.DiverseRobustCE import DiverseRobustCE
# from robustx.generators.robust_CE_methods.MCER import MCER
# from robustx.generators.robust_CE_methods.ModelMultiplicityMILP import ModelMultiplicityMILP
# from robustx.generators.robust_CE_methods.PROPLACE import PROPLACE
# from robustx.generators.robust_CE_methods.RNCE import RNCE
# from robustx.generators.robust_CE_methods.ROAR import ROAR
# from robustx.generators.robust_CE_methods.STCE import STCE

# METHODS = {"APAS": APAS, "ArgEnsembling": ArgEnsembling, "DiverseRobustCE": DiverseRobustCE, "MCER": MCER,
#            "STCE": TrexNN, "BinaryLinearSearch": BinaryLinearSearch, "GuidedBinaryLinearSearch": GuidedBinaryLinearSearch,
#            "NNCE": NNCE, "KDTreeNNCE": KDTreeNNCE, "MCE": MCE, "Wachter": Wachter}
# EVALUATIONS = {"Distance": DistanceEvaluator, "Validity": ValidityEvaluator, "Manifold": ManifoldEvaluator,
#                "Delta-robustness": RobustnessProportionEvaluator}


METHODS = {
    "BinaryLinearSearch": BinaryLinearSearch,
    "GuidedBinaryLinearSearch": GuidedBinaryLinearSearch,
    # "MMMILP": ModelMultiplicityMILP,
    "NNCE": NNCE,
    "KDTreeNNCE": KDTreeNNCE,
    "MCE": MCE,
    "Wachter": Wachter,
    "RNCE": RNCE,
    "MCER": MCER,
    "RoCourseNet": RoCourseNet,
    "STCE": TrexNN
}

EVALUATIONS = {
    "Distance": DistanceEvaluator,
    "Validity": ValidityEvaluator,
    "DeltaRobustnessEvaluator": RobustnessProportionEvaluator,

    }

TIMEOUT_SECONDS = 60


class ClassificationTask(Task):
    """
    A specific task type for classification problems that extends the base Task class.

    This class provides methods for training the model and retrieving positive instances
    from the training data.

    Attributes:
        model: The model to be trained and used for predictions.
        _dataset: The dataset used for training the model.
    """



    def get_random_positive_instance(self, neg_value, column_name="target") -> pd.Series:
        """
        Retrieves a random positive instance from the training data that does not have the specified negative value.

        This method continues to sample from the training data until a positive instance
        is found whose predicted label is not equal to the negative value.

        @param neg_value: The value considered negative in the target variable.
        @param column_name: The name of the target column used to identify positive instances.
        @return: A Pandas Series representing a random positive instance.
        """
        # Get a random positive instance from the training data
        pos_instance = self._dataset.get_random_positive_instance()

        # Loop until a positive instance whose prediction is positive is found
        while self.model.predict_single(pos_instance) == neg_value:
            pos_instance = self._dataset.get_random_positive_instance()

        return pos_instance
    
    def generate(self, methods: List[str], type="DataFrame", **kwargs) -> Dict[str, Tuple[pd.DataFrame, float]]:#List[pd.DataFrame]:
        """
        Generates counterfactual explanations for the specified methods and stores the results.

        @param methods: List of recourse methods (by name) to use for counterfactual generation.
        """

        for method in methods:
            
            try:
                # Check if the method exists in the dictionary
                if method not in METHODS:
                    raise ValueError(f"Recourse method '{method}' not found. Available methods: {list(METHODS.keys())}")

                # Instantiate the recourse method
                recourse_method = METHODS[method](self)  # Pass the classification task to the method

                # Start timer
                start_time = time.perf_counter()

                res = recourse_method.generate_for_all(**kwargs)  # Generate counterfactuals
                res_correct_type = self.convert_datatype(res, type)
                # End timer
                end_time = time.perf_counter()

                # Store the result in the counterfactual explanations dictionary
                self._CEs[method] = [res, end_time - start_time]  

            except Exception as e:
                print(f"Error generating counterfactuals with method '{method}': {e}")
            
        return self.CEs
    
    def evaluate(self, methods: List[str], evaluations: List[str], **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Evaluates the generated counterfactual explanations using specified evaluation metrics.

        @param methods: List of recourse methods to evaluate.
        @param evaluations: List of evaluation metrics to apply.
        @return: Dictionary containing evaluation results per method and metric.
        """
        evaluation_results = {}

        # Validate evaluation names
        invalid_evaluations = [ev for ev in evaluations if ev not in EVALUATIONS]
        if invalid_evaluations:
            raise ValueError(f"Invalid evaluation metrics: {invalid_evaluations}. Available: {list(EVALUATIONS.keys())}")

        # Filter out methods that haven't been generated
        valid_methods = [method for method in methods if method in self._CEs]
        if not valid_methods:
            print("No valid methods have been generated for evaluation.")
            return evaluation_results

        # Perform evaluation
        for evaluation in evaluations:
            evaluator_class = EVALUATIONS[evaluation]

            try:
                    # Create evaluator instance
                evaluator = evaluator_class(self)

                for method in valid_methods:
                    # Retrieve generated counterfactuals
                    counterfactuals = self._CEs[method][0]  # Extract DataFrame from stored list
                    print(f"Shape of CEs for {method}: {counterfactuals.shape}")

                    # Ensure counterfactuals are not empty
                    if counterfactuals is None or counterfactuals.empty:
                        print(f"Skipping evaluation for method '{method}' as no counterfactuals were generated.")
                        continue

                    # Perform evaluation
                    score = evaluator.evaluate(method, **kwargs)

                    # Store results
                    if method not in evaluation_results:
                        evaluation_results[method] = {}
                    evaluation_results[method][evaluation] = score

            except Exception as e:
                print(f"Error evaluating '{evaluation}': {e}")

        # Print results in table format
        self._print_evaluation_results(evaluation_results, evaluations)

        return evaluation_results

    def _print_evaluation_results(self, evaluation_results: Dict[str, Dict[str, Any]], evaluations: List[str]):
        """
        Prints the evaluation results in a table format.

        @param evaluation_results: Dictionary containing evaluation scores per method and metric.
        @param evaluations: List of evaluation metrics that were actually requested.
        """
        if not evaluation_results:
            print("No evaluation results to display.")
            return

        # Prepare table data
        table_data = []
        headers = ["Recourse Method"] + evaluations  # Only include requested evaluations

        for method, scores in evaluation_results.items():
            row = [method] + [scores.get(metric, "N/A") for metric in evaluations]
            table_data.append(row)

        print("\nEvaluation Results:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    

    def convert_datatype(self, data: pd.DataFrame, target_type: str):
        """
        Converts a Pandas DataFrame to the specified data type.

        @param data: pd.DataFrame - The input DataFrame.
        @param target_type: str - The target data type: "DataFrame", "NPArray", or "TTensor".
        @return: Converted data in the specified format.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a Pandas DataFrame.")

        target_type = target_type.lower()  # Normalize input for case insensitivity

        if target_type == "dataframe":
            return data
        elif target_type == "nparray":
            return data.to_numpy()
        elif target_type == "tensor":
            return torch.tensor(data.to_numpy(), dtype=torch.float32)
        else:
            raise ValueError("Invalid target_type. Choose from: 'DataFrame', 'NPArray', 'Tensor'.")