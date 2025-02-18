import pandas as pd
import time
import torch
import numpy as np

from rocelib.tasks.Task import Task
from typing import List, Dict, Any, Union
from rocelib.recourse_methods.BinaryLinearSearch import BinaryLinearSearch
from rocelib.recourse_methods.NNCE import NNCE
from rocelib.recourse_methods.KDTreeNNCE import KDTreeNNCE
from rocelib.recourse_methods.MCE import MCE
from rocelib.recourse_methods.Wachter import Wachter
from rocelib.recourse_methods.RNCE import RNCE
from rocelib.recourse_methods.MCER import MCER

from rocelib.evaluations.DistanceEvaluator import DistanceEvaluator


# from robustx.generators.robust_CE_methods.APAS import APAS
# from robustx.generators.robust_CE_methods.ArgEnsembling import ArgEnsembling
# from robustx.generators.robust_CE_methods.DiverseRobustCE import DiverseRobustCE
# from robustx.generators.robust_CE_methods.MCER import MCER
# from robustx.generators.robust_CE_methods.ModelMultiplicityMILP import ModelMultiplicityMILP
# from robustx.generators.robust_CE_methods.PROPLACE import PROPLACE
# from robustx.generators.robust_CE_methods.RNCE import RNCE
# from robustx.generators.robust_CE_methods.ROAR import ROAR
# from robustx.generators.robust_CE_methods.STCE import STCE

# METHODS = {"APAS": APAS, "ArgEnsembling": ArgEnsembling, "DiverseRobustCE": DiverseRobustCE, "MCER": MCER,
#            "ModelMultiplicityMILP": ModelMultiplicityMILP, "PROPLACE": PROPLACE, "RNCE": RNCE, "ROAR": ROAR,
#            "STCE": STCE, "BinaryLinearSearch": BinaryLinearSearch, "GuidedBinaryLinearSearch": GuidedBinaryLinearSearch,
#            "NNCE": NNCE, "KDTreeNNCE": KDTreeNNCE, "MCE": MCE, "Wachter": Wachter}
# EVALUATIONS = {"Distance": DistanceEvaluator, "Validity": ValidityEvaluator, "Manifold": ManifoldEvaluator,
#                "Delta-robustness": RobustnessProportionEvaluator}


METHODS = {
    "BinaryLinearSearch": BinaryLinearSearch,
    "NNCE": NNCE,
    "KDTreeNNCE": KDTreeNNCE,
    "MCE": MCE,
    "Wachter": Wachter,
    "RNCE": RNCE,
    "MCER": MCER
}
EVALUATIONS = {"Distance": DistanceEvaluator}


DATA_TYPES = {"DataFrame"}

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
        pos_instance = self._dataset.get_random_positive_instance(neg_value, column_name=column_name)

        # Loop until a positive instance whose prediction is positive is found
        while self.model.predict_single(pos_instance) == neg_value:
            pos_instance = self._dataset.get_random_positive_instance(neg_value, column_name=column_name)

        return pos_instance
    
    def generate(self, methods: List[str], type="DataFrame") -> List[pd.DataFrame]:
        """
        Generates counterfactual explanations for the specified methods and stores the results.

        @param methods: List of recourse methods (by name) to use for counterfactual generation.
        """
        all_generated_CEs = []  # List to store lists of generated counterfactuals

        for method in methods:
            try:
                # Check if the method exists in the dictionary
                if method not in METHODS:
                    raise ValueError(f"Recourse method '{method}' not found. Available methods: {list(METHODS.keys())}")

                # Instantiate the recourse method
                recourse_method = METHODS[method](self)  # Pass the classification task to the method

                # Start timer
                start_time = time.perf_counter()

                res = recourse_method.generate_for_all()  # Generate counterfactuals
                res_correct_type = self.convert_datatype(res, type)
                # End timer
                end_time = time.perf_counter()

                # Store the result in the counterfactual explanations dictionary
                self._CEs[method] = [res, end_time - start_time]  

                all_generated_CEs.append(res_correct_type)


            except Exception as e:
                print(f"Error generating counterfactuals with method '{method}': {e}")
            
        return all_generated_CEs
    
    def evaluate(self, methods: List[str], evaluations: List[str]):
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

        # Instantiate evaluators and evaluate
        for evaluation in evaluations:
            evaluator_class = EVALUATIONS[evaluation]

            try:
                # Create evaluator instance
                evaluator = evaluator_class(self, valid_methods)

                # Perform evaluation across all methods
                eval_scores = evaluator.evaluate()
                
                # Store results
                for method, score in eval_scores.items():
                    if method not in evaluation_results:
                        evaluation_results[method] = {}
                    evaluation_results[method][evaluation] = score

            except Exception as e:
                print(f"Error evaluating '{evaluation}': {e}")

        return evaluation_results
    

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