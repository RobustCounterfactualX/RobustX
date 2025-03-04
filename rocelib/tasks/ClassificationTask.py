import os
import zipfile

import pandas as pd
import time
import torch
import numpy as np
from tabulate import tabulate  # For better table formatting
import matplotlib.pyplot as plt

from rocelib.datasets.DatasetLoader import DatasetLoader
from rocelib.models.TrainedModel import TrainedModel
from rocelib.tasks.Task import Task
from typing import List, Dict, Any, Union, Tuple

from rocelib.recourse_methods.ArgEnsembling import ArgEnsembling
from rocelib.recourse_methods.RecourseGenerator import RecourseGenerator
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
from rocelib.recourse_methods.ModelMultiplicityMILP import ModelMultiplicityMILP
from rocelib.recourse_methods.APAS import APAS
from rocelib.recourse_methods.DiverseRobustCE import DiverseRobustCE
from rocelib.recourse_methods.PROPLACE import PROPLACE

# Evaluators
from rocelib.evaluations.robustness_evaluations.Evaluator import Evaluator
from rocelib.evaluations.ManifoldEvaluator import ManifoldEvaluator
from rocelib.evaluations.DistanceEvaluator import DistanceEvaluator
from rocelib.evaluations.ValidityEvaluator import ValidityEvaluator
from rocelib.evaluations.robustness_evaluations.MC_Robustness_Implementations.DeltaRobustnessEvaluator import DeltaRobustnessEvaluator
from rocelib.evaluations.robustness_evaluations.MC_Robustness_Implementations.VaRRobustnessEvaluator import VaRRobustnessEvaluator
from rocelib.evaluations.robustness_evaluations.ModelMultiplicityRobustnessEvaluator import ModelMultiplicityRobustnessEvaluator
from rocelib.evaluations.robustness_evaluations.NE_Robustness_Implementations.InvalidationRateRobustnessEvaluator import InvalidationRateRobustnessEvaluator
from rocelib.evaluations.robustness_evaluations.MM_Robustness_Implementations.MultiplicityValidityRobustnessEvaluator import MultiplicityValidityRobustnessEvaluator
from rocelib.evaluations.robustness_evaluations.IC_Robustness_Implementations.SetDistanceRobustnessEvaluator import SetDistanceRobustnessEvaluator


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


    def __init__(self, model: TrainedModel, dataset: DatasetLoader, mm_models: Dict[str, TrainedModel] = None):
        super().__init__(model, dataset, mm_models)
        self.methods = {
            "BinaryLinearSearch": BinaryLinearSearch,
            # "GuidedBinaryLinearSearch": GuidedBinaryLinearSearch,
            "MMMILP": ModelMultiplicityMILP,
            "NNCE": NNCE,
            "KDTreeNNCE": KDTreeNNCE,
            "MCE": MCE,
            "Wachter": Wachter,
            "RNCE": RNCE,
            "MCER": MCER,
            "RoCourseNet": RoCourseNet,
            "STCE": TrexNN,
            "APAS": APAS,
            "ArgEnsembling": ArgEnsembling,
            "DiverseRobustCE": DiverseRobustCE,
            "PROPLACE": PROPLACE
        }

        self.evaluation_metrics = {
            "Distance": DistanceEvaluator,
            "Validity": ValidityEvaluator,
            "ModelMultiplicityRobustness": MultiplicityValidityRobustnessEvaluator,
            "DeltaRobustnessEvaluator": DeltaRobustnessEvaluator,
            "InvalidationRateRobustnessEvaluator": InvalidationRateRobustnessEvaluator,
            "SetDistanceRobustnessEvaluator": SetDistanceRobustnessEvaluator,
            "ManifoldEvaluator": ManifoldEvaluator,
            # "VaRRobustnessEvaluator": VaRRobustnessEvaluator
        }

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

    def generate(self, methods: List[str]=None, type="DataFrame", **kwargs) -> Dict[str, Tuple[pd.DataFrame, float]]:
        """
        Generates counterfactual explanations for the specified methods and stores the results.

        @param methods: List of recourse methods (by name) to use for counterfactual generation. If not provided, then counterfactuals will be generated for all methods
        @param type: The datatype your instances are in e.g. dataframe, nparray, tensor
        @return: A dictionary from recourse method to a tuple of (Pandas dataframe holding the counterfactual, time taken to generate the counterfactual)
        """

        if methods is None:
            methods = self.get_recourse_methods()

        for method in methods:
            print(f"Generating for {method}")

            try:
                # Check if the method exists in the dictionary
                if method not in self.methods:
                    raise ValueError(f"Recourse method '{method}' not found. Available methods: {list(self.methods.keys())}")

                # Instantiate the recourse method
                recourse_method = self.methods[method](self)  # Pass the classification task to the method

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

    def generate_mm(self, methods: List[str]=None, type="DataFrame", **kwargs) -> Dict[str, Dict[str, Tuple[pd.DataFrame, float]]]:
        """
        Generates counterfactual explanations for the specified methods for each of the stored models and stores the results.

        @param methods: List of recourse methods (by name) to use for counterfactual generation.
        @return: A nested dictionary from recourse method to model name to a tuple of (Pandas dataframe holding the counterfactual, time taken to generate the counterfactual)
        """
        if methods is None:
            methods = self.get_recourse_methods()

        if not self.mm_flag:
            raise ValueError("Multiple models must be added in order to generate for MM")

        for method in methods:
            print(f"Generating for {method}")

            for i, model_name in enumerate(self.mm_models):
                ces = self.generate_for_model_method(model_name, method, type, **kwargs)
                if i == 0:
                    # Primary model so we should store results in self._CEs
                    self._CEs[method] = ces

                # Store results in mm_CEs
                if method not in self.mm_CEs:
                    self.mm_CEs[method] = {}
                self.mm_CEs[method][model_name] = ces

        return self.mm_CEs


    def generate_for_model_method(self, model_name, method, type, **kwargs) -> Tuple[pd.DataFrame, float]:
        print(f"GENERATING FOR: model: {model_name}, method: {method}")
        try:
            # Check if the method exists in the dictionary
            if method not in self.methods:
                raise ValueError(f"Recourse method '{method}' not found. Available methods: {list(self.methods.keys())}")

            # Instantiate the recourse method
            task = ClassificationTask(self.mm_models[model_name], dataset=self.dataset, mm_models=self.mm_models)
            recourse_method = self.methods[method](task)  # Pass the classification task to the method

            # Start timer
            start_time = time.perf_counter()

            res = recourse_method.generate_for_all(**kwargs)  # Generate counterfactuals
            res_correct_type = self.convert_datatype(res, type)
            # End timer
            end_time = time.perf_counter()

            # Store the result in the counterfactual explanations dictionary
            return [res, end_time - start_time]

        except Exception as e:
            print(f"Error generating counterfactuals with method '{method}': {e}")
            return None



    def evaluate(self, methods: List[str]=None, evaluations: List[str]=None, visualisation=False, **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Evaluates the generated counterfactual explanations using specified evaluation metrics.

        @param methods: List of recourse methods to evaluate.
        @param evaluations: List of evaluation metrics to apply.
        @return: Dictionary containing evaluation results per method and metric.
        """
        if methods is None:
            methods = self.get_recourse_methods()
        if evaluations is None:
            evaluations = self.get_evaluation_metrics()

        evaluation_results = {}

        # Validate evaluation names
        invalid_evaluations = [ev for ev in evaluations if ev not in self.evaluation_metrics]
        if invalid_evaluations:
            raise ValueError(f"Invalid evaluation metrics: {invalid_evaluations}. Available: {list(self.evaluation_metrics.keys())}")

        # Filter out methods that haven't been generated
        valid_methods = [method for method in methods if method in self._CEs]

        if valid_methods != methods:
            print(f"generate has not been called for {list(set(methods) - set(valid_methods))} so evaluations were not performed for these")

        # Filter out methods that haven't been generated for MM if mm_flag is on
        mm_metric = [isinstance(self.evaluation_metrics[metric](self), ModelMultiplicityRobustnessEvaluator) for metric in evaluations]
        if any(mm_metric):
            if not self.mm_flag:
                print("Multiple models must be added to the task in order to evaluate model multiplicity")
                #Remove the MM metrics from this evaluation
                evaluations = [metric for (i,metric) in enumerate(evaluations) if not mm_metric[i]]
            else:
                valid_methods = [method for method in methods if (method in self.mm_CEs and len(self.mm_CEs[method].keys()) == len(self.mm_models))]
                if not valid_methods:
                    print("No valid methods have been generated for MM for evaluation. Call generate_mm for these methods")
                    return evaluation_results
                print(f"generate_mm has not been called for {list(set(methods) - set(valid_methods))} so evaluations were not performed for these")

        if not valid_methods:
            print("No valid methods have been generated for evaluation.")
            return evaluation_results

        # Perform evaluation
        for evaluation in evaluations:
            print(f"Evaluation technique {evaluation}")
            evaluator_class = self.evaluation_metrics[evaluation]

            
                # Create evaluator instance
            evaluator = evaluator_class(self)

            for method in valid_methods:
                try:
                    print(f"Method: {method}")

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
                    print(f"'{method}': Error evaluating '{evaluation}' for : {e}")

            

        # Print results in table format
        table_data, headers = self._print_evaluation_results(evaluation_results, evaluations)
        csv_filename = "recourse_evaluations.csv"
        time.sleep(2)
        graph_filename = self._visualise_results(evaluation_results, evaluations, visualisation)

        self.save_evals_as_zip(table_data, headers, csv_filename, graph_filename)

        return evaluation_results
    
    def _visualise_results(self, evaluations_results: Dict[str, Dict[str, Any]], evaluations: List[str], visualisation = False):
        if not evaluations_results:
            print("No evaluation results to display.")
            return

        if len(evaluations) > 3:
            return self._visualise_results_radar_chart(evaluations_results, evaluations, visualisation)
        else:
            return self._visualise_results_bar_chart(evaluations_results, evaluations, visualisation)

    def _visualise_results_bar_chart(self, evaluation_results: Dict[str, Dict[str, Any]], evaluations: List[str], visualisation = False):
        recourse_methods = list(evaluation_results.keys())

        # Extract metric values
        metric_values = {method: [evaluation_results[method].get(metric, 0) for metric in evaluations] for method in recourse_methods}

        x = np.arange(len(recourse_methods))
        width = 0.2
        fig, ax = plt.subplots(figsize=(8, 6))

        for i, metric in enumerate(evaluations):
            values = [metric_values[method][i] for method in recourse_methods]
            ax.bar(x + i * width, values, width, label=metric)

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(recourse_methods)
        ax.set_xlabel("Recourse Methods")
        ax.set_ylabel("Metric Values")
        ax.set_title("Bar Chart of Evaluation Metrics")
        ax.legend()
        filename = "/tmp/evaluation_chart.png"
        plt.savefig(filename)  # Save the figure instead of displaying it

        if visualisation:
            plt.show()
        plt.close(fig)  # Ensure the figure is closed properly

        return filename



    def _visualise_results_radar_chart(self, evaluation_results: Dict[str, Dict[str, Any]], evaluations: List[str], visualisation = False):
        """
        Generate a radar chart for evaluation results.

        Parameters:
            evaluation_results (Dict[str, Dict[str, Any]]):
                A dictionary where keys are recourse methods, and values are dictionaries mapping metric names to values.
            evaluations (List[str]):
                A list of metric names to be visualized (must have at least 4 metrics).
        """
        assert len(evaluations) >= 4, "There must be at least 4 evaluation metrics to plot a radar chart."

        # Extract recourse methods
        recourse_methods = list(evaluation_results.keys())

        # Extract metric values for each recourse method
        metric_values = {method: [evaluation_results[method].get(metric, 0) for metric in evaluations]
                         for method in recourse_methods}

        # Define radar chart angles
        num_vars = len(evaluations)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # Close the radar chart loop
        angles += angles[:1]

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # Plot each recourse method
        for method, values in metric_values.items():
            values += values[:1]  # Close the loop
            ax.plot(angles, values, label=method, linewidth=2)
            ax.fill(angles, values, alpha=0.2)

        # Add labels and legend
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(evaluations, fontsize=12)
        ax.set_yticklabels([])
        ax.set_title("Radar Chart of Evaluation Metrics", fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        filename = "/tmp/evaluation_chart.png"
        plt.savefig(filename)  # Save the figure instead of displaying it

        plt.pause(0.001)  # Allows the figure to be shown briefly without blocking

        if visualisation:
            plt.show()

        plt.close(fig)  # Closes the figure to avoid blocking execution

        return filename

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
        return table_data, headers


    def save_evals_as_zip(self, table_data, headers, csv_filename, graph_filename):
        # Create DataFrame
        df = pd.DataFrame(table_data, columns=headers)

        # Save to CSV
        df.to_csv(csv_filename, index=False)

        # Create a zip file containing the CSV and graph files
        zip_filename = "evaluations.zip"  # Output zip file name
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            # Add the CSV file to the zip
            zipf.write(csv_filename, os.path.basename(csv_filename))
            # Add the graph image to the zip
            zipf.write(graph_filename, os.path.basename(graph_filename))



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

    def add_recourse_method(self, method_name: str, method_class: RecourseGenerator):
        self.methods[method_name] = method_class

    def add_evaluation_metric(self, metric_name: str, metric_class: Evaluator):
        self.evaluation_metrics[metric_name] = metric_class
