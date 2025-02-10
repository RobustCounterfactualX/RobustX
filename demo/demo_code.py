import copy
import time
from matplotlib.lines import Line2D
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from robustx.datasets.ExampleDatasets import get_example_dataset
from robustx.generators.CE_methods.BinaryLinearSearch import BinaryLinearSearch
from robustx.generators.CE_methods.GuidedBinaryLinearSearch import GuidedBinaryLinearSearch
from robustx.generators.CE_methods.KDTreeNNCE import KDTreeNNCE
from robustx.generators.CE_methods.MCE import MCE
from robustx.generators.CE_methods.NNCE import NNCE
from robustx.generators.CE_methods.Wachter import Wachter
from robustx.generators.robust_CE_methods.APAS import APAS
from robustx.generators.robust_CE_methods.STCE import TrexNN
from robustx.generators.robust_CE_methods.MCER import MCER
from robustx.generators.robust_CE_methods.RNCE import RNCE
from robustx.lib.models.pytorch_models.SimpleNNModel import SimpleNNModel
from robustx.datasets.custom_datasets.CsvDatasetLoader import CsvDatasetLoader
from robustx.lib.tasks.ClassificationTask import ClassificationTask
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import streamlit as st
from robustx.robustness_evaluations.ApproximateDeltaRobustnessEvaluator import ApproximateDeltaRobustnessEvaluator
from robustx.robustness_evaluations.DeltaRobustnessEvaluator import DeltaRobustnessEvaluator
import warnings
warnings.filterwarnings("ignore")




def get_ces(input_instance, task, generators_not_robust):
    ces = []
    for method in generators_not_robust:
        print("Generating counterfactual explanation using", method)

        if method == "BinaryLinearSearch":
            generator = BinaryLinearSearch(task)
        elif method == "KDTreeNNCE":
            generator = KDTreeNNCE(task)
        elif method == "MCE":
            generator = MCE(task)
        elif method == "NNCE":
            generator = NNCE(task)
        elif method == "Wachter":
            generator = Wachter(task)

        ce = generator._generation_method(input_instance, neg_value=0)[['feature1', 'feature2']].values
        ces.append(ce[0])


    # Display results
    print("\nOriginal instance: ", input_instance.values)
    ces_dict = []
    for i, ce in enumerate(ces):
        print(f"{generators_not_robust[i]}: {ce}")
        ces_dict.append({generators_not_robust[i]: ce})

    return ces, ces_dict


def get_robust_ces(input_instance, task, delta_robust_methods, max_distance):
    robust_ces = []
    robust_ces_dict = []
   
    for method in delta_robust_methods:
        print("Generating counterfactual explanation using", method)

        if method == "MCER":
            generator = MCER(task)
            ce = generator._generation_method(input_instance, neg_value=0, delta=max_distance)[['feature1', 'feature2']].values
        elif method == "RNCE":
            generator = RNCE(task)
            ce = generator._generation_method(input_instance, neg_value=0, delta=max_distance)[['feature1', 'feature2']].values
        elif method == "STCE":
            generator = TrexNN(task)
            ce = generator._generation_method(input_instance, neg_value=0)[['feature1', 'feature2']].values
        elif method == "APAS":
            generator = APAS(task, CE_generator=Wachter , alpha=0.999)
            ce = generator._generation_method(input_instance, delta_max=max_distance, desired_outcome=1, maximum_iterations=100, verbose=False)[['feature1', 'feature2']].values

        robust_ces.append(ce[0])
        robust_ces_dict.append({method: ce[0]})

    print("\nOriginal instance: ", input_instance.values)
    for i, ce in enumerate(robust_ces):
        print(f"{delta_robust_methods[i]}: {ce}")

    return robust_ces, robust_ces_dict


def compute_max_delta(model, model_new):
    # compute the maximum weight distance between the original and new model
    weights = {}
    for i, layer in enumerate(model._model):
        if isinstance(layer, torch.nn.Linear):
            weights[f'fc{i}_weight'] = layer.weight
            weights[f'fc{i}_bias'] = layer.bias

    weights_new = {}
    for i, layer in enumerate(model_new._model):
        if isinstance(layer, torch.nn.Linear):
            weights_new[f'fc{i}_weight'] = layer.weight
            weights_new[f'fc{i}_bias'] = layer.bias

    max_distance = 0
    for key in weights:
        max_distance = max(max_distance, torch.dist(weights[key], weights_new[key], p=2).item())
    print("Maximum distance between the original and new model weights: ", max_distance)

    return max_distance


# Plot decision boundary and counterfactual point
def plot_decision_boundary_with_counterfactual(model, X, y, Z=None, original_point=None, counterfactual_points=None, updated=False, title='Decision Boundary', name_file="fig", visualize=False):
  
    x_min, x_max = X[:, 0].min() - 1.5, X[:, 0].max() + 1.5
    y_min, y_max = X[:, 1].min() - 1.5, X[:, 1].max() + 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000),np.linspace(y_min, y_max, 1000))
    xx_old, yy_old = np.meshgrid(np.linspace(-5.889530873443022, 4.94498246954976, 1000),np.linspace(-2.4760320018244264, 6.26774828920664, 1000))
    
    
    X_grid = np.c_[xx.ravel(), yy.ravel()]

    if Z is None:
        with torch.no_grad():
            Z = model(torch.FloatTensor(X_grid))
            Z = Z.reshape(xx.shape)
    
    else:
        with torch.no_grad():
            Z_new = model(torch.FloatTensor(X_grid))
            Z_new = Z_new.reshape(xx.shape)

    if updated: 
        plt.contourf(xx_old, yy_old, Z, alpha=0.5, levels=[0, 0.5, 1], cmap='coolwarm')
    else:
        plt.contourf(xx, yy, Z, alpha=0.7, levels=[0, 0.5, 1], cmap='coolwarm')

    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap='coolwarm')
    if updated: 
        plt.contour(xx_old, yy_old, Z, levels=[0, 0.5], colors='k', linestyles='--', linewidths=2.0)
        plt.contour(xx, yy, Z_new, levels=[0, 0.5], colors='k', linestyles='-', linewidths=2.0)

        legend_elements = [ Line2D([0], [0], color='black', linestyle='--', lw=1.0, label="original decision boundary"),
                            Line2D([0], [0], color='black', lw=2, label="new decision boundary")]
        
        plt.contourf(xx, yy, Z_new, alpha=0.8, levels=[0, 0.5, 1], cmap='coolwarm')

    else:
        plt.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='-', linewidths=1.0, label='Decision Boundary')

    if original_point is not None and counterfactual_points is not None:

        plt.scatter(original_point[0], original_point[1], facecolors='yellow', edgecolors='black', s=100, label='Original Point')

        if updated: legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Original Point'))
        colors = sns.color_palette('husl', n_colors=len(counterfactual_points))

        for i, ce in enumerate(counterfactual_points):
            ce_name = list(ce.keys())[0]
            ce = list(ce.values())[0]
            
            plt.scatter(ce[0], ce[1], color=colors[i], marker='x', s=100, label=f'{ce_name} CE', linewidths=2.0)

            if updated: 
                legend_elements.append(Line2D([0], [0], marker='x', color=colors[i], label=f'{ce_name} CE', markersize=10))
                plt.plot([original_point[0], ce[0]], [original_point[1], ce[1]], 'k-', linewidth=1.0)
            else:
                # Draw a dotted line connecting original and counterfactual points
                plt.plot([original_point[0], ce[0]], [original_point[1], ce[1]], 'k--')

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    if counterfactual_points is not None: plt.legend()

    if updated:
        # Add the proxy artists (custom lines) to the legend
        plt.legend(handles=legend_elements, loc='upper right',fontsize='small')
        plt.xlim(-4.889530873443022, 4.94498246954976)
        plt.ylim(-2.4760320018244264, 6.06774828920664)
    

    if visualize: 
        plt.show()
    else:
        plt.savefig(f"images/{name_file}.png")
    return Z

def main(plot = True):  

    model = SimpleNNModel(2, [8, 8], 1, seed=0)   
    csv_path = "dataset/custom_dataset.csv"  # Path to the CSV file
    target_column = "target"  # Name of the target column

    # Create an instance of CsvDatasetLoader
    dataset_loader = CsvDatasetLoader(csv=csv_path, target_column=target_column)
  
    # create a train-test split
    X_train, X_test, y_train, y_test = train_test_split(dataset_loader.data.drop(columns=[target_column]), dataset_loader.data[target_column], test_size=0.2, random_state=0)

    accuracy = model.compute_accuracy(X_test.values, y_test.values)

    # Display results
    print("Model accuracy before training accuracy: ", accuracy)

    model.train(X_train, y_train, epochs=100)
    accuracy = model.compute_accuracy(X_test.values, y_test.values)

    # Display results
    print("Model accuracy after training accuracy: ", accuracy)
    print()

    if plot:
        Z = plot_decision_boundary_with_counterfactual(model._model, X_test.values, y_test.values, title='Decision Boundary', name_file="decision_boundary", visualize=True)

    # Create a classification task
    task = ClassificationTask(model, dataset_loader)


    ############################ GENERATION CES ############################

    print("====================================\n")
    # Get the counterfactual explanation
    # first select an input instance from the test set that has been classified as 0 (loan not approved)
    input_instance = X_test[y_test == 0].iloc[0]

    generators_not_robust = ["BinaryLinearSearch", "KDTreeNNCE", "MCE", "NNCE", "Wachter"]
    #generators_not_robust = ["KDTreeNNCE", "MCE", "NNCE", "Wachter"]
    _, ces_dict = get_ces(input_instance, task, generators_not_robust)
    
    ############################ RETRAINING ############################
    print("====================================\n")
    # extend the dataset and retrain the model
    X_new, y_new = make_classification(n_samples=1000, n_features=2, n_classes=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,class_sep=2.0, flip_y=0, random_state=2001)
   
    df_new = pd.DataFrame(X_new, columns=['feature1', 'feature2'])
    df_new['target'] = y_new
    
    # Combine the old and new datasets
    df_combined = pd.concat([dataset_loader.data, df_new], ignore_index=True)
    df_combined.to_csv("dataset/combined_dataset.csv", index=False)

    # Create an instance of CsvDatasetLoader
    dataset_loader_new = CsvDatasetLoader(csv="dataset/combined_dataset.csv", target_column=target_column)
    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(dataset_loader_new.data.drop(columns=[target_column]), dataset_loader_new.data[target_column], test_size=0.2, random_state=2001)

    model_new = copy.deepcopy(model)
    model_new.train(X_train_new, y_train_new, epochs=100)
    accuracy = model_new.compute_accuracy(X_test_new.values, y_test_new.values)
    print("Model accuracy after retraining: ", accuracy)


 
    ############################ GENERATION ROBUST CES ############################
    print("====================================\n")
   
    #max_distance = compute_max_delta(model, model_new)
    delta_max = 0.05

    # Get the robust counterfactual explanation
    delta_robust_methods = ["RNCE"]
    _, rob_ces_dict = get_robust_ces(input_instance, task, delta_robust_methods, max_distance=delta_max)


    complete_ces_dict = ces_dict + rob_ces_dict
    methods = [list(d.keys())[0] for d in complete_ces_dict]


    name_file = "decision_boundary_with_ces_"
    for i, method in enumerate(methods):
        if i == len(methods) - 1:
            name_file += method
        else:
            name_file += method + "_"
    
    if plot: 

        _ = plot_decision_boundary_with_counterfactual(model._model, X_test.values, y_test.values, Z, original_point=input_instance.values, counterfactual_points=complete_ces_dict, updated=False, title='Decision Boundary with Counterfactuals', name_file=f"{name_file}_before_retraining", visualize=True)
     
        
        print(name_file+"_after_retraining")
        _ = plot_decision_boundary_with_counterfactual(model_new._model, X_test_new.values, y_test_new.values, Z, original_point=input_instance.values, counterfactual_points=complete_ces_dict, updated=True, title='Decision Boundary with Counterfactuals after retraining', name_file=f"{name_file}_after_retraining", visualize=True)

    quit()
    # evaluate the robustness of the counterfactual explanations
    print("====================================\n")
    print("Evaluation of the robustness of the counterfactual explanations")
    task = ClassificationTask(model_new, dataset_loader_new)
   
    mce_ce = ces_dict[2]["MCE"]
    print("MCE countefactual explanation:", mce_ce)
    print()

    evaluator = ApproximateDeltaRobustnessEvaluator(task)
    mce_ce = torch.FloatTensor(mce_ce)
    print(f"The counterfactual {mce_ce} is robust against Δ={delta_max} model changes with a confidence 99.9%" if evaluator.evaluate(mce_ce, desired_outcome=1, delta=delta_max) else f"The counterfactual {mce_ce} is NOT robust against Δ=0.05 model changes with a confidence 99.9%")


    evaluator = DeltaRobustnessEvaluator(task)
    print(f"The counterfactual {mce_ce} is robust against Δ={delta_max} model changes" if evaluator.evaluate(mce_ce, desired_output=1, delta=delta_max) else f"The counterfactual {mce_ce} is NOT robust against Δ=0.05 model changes")


if __name__ == "__main__":
    main(plot=True)
    







