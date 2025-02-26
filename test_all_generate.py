import numpy as np
import pandas as pd
import torch
from test_helpers.TestingModels import TestingModels  # Import your testing_models class


def test_correct_recourses_generated_for(testing_models) -> None:
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
    ces = ct.generate(["BinaryLinearSearch",
    "GuidedBinaryLinearSearch",
    "NNCE",
    "KDTreeNNCE",
    "MCE",
    "Wachter",
    "RNCE",
    "MCER",
    "STCE"])
    evals = ct.evaluate(["BinaryLinearSearch",
    "GuidedBinaryLinearSearch",
    "NNCE",
    "KDTreeNNCE",
    "MCE",
    "Wachter",
    "RNCE",
    "MCER",
    "STCE"], ["Validity", "Distance", "DeltaRobustnessEvaluator"])
    # ces = ct.generate(["MCE", "BinaryLinearSearch", "NNCE", "KDTreeNNCE", "Wachter", "RNCE", "MCER", "RoCourseNet", "STCE"])
    # print(ces)
    # evals = ct.evaluate(["MCE", "BinaryLinearSearch", "NNCE", "KDTreeNNCE", "Wachter", "RNCE", "MCER", "RoCourseNet", "STCE"], ["Distance", "Validity", "De"])
    # assert False



if __name__ == "__main__":
    testing_models = TestingModels()  # Initialize your testing_models
    test_correct_recourses_generated_for(testing_models)