import numpy as np
import pandas as pd

from rocelib.lib.distance_functions.DistanceFunctions import euclidean, manhattan


def test_euclidean():
    # Test case 1: Known input
    df1 = pd.DataFrame([[1, 2], [3, 4]])
    df2 = pd.DataFrame([[1, 2], [3, 4]])
    assert euclidean(df1, df2) == 0  # The distance between identical DataFrames should be 0

    # Test case 2: Simple example
    df1 = pd.DataFrame([[0, 0]])
    df2 = pd.DataFrame([[3, 4]])
    assert euclidean(df1, df2) == 5  # Euclidean distance between (0, 0) and (3, 4) is 5

    # Test case 3: Another example
    df1 = pd.DataFrame([[1, 1]])
    df2 = pd.DataFrame([[4, 5]])
    assert np.isclose(euclidean(df1, df2), 5)  # Distance between (1,1) and (4,5) is 5


def test_manhattan():
    # Test case 1: Known input
    df1 = pd.DataFrame([[1, 2], [3, 4]])
    df2 = pd.DataFrame([[1, 2], [3, 4]])
    assert manhattan(df1, df2) == 0  # The distance between identical DataFrames should be 0

    # Test case 2: Simple example
    df1 = pd.DataFrame([[0, 0]])
    df2 = pd.DataFrame([[3, 4]])
    assert manhattan(df1, df2) == 7  # Manhattan distance between (0, 0) and (3, 4) is 7

    # Test case 3: Another example
    df1 = pd.DataFrame([[1, 1]])
    df2 = pd.DataFrame([[4, 5]])
    assert manhattan(df1, df2) == 7  # Distance between (1,1) and (4,5) is 7
