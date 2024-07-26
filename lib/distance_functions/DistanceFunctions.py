import pandas as pd
import numpy as np


def euclidean(x: pd.DataFrame, c: pd.DataFrame):
    return np.sqrt(np.sum((x.values - c.values) ** 2))


def manhattan(x: pd.DataFrame, c: pd.DataFrame):
    return np.sum(np.abs(x.values - c.values))
