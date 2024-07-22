import pandas as pd
import numpy as np
import torch

def generate_counterfactuals_binary_linear_search(model, negative, random_pos_instance, column_name="target",
                                                  gamma=0.1, distance_func="euclidean") -> pd.DataFrame:
    """
    Generates a CE for a given instance, negative, and returns it
    :param model: The model you wish to use to generate the CEs, BaseModel
    :param negative: The instance for which you wish to generate a CE, DataFrame
    :param random_pos_instance: A random instance which is classified as positive by the model, DataFrame
    :param column_name: The name of the column which stores the target variable
    :param gamma: Distance threshold parameter, higher gamma means CE generated is closer to original negative instance
    :param distance_func: The name of the distance function you wish to use, default is l2 ('l1' / 'manhattan', 'l2' / 'euclidean')
    :return:
    """

    # Get initial counterfactual
    c = random_pos_instance.T

    # Make sure column names are same so return result has same indices
    negative = negative.to_frame()
    c.columns = negative.columns

    # Decide which distance function to use
    if distance_func == "l1" or distance_func == "manhattan":
        dist = l1
    else:
        dist = euclidean

    # Loop until CE is under gamma threshold
    while dist(negative, c) > gamma:

        # Calculate new CE by finding midpoint
        new_neg = c.add(negative, axis=0) / 2

        # Reassign endpoints based on model prediction
        if model.predict_single(new_neg.T) == model.predict_single(negative.T):
            negative = new_neg
        else:
            c = new_neg

    # Form the dataframe
    ct = c.T

    # Store model prediction in return CE (this should ALWAYS be the positive value)
    res = model.predict(ct)

    if isinstance(res, torch.Tensor):
        res = pd.DataFrame(res.detach().numpy())

    ct[column_name] = res

    # Store the loss
    ct["loss"] = dist(negative, c)

    return ct


def compute_nnce(model, train_set, negative, distance_func="euclidean") -> pd.DataFrame:
    """
    Function to compute NNCE.

    Parameters:
        :param model: The model you wish to use to generate the CEs, type BaseModel
        :param train_set: The DatasetLoader training set used to train the model
        :param negative: The instance for which you wish to generate a CE, DataFrame
        :param distance_func: The name of the distance function you wish to use, default is l2 ('l1' / 'manhattan', 'l2' / 'euclidean')

    Returns:
        nnce (Tensor): Nearest neighbour counterfactual explanation, an 1-d array of shape (k,)
    """

    # Convert X values of dataset to tensor
    X_tensor = torch.tensor(train_set.X.values, dtype=torch.float32)

    # Get all model predictions of model, turning them to 0s or 1s
    model_labels = model.predict(X_tensor)
    model_labels = (model_labels >= 0.5)

    # Determine the target label
    y = 1 if model.predict_single(negative) >= 0.5 else 0
    nnce_y = 1 - y

    # Set initial CE and minimum distance of CE
    nnce = None
    nnce_dist = np.inf

    # Decide distance function to use
    if distance_func == "l1" or distance_func == "manhattan":
        dist = l1
    else:
        dist = euclidean

    if isinstance(negative, pd.Series):
        negative_df = negative.to_frame()
    else:
        negative_df = negative

    # Iterate through each model prediction
    for sample, label in zip(train_set.X.values, model_labels):

        # Skip if the current instance is not the desired outcome
        if label != nnce_y:
            continue

        # Calculate distance between negative instance and current sample
        sample_dist = dist(negative_df, pd.DataFrame(sample))

        # If distance is less than any other encountered yet, we have found a new NNCE
        if sample_dist < nnce_dist:
            nnce = sample
            nnce_dist = sample_dist

    nnce_df = pd.DataFrame(nnce)
    nnce_df.index = negative.to_frame().index
    nnce_df = nnce_df.T
    nnce_df["Loss"] = nnce_dist

    return nnce_df.T


def euclidean(x: pd.DataFrame, c: pd.DataFrame):
    return np.sqrt(np.sum((x.values - c.values) ** 2))


def l1(x: pd.DataFrame, c: pd.DataFrame):
    return np.sum(np.abs(x.values - c.values))
