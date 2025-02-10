import datetime
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from robustx.generators.CEGenerator import CEGenerator


class CostLoss(nn.Module):
    """
    Custom loss function to calculate the absolute difference between two tensors.

    Inherits from nn.Module.
    """

    def __init__(self):
        """
        Initializes the CostLoss module.
        """
        super(CostLoss, self).__init__()

    def forward(self, x1, x2):
        """
        Computes the forward pass of the loss function.

        @param x1: The first tensor (e.g., the original instance).
        @param x2: The second tensor (e.g., the counterfactual instance).
        @return: The absolute difference between x1 and x2.
        """
        dist = torch.abs(x1 - x2)
        return dist


class Wachter(CEGenerator):
    """
    A counterfactual explanation generator that uses Wachter's method for finding counterfactual explanations.

    Inherits from CEGenerator and implements the _generation_method to find counterfactuals
    using gradient descent.
    """

    def _generation_method(self, instance, column_name="target", neg_value=0, lamb=0.1, lr=0.02,
                           max_iter=10000000, max_allowed_minutes=0.5, epsilon=0.001, **kwargs):
        """
        Generates a counterfactual explanation using gradient descent, based on Wachter's method.

        @param instance: The input instance for which to generate a counterfactual. Provided as a Tensor.
        @param column_name: The name of the target column. (Not used in this method)
        @param neg_value: The value considered negative in the target variable.
        @param lamb: The tradeoff term in the loss function.
        @param lr: The learning rate for gradient descent.
        @param max_iter: The maximum number of iterations allowed for gradient descent.
        @param max_allowed_minutes: The maximum time allowed for the gradient descent process (in minutes).
        @param epsilon: A small constant used for the break condition.
        @param kwargs: Additional keyword arguments.
        @return: A DataFrame containing the counterfactual explanation if found, otherwise the original instance.
        """

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initialise the counterfactual search at the input point
        x = torch.Tensor(instance.to_numpy()).to(DEVICE)
        wac = Variable(x.clone(), requires_grad=True).to(DEVICE)

        # initialise an optimiser for gradient descent over the wac counterfactual point
        optimiser = Adam([wac], lr, amsgrad=True)

        # instantiate the two components of the loss function
        validity_loss = torch.nn.BCELoss()
        cost_loss = CostLoss()

        # TASK: specify target label y: either 0 or 1, depending on the original prediction
        # something like this
        y_target = torch.Tensor([1 - neg_value])

        # the total loss in the instructions: loss = validity_loss + lamb * cost_loss

        # compute class probability
        class_prob = self.task.model.predict_proba_tensor(wac)
        wac_valid = False
        iterations = 0
        if y_target == 0 and class_prob < 0.5 or y_target == 1 and class_prob >= 0.5:
            wac_valid = True

        # set maximum allowed time for computing 1 counterfactual
        t0 = datetime.datetime.now()
        t_max = datetime.timedelta(minutes=max_allowed_minutes)

        # start gradient descent
        while not wac_valid and iterations <= max_iter:

            optimiser.zero_grad()
            class_prob = self.task.model.predict_proba_tensor(wac)
            wac_loss = validity_loss(class_prob, y_target) + lamb * cost_loss(x, wac)
            wac_loss.sum().backward()
            optimiser.step()

            # break conditions
            p = class_prob[0].item()
            if (neg_value and p + epsilon < 0.5) or (not neg_value and p - epsilon >= 0.5):
                wac_valid = True
            if datetime.datetime.now() - t0 > t_max:
                break
            iterations += 1

        res = pd.DataFrame(wac.detach().numpy()).T
        res.columns = instance.index
        # if not self.task.model.predict_single(res.T):
        #     print("Failed!")
        #     pd.DataFrame(instance)

        return res
