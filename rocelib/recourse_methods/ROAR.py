import pandas as pd
import numpy as np
from scipy.optimize import linprog
from rocelib.recourse_methods.RecourseGenerator import RecourseGenerator
import torch
from torch.autograd import Variable, grad
import torch.optim as optim


class ROAR(RecourseGenerator):
    """
    A CE generator that uses gradient based robust optimisation, targeting robustness to model changes,
    for linear models.
    Below codes are adapted from the carla-recourse library
    https://github.com/carla-recourse/CARLA:
    https://github.com/carla-ce/CARLA/blob/main/carla/ce_methods/catalog/roar/library/roar.py

    Attributes:
        _task (Task): The task to solve, inherited from CEGenerator. The classifier should be a linear model.
    """

    def _generation_method(self, instance,
                           column_name="target", neg_value=0, lambda_param=0.01, lr=0.01, delta=0.01,
                           norm=1, seed=0, loss_threshold=0.0001, **kwargs):
        """
        Generate CE

        Args:
            instance: The instance for which to generate a counterfactual. Can be a DataFrame or Series.
            column_name: The name of the target column.
            neg_value: The value considered negative in the target variable.
            lambda_param: Hyperparameter. Trade-off between cost and validity loss terms.
            lr: Hyperparameter. Learning rate for gradient descent.
            delta: Hyperparameter. The tolerance for robustness in the model parameter space.
            norm: Hyperparameter. The norm used in costs.
            seed: Random seed.
            loss_threshold: Threshold for loss difference.
            **kwargs: Additional keyword arguments.

        Returns: A DataFrame containing the counterfactual explanation.

        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(seed)

        coeff = torch.from_numpy(self.task.model.model.coef_[0]).float().to(device)
        intercept = torch.from_numpy(np.asarray(self.task.model.model.intercept_)).float().to(device)
        x = torch.from_numpy(instance.values).float().to(device)
        y_target = torch.tensor(1 - neg_value).float().to(device)
        lamb = torch.tensor(lambda_param).float().to(device)

        # x_new is used for gradient search in optimizing process
        x_new = Variable(x.clone(), requires_grad=True)

        optimizer = optim.Adam([x_new], lr=lr, amsgrad=True)

        target_class = y_target.clone().detach().requires_grad_(True)
        loss_fn = torch.nn.MSELoss()
        loss = torch.tensor(0)
        loss_diff = loss_threshold + 1

        while loss_diff > loss_threshold:
            loss_prev = loss.clone().detach()
            delta_W, delta_W0 = self._get_worse_perturbation(
                x_new.squeeze(), coeff, intercept, delta, target_class
            )
            delta_W, delta_W0 = (
                torch.from_numpy(delta_W).float().to(device),
                torch.from_numpy(delta_W0).float().to(device),
            )
            optimizer.zero_grad()
            f_x_new = torch.nn.Sigmoid()(
                torch.matmul(coeff + delta_W, x_new.squeeze()) + intercept + delta_W0
            ).squeeze()
            f_x_new = torch.log(f_x_new / (1 - f_x_new))
            cost = (torch.dist(x_new, x, norm))
            loss = loss_fn(f_x_new, target_class) + lamb * cost
            loss.backward()
            optimizer.step()
            loss_diff = torch.dist(loss_prev, loss, 2)

        return pd.DataFrame(x_new.cpu().detach().numpy().reshape(1, -1))

    def _get_worse_perturbation(self, ce, coeff, intercept, delta, target_class):
        """
        Calculate the worst case perturbation of model parameters

        Args:
            ce: Counterfactual Explanation.
            coeff: Coefficients of the linear model.
            intercept: Intercept of the linear model.
            delta: The tolerance for robustness in the model parameter space.
            target_class: Desirable target class.
        """
        W = torch.cat((coeff, intercept), 0)  # Add intercept to weights
        ce = torch.cat(
            (ce, torch.ones(1, device=ce.device)), 0
        )  # Add 1 to the feature vector for intercept

        loss_fn = torch.nn.BCELoss()
        W.requires_grad = True
        f_x_new = torch.nn.Sigmoid()(torch.matmul(W, ce))
        w_loss = loss_fn(f_x_new, target_class)

        gradient_w_loss = grad(w_loss, W)[0]
        c = list(np.array(gradient_w_loss.cpu()) * np.array([-1] * len(gradient_w_loss)))
        bound = (-delta, delta)
        bounds = [bound] * len(gradient_w_loss)

        res = linprog(c, bounds=bounds)

        delta_opt = res.x
        delta_W, delta_W0 = np.array(delta_opt[:-1]), np.array([delta_opt[-1]])

        return delta_W, delta_W0
