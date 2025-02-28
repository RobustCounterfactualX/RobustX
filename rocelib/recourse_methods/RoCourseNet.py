import pandas as pd
import numpy as np
import torch
from rocelib.recourse_methods.RecourseGenerator import RecourseGenerator
from rocelib.lib.distance_functions.DistanceFunctions import euclidean
from rocelib.evaluations.robustness_evaluations.MC_Robustness_Implementations.DeltaRobustnessEvaluator import DeltaRobustnessEvaluator
from rocelib.tasks.Task import Task
from rocelib.models.TrainableModel import TrainableModel


class RoCourseNet(RecourseGenerator):
    """
    A recourse generator using the RoCourseNet methodology, integrated with the TrainablePyTorchModel.
    """

    def __init__(self, task: Task):
        super().__init__(task)
        self.intabs = DeltaRobustnessEvaluator(task)
        self.model = task.model  # Should be your TrainablePyTorchModel or any TrainableModel

    def _generation_method(
        self, 
        instance: pd.Series, 
        gamma=0.1,        # Not always used; leftover param
        column_name="target", 
        neg_value=0,
        distance_func=euclidean, 
        max_iter=50,      # n_steps for CF
        lr=0.1, 
        delta=0.01,       # max_delta for adv
        n_attacker_steps=10,
        lambda_=0.5,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generates a single counterfactual for 'instance' using a local linear approximation
        and an adversarial approach to the surrogate parameters.
        """
        # 1) Convert the single instance from pd.Series to a NumPy array x
        x_np = instance.drop(labels=[column_name], errors='ignore').values.astype(float)
        x_shape = x_np.shape

        # 2) Build a local linear approximation around x_np.
        #    Let's do a simple "gradient-based linearization" on log-odds:
        #    w, b s.t. approx f(x') = sigmoid( w^T x' + b ).
        w_approx, b_approx = self._local_linear_approx(x_np)

        # 3) The target label we want (binary flip).
        #    We'll assume your model’s predict_proba() returns a 2D array: [p0, p1].
        pred_probs = self._predict_proba_single(x_np.reshape(1, -1))
        y_pred = 1 if pred_probs[0, 1] > 0.5 else 0
        y_target = 1.0 - y_pred  # Flip from 0->1 or 1->0

        # 4) Initialize cf to x's value
        cf = x_np.copy()

        # 5) Optimization loop
        for _ in range(max_iter):
            # 5a) Adversarial step on w
            w_pert = self._adversarial_step(cf, w_approx, b_approx, y_target, delta, n_attacker_steps)

            # 5b) Update cf by taking a gradient step w.r.t.:
            #     loss = BCE( sigma( (w + w_pert) dot cf + b_approx ), y_target ) + lambda_ * ||cf - x||^2
            cf = self._gradient_update_cf(cf, x_np, w_approx + w_pert, b_approx, y_target, lr, lambda_)

        # 6) Return the final CF as a single-row DataFrame
        #    Make sure it has the same columns as your original data
        cf_df = pd.DataFrame([cf], columns=instance.drop(labels=[column_name], errors='ignore').index)
        # Optionally, put a final "target" column or so
        cf_df[column_name] = y_target
        return cf_df


    def _local_linear_approx(self, x_np: np.ndarray):
        # 1) Make a LEAF tensor of shape (1, d)
        x_leaf = torch.tensor(x_np.reshape(1, -1),
                            dtype=torch.float32,
                            requires_grad=True)  # this is a leaf

        # 2) Forward pass
        out = self.model.model(x_leaf)  # shape (1,1)

        # 3) Convert out to log-odds
        p_val = out.item()  # float in (0,1)
        eps = 1e-7
        p_val = np.clip(p_val, eps, 1 - eps)
        logit_val = np.log(p_val / (1.0 - p_val))

        # 4) Now compute gradient wrt x_leaf
        out_logit = torch.log(out / (1 - out + 1e-12))  
        out_logit.backward()

        # 5) Get gradient
        # Because x_leaf is shape (1,d), x_leaf.grad is shape (1,d).
        # So x_leaf.grad[0] is shape (d,)
        grad_w = x_leaf.grad[0].detach().numpy()

        # 6) w, b for local linear approximation
        w = grad_w
        x0 = x_np.reshape(-1)  # shape (d,)
        b = logit_val - np.dot(w, x0)

        return w, b


    def _adversarial_step(self, cf, w_approx, b_approx, y_target, max_delta, n_steps):
        """
        Adversarially tweak w_approx -> w_approx + delta to degrade the local approximation's 
        performance. We'll do a quick projected gradient approach in NumPy on:
           adv_loss(delta) = (pred_fn(cf, w_approx + delta) - y_target)^2

        For simplicity, we only produce one final delta, ignoring repeated steps on delta
        unless you want to do a small loop.
        """
        # initialize delta in [-max_delta, max_delta]
        rng = np.random.default_rng(seed=42)
        delta_vec = rng.uniform(-max_delta, max_delta, size=w_approx.shape)

        def _adv_loss(delta):
            w_perturbed = w_approx + delta
            y_hat = self._sigmoid(cf @ w_perturbed + b_approx)
            return (y_hat - y_target)**2

        # Simple repeated sign gradient approach
        for _ in range(n_steps):
            grad = self._numeric_grad(_adv_loss, delta_vec)
            # move in direction that increases the loss
            alpha = 1.25 * max_delta
            delta_vec += alpha * np.sign(grad)
            # project to L_infinity ball
            delta_vec = np.clip(delta_vec, -max_delta, max_delta)

        return delta_vec

    def _gradient_update_cf(self, cf, x, w, b, y_target, lr, lambda_):
        """
        One gradient step on:
          loss(cf) = BCE( sigmoid(w^T cf + b), y_target ) + lambda_ * ||cf - x||^2.
        We'll do an analytic gradient for BCE + L2:

          If L(cf) = BCE + lambda * L2,
          BCE'(cf) = (y_hat - y_target)*w * (derivative of sigmoid part),
          L2'(cf) = 2*(cf - x).
        """
        # forward pass
        y_hat = self._sigmoid(cf @ w + b)

        grad_bce = (y_hat - y_target) * w

        # partial derivative wrt cf for the L2 part = 2 * lambda_ * (cf - x)
        grad_l2 = 2 * lambda_ * (cf - x)

        # total gradient
        grad_cf = grad_bce + grad_l2

        # gradient descent step
        cf_new = cf - lr * grad_cf
        return cf_new

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------

    def _predict_proba_single(self, x_np_2d):
        """
        Utility to get model's probability predictions for a single instance
        shaped (1, d). Returns np.array of shape (1,2): [p0, p1].
        """
        # This depends on your TrainablePyTorchModel implementation
        # We will assume .predict_proba(x) returns a DataFrame with columns [0,1].
        out_df = self.model.predict_proba(x_np_2d)
        return out_df.values  # shape (1,2)

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _numeric_grad(self, fun, x, eps=1e-5):
        """
        Approximate gradient of fun w.r.t. x using finite differences,
        for demonstration. x is a 1D np.array, fun(x) -> scalar.
        """
        grad = np.zeros_like(x)
        fx = fun(x)
        for i in range(len(x)):
            orig = x[i]
            x[i] = orig + eps
            fx_plus = fun(x)
            x[i] = orig - eps
            fx_minus = fun(x)
            x[i] = orig
            grad[i] = (fx_plus - fx_minus) / (2.0 * eps)
        return grad