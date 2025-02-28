from rocelib.lib.distance_functions.DistanceFunctions import euclidean
from rocelib.recourse_methods.RecourseGenerator import RecourseGenerator
from rocelib.evaluations.robustness_evaluations.MC_Robustness_Implementations.DeltaRobustnessEvaluator import DeltaRobustnessEvaluator


class GuidedBinaryLinearSearch(RecourseGenerator):
    """
    A recourse method that performs a guided binary search between an original
    instance and a positively classified instance, attempting to meet a user-defined
    robustness threshold.
    """

    def _generation_method(self, instance, gamma=0.1, column_name="target", neg_value=0,
                           distance_func=euclidean, **kwargs):
        """
        Generate a counterfactual by repeatedly searching for a robust positive instance
        and then performing binary search until the Euclidean distance to the original
        instance is below `gamma`.

        Parameters
        ----------
        instance : pd.Series
            The original instance for which to generate a recourse.
        gamma : float, optional
            The distance threshold at which to stop the binary search, by default 0.1.
        column_name : str, optional
            The name of the target column, by default "target".
        neg_value : int, optional
            Controls whether a 'negative' instance is predicted by the model, by default 0.
        distance_func : callable, optional
            The distance function used to measure how close two instances are,
            by default `euclidean`.
        kwargs : dict
            Additional arguments to pass to the method (not used here).

        Returns
        -------
        pd.DataFrame
            A single-row DataFrame containing the counterfactual instance,
            along with the 'target' column indicating model prediction
            and a 'loss' column representing the final distance.
        """

        # Obtain a random positive instance and evaluate its robustness
        c = self.task.get_random_positive_instance(neg_value, column_name).T
        opt = DeltaRobustnessEvaluator(self.task)
        MAX_ITERATIONS = 10

        iteration = 0
        # Keep searching for a robust positive instance if not robust
        while not opt.evaluate_single_instance(instance):
            c = self.task.get_random_positive_instance(neg_value, column_name).T
            iteration += 1

            # Stop if too many attempts
            if iteration > MAX_ITERATIONS:
                break

        # Align columns with the original instance
        negative = instance.to_frame()
        c.columns = negative.columns

        model = self.task.model

        # Perform binary search until instance distance is below gamma
        while distance_func(negative, c) > gamma:
            # Midpoint between the negative instance and current candidate
            new_neg = c.add(negative, axis=0) / 2

            # Update either the negative or candidate based on model's prediction
            if model.predict_single(new_neg.T) == model.predict_single(negative.T):
                negative = new_neg
            else:
                c = new_neg

        # Format the resulting counterfactual
        ct = c.T
        res = model.predict_single(ct)
        ct["target"] = res
        ct["loss"] = distance_func(negative, c)

        return ct