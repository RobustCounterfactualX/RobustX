from robustx.lib.distance_functions.DistanceFunctions import euclidean
from robustx.generators.CEGenerator import CEGenerator
from robustx.robustness_evaluations.DeltaRobustnessEvaluator import DeltaRobustnessEvaluator


class GuidedBinaryLinearSearch(CEGenerator):

    def _generation_method(self, instance, gamma=0.1, column_name="target", neg_value=0,
                           distance_func=euclidean, **kwargs):

        # Get initial counterfactual
        c = self.task.get_random_positive_instance(neg_value, column_name).T

        opt = DeltaRobustnessEvaluator(self.task)

        # Keep getting random positive counterfactual until we find one that is robust
        while not opt.evaluate(instance, desired_output=1 - neg_value):
            c = self.task.get_random_positive_instance(neg_value, column_name).T

        # Make sure column names are same so return result has same indices
        negative = instance.to_frame()
        c.columns = negative.columns

        model = self.task.model

        # Loop until CE is under gamma threshold
        while distance_func(negative, c) > gamma:

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
        res = model.predict_single(ct)

        ct["target"] = res

        # Store the loss
        ct["loss"] = distance_func(negative, c)

        return ct
