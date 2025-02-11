from robustx.robustness_evaluations.ModelMultiplicityRobustnessEvaluator import ModelMultiplicityRobustnessEvaluator


class MultiplicityValidityRobustnessEvaluator(ModelMultiplicityRobustnessEvaluator):
    """
    The robustness evaluator that examines how many models (in %) each counterfactual is valid on.
    """

    def evaluate(self, instance, counterfactuals):
        """
        Evaluate onn average how many models (in %) each counterfactual is valid on.

        @param instance: An input instance.
        @param counterfactuals: A series of CEs.
        """
        avg_valid_num = 0
        for c in counterfactuals:
            avg_valid_num += self.evaluate_single(instance, c)
        return avg_valid_num / len(counterfactuals)

    def evaluate_single(self, instance, counterfactual):
        """
        Evaluate how many models (in %) one counterfactual is valid on.

        @param instance: An input instance.
        @param counterfactual: A CE.
        """
        num_models = len(self.models)
        num_valid = 0
        for m in self.models:
            if m.predict_single(instance) != m.predict_single(counterfactual):
                num_valid += 1
        return num_valid / num_models
