from rocelib.evaluations.robustness_evaluations.ModelMultiplicityRobustnessEvaluator import ModelMultiplicityRobustnessEvaluator


class MultiplicityValidityRobustnessEvaluator(ModelMultiplicityRobustnessEvaluator):
    """
    The robustness evaluator that examines how many models (in %) each counterfactual is valid on.
    """

    def evaluate_single_instance(self, instance, counterfactuals, **kwargs):
        """
        Evaluate on average how many models (in %) each counterfactual is valid on.

        @param index: An index for the input instance.
        """
        # instance = self.task.dataset.data.iloc[index]
        # instance = list(self.task.dataset.get_negative_instances())[index]
        # instance = instance.drop('target')

        # mm_CEs: Dict[str, Dict[str, Tuple[pd.DataFrame, float]]]
        # instance = instance.drop('target')

        avg_valid_num = 0
        for c in counterfactuals:
            # Drop unwanted columns from each counterfactual before evaluation
            c = c.drop(columns=["predicted", "Loss"], errors="ignore")
            avg_valid_num += self.evaluate_single_counterfactual(instance, c)
        return (avg_valid_num / len(counterfactuals)) == 1

    def evaluate_single_counterfactual(self, instance, counterfactual):
        """
        Evaluate how many models (in %) one counterfactual is valid on.

        @param instance: An input instance.
        @param counterfactual: A CE.
        """
        num_models = len(self.task.mm_models)
        num_valid = 0
        counterfactual = counterfactual.drop(labels=["predicted", "Loss"], errors='ignore')

        for m in self.task.mm_models:
            model = self.task.mm_models[m]
            if model.predict_single(instance) != model.predict_single(counterfactual):
                num_valid += 1
        return num_valid / num_models