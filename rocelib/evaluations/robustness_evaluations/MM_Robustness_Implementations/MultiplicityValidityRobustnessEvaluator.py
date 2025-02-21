from rocelib.evaluations.robustness_evaluations.ModelMultiplicityRobustnessEvaluator import ModelMultiplicityRobustnessEvaluator


class MultiplicityValidityRobustnessEvaluator(ModelMultiplicityRobustnessEvaluator):
    """
    The robustness evaluator that examines how many models (in %) each counterfactual is valid on.
    """

    def evaluate_single_instance(self, index, recourse_method, **kwargs):
        # def evaluate_single_instance(self, index, counterfactuals):
        """
        Evaluate on average how many models (in %) each counterfactual is valid on.

        @param index: An index for the input instance.
        @param recourse_method: the recourse method to perform evaluation for.
        """
        instance = self.task.dataset.data.iloc[index]
        instance = instance.drop('target')

        # mm_CEs: Dict[str, Dict[str, Tuple[pd.DataFrame, float]]]

        # Get the counterfactual for each model for this instance and put all into a list
        counterfactuals = []
        ces = self.task.mm_CEs[recourse_method]
        for model_name in ces:
            counterfactual = ces[model_name][0].iloc[index].drop('predicted').drop('Loss')
            counterfactuals.append(counterfactual)

        avg_valid_num = 0
        for c in counterfactuals:
            avg_valid_num += self.evaluate_single_counterfactual(instance, c, recourse_method)
        return avg_valid_num / len(counterfactuals)

    def evaluate_single_counterfactual(self, instance, counterfactual, recourse_method):
        """
        Evaluate how many models (in %) one counterfactual is valid on.

        @param instance: An input instance.
        @param counterfactual: A CE.
        """
        num_models = len(self.task.mm_models)
        num_valid = 0
        for m in self.task.mm_models:
            model = self.task.mm_models[m]
            if model.predict_single(instance) != model.predict_single(counterfactual):
                num_valid += 1
        return num_valid / num_models