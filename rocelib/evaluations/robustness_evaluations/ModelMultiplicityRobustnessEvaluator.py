from rocelib.evaluations.robustness_evaluations.BaseRobustnessEvaluator import BaseRobustnessEvaluator

class ModelMultiplicityRobustnessEvaluator(BaseRobustnessEvaluator):
    """
    Abstract base class for evaluating the robustness of model predictions with respect to Model multiplicity and acts 
    as a holder for concrete implementations

    """


    def evaluate(self, recourse_method, **kwargs):
        """
        We override BaseRobustnessEvaluator as we need a list of counterfactuals rather than a single counterfactual
        Returns: a list of evaluation scores
        """
        evaluations = []

        for index, (_,instance) in enumerate(self.task.dataset.get_negative_instances().iterrows()):
            # Get the counterfactual for each model for this instance and put all into a list
            counterfactuals = []
            ces = self.task.mm_CEs[recourse_method]
            for model_name in ces:
                counterfactual = ces[model_name][0].iloc[index]
                counterfactuals.append(counterfactual)

            evaluations.append(self.evaluate_single_instance(instance, counterfactuals, **kwargs))

        return evaluations