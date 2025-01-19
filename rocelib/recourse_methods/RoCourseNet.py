from rocelib.recourse_methods.RecourseGenerator import RecourseGenerator


class RoCourseNet(RecourseGenerator):
    """
    A recourse generator that uses two stage approach to generating CEs. First, we discuss the attacker’s problem: 
    (i) we propose a novel bi-level attacker problem to find the worstcase data shift that leads to an adversarially shifted ML model; 
    and (ii) we propose a novel Virtual Data Shift (VDS) algorithm for solving this bi-level attacker problem. Second, we discuss the
    defender’s problem: (i) we derive a novel tri-level learning problem based on the attacker’s bi-level problem; and (ii) we propose the 
    RoCourseNet training framework for optimizing this tri-level optimization problem, which leads to the simultaneous generation of
    accurate predictions and robust recourses.

    Attributes:
        _task (Task): The task to solve, inherited from RecourseGenerator.
        __customFunc (callable, optional): A custom distance function, inherited from RecourseGenerator.
    """

    def _generation_method(self, instance, gamma=0.1, column_name="target", neg_value=0,
                           distance_func=euclidean, **kwargs) -> pd.DataFrame:
        """
        Generates a nearest-neighbor counterfactual explanation for a provided instance.

        @param instance: The instance for which to generate a counterfactual. Can be a DataFrame or Series.
        @param gamma: The threshold for the distance between the instance and the counterfactual. (Not used in this method)
        @param column_name: The name of the target column. (Not used in this method)
        @param neg_value: The value considered negative in the target variable.
        @param distance_func: The function used to calculate the distance between two points. Defaults to euclidean.
        @param kwargs: Additional keyword arguments.
        @return: A DataFrame containing the nearest-neighbor counterfactual explanation for the provided instance.
        """
        # TODO
        return None
