import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

from rocelib.evaluations.RecourseEvaluator import RecourseEvaluator


### Work In Progress ###
class ManifoldEvaluator(RecourseEvaluator):
    """
     An Evaluator class which evaluates the proportion of counterfactuals which are on the data manifold using LOF

        ...

    Attributes / Properties
    -------

    task: Task
        Stores the Task for which we are evaluating the robustness of CEs

    -------

    Methods
    -------

    evaluate() -> int:
        Returns the proportion of CEs which are robust for the given parameters

    -------
    """

    def evaluate(self, recourse_method, n_neighbors=20, column_name="target", **kwargs):
        """
        Determines the proportion of CEs that lie on the data manifold based on LOF
        @param counterfactual_explanations: DataFrame, containing the CEs in the same order as the negative instances in the dataset
        @param n_neighbors: int, number of neighbours to compare to in order to find if outlier
        @param column_name: str, name of target column
        @param kwargs: other arguments
        @return: proportion of CEs on manifold
        """
        counterfactual_explanations = self.task._CEs[recourse_method][0]
        on_manifold = 0
        cnt = 0

        data = self.task.dataset.X
        counterfactual_explanations = counterfactual_explanations.drop(columns=[column_name, "loss", "predicted"], errors='ignore')

        # TODO: Compute raw LoF score, proplace code to see what else is working
        for _, ce in counterfactual_explanations.iterrows():

            if ce is None:
                cnt += 1
                continue

            # Combine the dataset with the new instance
            data_with_instance = pd.concat([data, pd.DataFrame([ce])], ignore_index=True)

            data_with_instance.columns = data_with_instance.columns.astype(str)

            # Drop NaN-containing "Loss" column before using LOF
            data_with_instance = data_with_instance.drop(columns=["Loss"], errors="ignore")

            if "HiringDecision" in data_with_instance.columns:
                data_with_instance = data_with_instance.drop(columns=["HiringDecision"], errors="ignore")

            # Apply Local Outlier Factor (LOF)
            lof = LocalOutlierFactor(n_neighbors=n_neighbors)
            lof_scores = lof.fit_predict(data_with_instance)  # Predict if data points are outliers (-1) or not (1)

            # The last point in the combined dataset is the instance we are checking
            instance_lof_score = lof.negative_outlier_factor_[-1]

            # Return True if the LOF score is below the threshold (i.e., not an outlier)
            if instance_lof_score > 0.9:
                on_manifold += 1

            cnt += 1

        return on_manifold / cnt