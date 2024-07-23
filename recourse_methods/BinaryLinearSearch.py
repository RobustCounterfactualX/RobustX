from recourse_methods.RecourseGenerator import RecourseGenerator
from lib.distance_functions.DistanceFunctions import l1, euclidean


class BinaryLinearSearch(RecourseGenerator):

    def generate_for_instance(self, instance, distance_func="euclidean", custom_func=None, gamma=0.1,
                              column_name="target", neg_value=0):

        # Get initial counterfactual
        c = self.ct.get_random_positive_instance(neg_value, column_name).T

        # Make sure column names are same so return result has same indices
        negative = instance.to_frame()
        c.columns = negative.columns

        # Decide which distance function to use
        if distance_func == "l1" or distance_func == "manhattan":
            dist = l1
        else:
            dist = euclidean

        model = self.ct.model

        # Loop until CE is under gamma threshold
        while dist(negative, c) > gamma:

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
        ct["loss"] = dist(negative, c)

        return ct
