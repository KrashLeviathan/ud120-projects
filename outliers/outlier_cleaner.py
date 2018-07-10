#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []

    ### your code goes here
    # cleaned_data.push([age, net_worth, error])
    for i in range(len(predictions)):
        error = net_worths[i] - predictions[i]
        cleaned_data.append([ages[i], net_worths[i], error])

    cleaned_data.sort(key=lambda x: abs(x[2]))

    end_index = len(cleaned_data) * 9 / 10
    cleaned_data = cleaned_data[0:end_index]

    return cleaned_data

