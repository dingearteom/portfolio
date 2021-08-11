import numpy as np

def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    # TODO: Implement computing accuracy
    assert isinstance(prediction, np.ndarray), "prediction must be of type numpy array"
    assert isinstance(ground_truth, np.ndarray), "ground_truth must be of type numpy array"

    num_samples = prediction.shape[0]
    accuracy = np.sum(prediction == ground_truth) / num_samples
    return accuracy
