from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score
from source_code.generate_noise_data import generate_noise_data

def model_evaluation(X, y, delta_star):

    """
    This evaluates the delta star model on a new, larger null dataset. 
    The delta star model is created using the optimal delta hyperparameter, delta star.
    It is then trained on the original null dataset and then evaluated on the new, larger null dataset. 
    The true error of the delta star model is determined by comparing the predicted class labels to the true class labels of the larger null dataset.

    Returns
    -------
    true_error_delta_star: float
        The true error of the delta star model on the new, larger null dataset.
    """
    # create delta star model with delta star hyperparameter
    model_delta_star = NearestCentroid(metric='euclidean', shrink_threshold=delta_star)

    # fit model on original dataset
    model_delta_star.fit(X, y)

    # generate new, null dataset with 1000 samples
    X_eval, y_eval = generate_noise_data(n_samples=1000, n_features=100)

    # use model to predict class on the new, larger null dataset
    y_pred = model_delta_star.predict(X_eval)

    # determine true error of the delta star model
    accuracy = accuracy_score(y_eval, y_pred)
    true_error_delta_star = 1 - accuracy

    return true_error_delta_star