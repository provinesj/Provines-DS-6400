from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score
from source_code.cross_validation import cross_validation

def nested_cross_validation(X, y):
    """
    This function takes the null data and performs nested cross-validation.
    The inner loop is 10-fold CV and is used to select the optimal delta star value.
    The outer loop is LOOCV and is used to evaluate the delta star model to compute the CV error.

    Parameters
    ----------
    X : ndarray
        Feature matrix.
    y : ndarray
        Class labels.

    Returns
    -------
    nested_cv_error : float
        nested CV error of the delta star model
    """

    # Create outer leave-one-out cross-validation
    loo = LeaveOneOut()

    # Create empty list to store the prediction results from outer loop
    outer_errors = []

    # Perform outer loop LOOCV
    for train_index, test_index in loo.split(X):

        # Split data into outer training and test sets
        X_train = X[train_index]
        X_test = X[test_index]

        y_train = y[train_index]
        y_test = y[test_index]

        # Inner loop 10-fold CV to select delta star
        # Don't need error values from cross_validation this time since CV error will be calculated in outer loop
        delta_star, _ = cross_validation(X_train, y_train)

        # Create model using delta star
        model = NearestCentroid(metric="euclidean", shrink_threshold=delta_star)

        # Fit model using all 39 outer training samples
        model.fit(X_train, y_train)

        # Predict the held-out outer test sample
        y_pred = model.predict(X_test)

        # Calculate classification error for this outer loop
        error = 1 - accuracy_score(y_test, y_pred)

        # Store outer loop error
        outer_errors.append(error)

    # Calculate mean error across all outer loops
    nested_cv_error = sum(outer_errors) / len(outer_errors)

    return nested_cv_error