import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import cross_val_score, KFold

def cross_validation(X, y):
    """
    This function runs cross validation on the input dataset. 
    The model used in this function is the NearestCentroid classifier.
    Delta values are defined as 10 evenly spaced values from 0.01 to 1.0.
    A 10-fold cross validation is performed on each delta value to determine the optimal delta value, delta star.
    The function returns the delta star value and the mean cross validation error for the delta star model.

    Returns
    -------
    delta_star: float
        The optimal delta value.
    cv_error_delta_star: float
        The mean cross validation error for the delta star model.
    """

    # 10 values of delta from 0.01 to 1.0
    delta_values = np.linspace(0.01, 1.0, 10)  # 10 evenly spaced values from 0.01 to 1.0

    # define nearest centroid model
    model = NearestCentroid(metric='euclidean')

    # Use 10-fold CV to determine the optimal delta hyperparameter
    kf = KFold(n_splits=10, shuffle=True, random_state=1) 

    # Initialize a list to store mean CV scores for each delta value
    cv_scores = []

    # loop through each delta value and perform cross validation
    for delta in delta_values:
        model.set_params(shrink_threshold=delta)
    
        scores = cross_val_score(
            model,  # nearest centroid model as defined above
            X,      # feature matrix
            y,      # labels - Class 1 and Class 2
            cv=kf,  # CV using the Kfold as defined above
            scoring='accuracy'
        )
    
        cv_scores.append(scores.mean())  # store mean CV score for each delta value

    idx_max_score = np.argmax(cv_scores)  # returns the index of the delta value that had the highest mean CV score
    delta_star = delta_values[idx_max_score]  # returns the delta value that had the highest mean CV score
    cv_error_delta_star = 1 - cv_scores[idx_max_score]  # returns the mean CV error for the optimal delta
    
    return delta_star, cv_error_delta_star