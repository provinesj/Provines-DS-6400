import numpy as np

def generate_noise_data(n_samples=40, n_features=100):
    """
    This function creates a Gaussian-noise dataset with a 50/50 split of Class 1 and Class 2. 
    The features are generated from a normal distribution.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : ndarray, shape (n_samples,)
        Balanced labels: Class 1 and Class 2.
    """
    # Prevents odd number of samples to ensure a 50/50 class split.
    if n_samples % 2 != 0:
        raise ValueError("n_samples must be even for an exact 50/50 class split.")

    # Generate noise dataset with mean=0 and std=1 for features, and balanced classes.
    X = np.random.normal(loc=0, scale=1, size=(n_samples, n_features))
    y = np.array([1] * (n_samples // 2) + [2] * (n_samples // 2))

    # Randomize row order so classes are not grouped together.
    indices = np.random.permutation(n_samples)

    return X[indices], y[indices]