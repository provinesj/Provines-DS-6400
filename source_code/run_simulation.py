from source_code.generate_noise_data import generate_noise_data
from source_code.cross_validation import cross_validation
from source_code.model_evaluation import model_evaluation

def run_simulation():

    # Generate training data
    X, y = generate_noise_data(n_samples=40, n_features=100)

    # Find delta star
    delta_star, cv_error_delta_star = cross_validation(X, y)

    # Evaluate delta star model
    true_error_delta_star = model_evaluation(X, y, delta_star)

    # return dictionary of CV error and true error values for the delta star model
    return {
        "cv_error_delta_star": cv_error_delta_star,
        "true_error_delta_star": true_error_delta_star
    }