from source_code.generate_noise_data import generate_noise_data
from source_code.nested_cross_validation import nested_cross_validation
from source_code.cross_validation import cross_validation
from source_code.model_evaluation import model_evaluation

def run_nested_simulation():

    # Generate training data
    X, y = generate_noise_data(n_samples=40, n_features=100)

    # Run nested CV
    nested_cv_error = nested_cross_validation(X, y)

    # Run CV to get the delta star value
    delta_star, cv_error_delta_star = cross_validation(X, y)

    # Run model_evaluation function to evaluate delta star model on larger dataset
    true_error_delta_star = model_evaluation(X, y, delta_star)

    # Return the nested CV error and true error for the delta star model.
    return {
        "nested_cv_error": nested_cv_error,
        "true_error_delta_star": true_error_delta_star
    }