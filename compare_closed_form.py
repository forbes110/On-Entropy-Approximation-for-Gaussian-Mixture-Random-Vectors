import numpy as np
import jax.numpy as jnp

def multivariate_gaussian_entropy(covariance_matrix): 
    """
    Calculate the entropy of a multivariate Gaussian distribution.

    Parameters:
    covariance_matrix (numpy.ndarray): The covariance matrix of the distribution.

    Returns:
    float: The entropy of the multivariate Gaussian distribution.
    """
    # Ensure the covariance matrix is a square matrix
    if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
        raise ValueError("Covariance matrix must be square.")

    # Calculate the determinant of the covariance matrix
    det_cov = np.linalg.det(covariance_matrix)

    # Get the dimensionality (N)
    N = covariance_matrix.shape[0]

    # Calculate the entropy
    entropy = 0.5 * np.log((2 * np.pi * np.e) ** N * det_cov)
    # entropy = 0.5 * np.log((2 * np.pi * np.e) * sigma ** 2)

    return entropy

# Example usage
cov_matrix = jnp.eye(3)  # Example covariance matrix
entropy = multivariate_gaussian_entropy(cov_matrix)
print("Entropy of the multivariate Gaussian distribution:", entropy)