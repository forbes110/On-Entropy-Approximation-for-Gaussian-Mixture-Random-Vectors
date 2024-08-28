import numpy as np
from time import time
import jax.numpy as jnp
from jax import vmap
from jax.scipy.stats import multivariate_normal
from jax import random
from scipy.integrate import nquad
from vec_taylor_expansion import vec_in_scalar_out_taylor_expansion
from entropy_bounds import EntropyLowerBoundEst, EntropyUpperBoundEst

def EntropyEst(gmm_params, R, num_samples, random_seed):
    """
    Compute the entropy approximation for a GMM.

    gmm_params: tuple
        Tuple containing (weights, means, covariances) of the GMM.
    x: numpy array
        The variable for which to calculate the entropy.
    R: int
        The order at which to truncate the Taylor series expansion.
    num_samples: int
        The number of samples to use for Monte Carlo integration.
    """

    weights, means, covariances = gmm_params
    L = len(weights)

    if R > L:
        assert "R should not be larger than L"

    H_approx = 0

    for i in range(0, L):

        expansion = 0
        mu_i = means[i]
        cov_i = covariances[i]
        integral = monte_carlo_integration(mu_i, cov_i, gmm_params, num_samples, R, random_seed)
        H_approx -= weights[i] * integral

    return H_approx.item()


def monte_carlo_integration(mean, covariance, gmm_params, num_samples=100000, R = 3, seed=None):

    if seed is None:
        seed = int(time()) % 10000

    weights, means, covariances = gmm_params
    L = len(weights)

    key = random.PRNGKey(seed)
    dim = mean.shape[0]
    minval, maxval = -5, 5

    # Generate samples uniformly within the integration region
    samples = random.uniform(key, shape=(num_samples, dim), minval=minval, maxval=maxval)

    # Evaluate the multivariate Gaussian PDF at the sample points
    pdf_values = multivariate_normal.pdf(samples, mean, covariance)

    # Calculate the Taylor expansion for each sample
    taylor_expansion_fn = vmap(vec_in_scalar_out_taylor_expansion, in_axes=(0, None, None, None, None))

    taylor_expansion_values = taylor_expansion_fn(samples, mean, log_g_function, gmm_params, R)

    # Compute the volume of the integration region
    volume = (maxval - minval) ** dim

    # Combine the PDF values with the Taylor expansion
    # This represents the integrand at each sampled point, by inner product
    integrand_values = pdf_values * taylor_expansion_values

    # Estimate the integral by taking the mean of the integrand values
    integral_estimate = volume * jnp.mean(integrand_values)

    return integral_estimate


def scipy_integration(mean, covariance, gmm_params):

    dim = mean.shape[0]
    minval, maxval = -5, 5

    # Define the integrand function to be integrated
    def integrand(*x):
        x = np.array(x)
        pdf_value = multivariate_normal.pdf(x, mean, covariance)
        taylor_expansion_value = vec_in_scalar_out_taylor_expansion(x, mean, gmm_params)
        return pdf_value * taylor_expansion_value

    # Define the integration limits for each dimension
    integration_limits = [[minval, maxval]] * dim

    # Perform the integration using nquad
    result, error = nquad(integrand, integration_limits)

    return result #, error

def log_g_function(x, gmm_params):

    weights, means, covariances = gmm_params

    L = len(weights)

    g = 0
    for i in range(0, L):
        g += weights[i] * multivariate_normal.pdf(x, mean=means[i], cov=covariances[i])

    return jnp.log(g)

if __name__ == '__main__':

    # ------------------------------------------------------------------------------
    # Example usage: L = 2 for dim 4
    # weights = jnp.array([0.1 for _ in range(10)])
    weights = jnp.array([0.125 for _ in range(8)])
    means = [jnp.array([0.0, 0.0, 0.0]) for _ in range(8)]
    # means = [np.random.rand(3) for _ in range(5)]
    covariances = [jnp.eye(3) for _ in range(8)]
    gmm_params = (weights, means, covariances)
    H_approx = EntropyEst(gmm_params, R = 2, num_samples = 1000000, random_seed=42)

    H_l = EntropyLowerBoundEst(gmm_params)
    H_u = EntropyUpperBoundEst(weights, covariances)
    print("Lower Bound:", H_l)
    print("Approx:", H_approx)
    print("Upper Bound:", H_u)
