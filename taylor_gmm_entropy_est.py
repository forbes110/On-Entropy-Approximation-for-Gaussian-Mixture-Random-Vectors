import numpy as np
import math
from time import time
import jax.numpy as jnp
from jax import grad, vmap, hessian, jacfwd, jacrev
from jax.scipy.stats import multivariate_normal
from jax import random
from scipy.integrate import nquad

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

def vec_in_scalar_out_taylor_expansion(x, x_0, func, params, R=4):
    """
    Compute the Taylor series expansion of order n around x_0.

    x: jax.numpy.ndarray
        The point at which to evaluate the expansion.
    x_0: jax.numpy.ndarray
        The point around which the expansion is made.
    params: tuple
        The GMM parameters (weights, means, covariances).
    R: int
        The order of the Taylor series expansion.
    """

    # Precompute difference vector and its powers
    delta_x = x - x_0

    # Start with the 0th order term (function value at x_0)

    if params != None:
        expansion = func(x_0, params)

    # Compute the 1st order (gradient) term
    if R >= 1:
        grad_vec = grad(func, argnums=0)(x_0, params)
        expansion += jnp.dot(grad_vec, delta_x)

    # Compute the 2nd order (Hessian) term
    if R >= 2:
        hessian_mat = hessian(func, argnums=0)(x_0, params)
        expansion += 0.5 * jnp.dot(delta_x, jnp.dot(hessian_mat, delta_x))

    # Compute higher-order terms
    if R > 2:
        for order in range(3, R + 1):
            # Compute the n-th order derivative tensor
            derivative_tensor = func
            for _ in range(order):
                derivative_tensor = jacfwd(derivative_tensor, argnums=0)
            
            # Tensor contraction with the derivative tensor and delta_x
            term = derivative_tensor(x_0, params)
            for _ in range(order):
                term = jnp.tensordot(term, delta_x, axes=1)
            
            # Add the term to the expansion
            expansion += (1 / math.factorial(order)) * term

    return expansion



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

    print(H_approx)