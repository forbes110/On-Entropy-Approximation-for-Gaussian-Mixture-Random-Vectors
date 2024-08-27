import numpy as np
import math
from jax import grad, hessian, jacfwd, jit
import jax.numpy as jnp

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
