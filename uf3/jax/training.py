import numpy as onp

import jax.numpy as jnp
from jax import vmap, grad, jit
from jax import lax

import optax

from functools import partial

from uf3.regression.regularize import get_regularizer_matrix, get_penalty_matrix_3D


def regularizer(coefficients, ridge=0.0, curvature=1.0):
    if len(coefficients.shape) == 1:
        return jnp.sum(
            jnp.einsum(
                "ij,j->i",
                get_regularizer_matrix(len(coefficients), ridge=ridge, curvature=curvature),
                coefficients,
            )
            ** 2
        )
    if len(coefficients.shape) == 3:
        return jnp.sum(
            jnp.einsum(
                "ij,j->i",
                get_penalty_matrix_3D(*coefficients.shape, ridge=ridge, curvature=curvature),
                coefficients.flatten(),
            )
            ** 2
        )
