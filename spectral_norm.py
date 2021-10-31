# Copyright 2020 Google LLC.
# SPDX-License-Identifier: Apache-2.0
# https://nbviewer.org/gist/shoyer/fa9a29fd0880e2e033d7696585978bfc

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
import scipy.linalg


# new method

def _l2_normalize(x, eps=1e-4):
    return x * jax.lax.rsqrt((x ** 2).sum() + eps)


def estimate_spectral_norm(f, input_shape, seed=0, n_steps=10):
    rng = jax.random.PRNGKey(seed)
    u0 = jax.random.normal(rng, input_shape)
    v0 = jnp.zeros_like(f(u0))

    def fun(carry, _):
        u, v = carry
        v, f_vjp = jax.vjp(f, u)
        v = _l2_normalize(v)
        u, = f_vjp(v)
        u = _l2_normalize(u)
        return (u, v), None

    (u, v), _ = lax.scan(fun, (u0, v0), xs=None, length=n_steps)
    return jnp.vdot(v, f(u))


# excat calculation

def exact_spectral_norm(f, input_shape):
    dummy_input = jnp.zeros(input_shape)
    jacobian = jax.jacfwd(f)(dummy_input)
    shape = (np.prod(jacobian.shape[:-dummy_input.ndim]), np.prod(input_shape))
    return scipy.linalg.svdvals(jacobian.reshape(shape)).max()
