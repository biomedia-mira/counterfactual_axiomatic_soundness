import itertools
from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from jax import grad, jacfwd, jacrev, jit, random, vmap, partial
from jax.experimental import optimizers, stax
from jax.experimental.stax import Dense, Sigmoid, elementwise, Relu


# def rot(angle=0):
#     rad = np.deg2rad(angle)
#     return torch.stack([
#         torch.tensor([np.cos(rad), -np.sin(rad)]),
#         torch.tensor([np.sin(rad), np.cos(rad)])
#     ]).to(torch.float32)

def hessian(fun):
    return jit(jacfwd(jacrev(fun)))


def data_generating_process(num_samples: int, noise: float = .01, variance=0.01) -> Tuple[jnp.ndarray, ...]:
    shape = [num_samples]
    key0 = random.PRNGKey(0)
    _, key1 = random.split(key0)
    _, key2 = random.split(key1)
    _, key3 = random.split(key2)
    c = jax.random.bernoulli(key0, shape=shape).astype(float)
    c_clamped = jax.lax.clamp(noise, c, 1 - noise)
    x1 = jax.random.bernoulli(key1, p=c_clamped)
    x2 = jax.random.bernoulli(key2, p=c_clamped)
    x3 = jax.random.multivariate_normal(key3, mean=jnp.stack((x1, x2), axis=-1).astype(float),
                                        cov=jnp.eye(2) * variance)
    return c, x1, x2, x3


def plot(x1: jnp.ndarray,
         x2: jnp.ndarray,
         x3: jnp.ndarray,
         model: Callable[[List[jnp.ndarray], jnp.ndarray], jnp.ndarray],
         params: List[jnp.ndarray],
         num_samples_per_dim: int = 100):
    linspace = jnp.linspace(-1, 2, num_samples_per_dim)
    x, y = jnp.meshgrid(linspace, linspace)
    test_data = jnp.stack((x, y), axis=-1)
    probs = model(params, test_data)

    plt.figure(figsize=(7, 7))
    plt.contourf(test_data[..., 0], test_data[..., 1], probs, alpha=0.5)
    sns.scatterplot(x3[..., 0], x3[..., 1], hue=x1, style=x2, alpha=0.5)
    plt.axis('equal')
    plt.show()


num_samples = 1000
step_size = .1
num_epochs = 100
momentum_mass = 0.9

c, x1, x2, x3 = data_generating_process(num_samples=num_samples)
# sns.kdeplot(x3[:, 0], x3[:, 1])
# plt.show()

init_random_params, model = stax.serial(Dense(2), Relu, Dense(1), Sigmoid, elementwise(jnp.squeeze))
opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)
# rng = random.PRNGKey(0)


_, init_params = init_random_params(random.PRNGKey(0), (-1, 2))
pred = model(init_params, x3)
opt_state = opt_init(init_params)
itercount = itertools.count()


def accuracy(probs, targets):
    return jnp.mean(jnp.round(probs).astype(bool) == targets)


def loss(params: List[jnp.ndarray],
         data: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
    inputs, targets = data
    probs = model(params, inputs)
    ce = -jnp.mean(jnp.log(probs) * targets + (1 - targets) * jnp.log(1 - probs))
    # hessians = vmap(hessian(partial(model, params)), in_axes=0)(inputs)
    # mean_hessian = jnp.mean(hessians, axis=0)
    # diagonal = jnp.diag(mean_hessian)
    # off_diagonal = mean_hessian - diagonal
    return ce


@jit
def update(i, opt_state, data):
    params = get_params(opt_state)
    return opt_update(i, grad(loss)(params, data), opt_state)


for epoch in range(num_epochs):
    opt_state = update(next(itercount), opt_state, (x3, x1))
    params = get_params(opt_state)
    print(epoch, loss(params, (x3, x1)), accuracy(model(params, x3), x1))

plot(x1, x2, x3, model, params, num_samples_per_dim=1000)

