import jax.image
from typing import Any
import tensorflow_datasets as tfds
from pathlib import Path
from datasets.confounded_mnist import confoudned_mnist
from jax.lib import xla_bridge
import tensorflow as tf
from staxplus import train
from jax import value_and_grad
from typing import Tuple, Dict
import optax
from staxplus.real_nvp import glow
from staxplus.types import is_shape
from staxplus import Model, Params, KeyArray, Array, ArrayTree, GradientTransformation, OptState, ShapeTree
import jax.numpy as jnp
from models.utils import concat_parents
print(xla_bridge.get_backend().platform)
tf.config.experimental.set_visible_devices([], 'GPU')

def nvp_model() -> Model:
    shape = (-1, 32, 32, 3)
    nvp_init_fn, log_prob, sample = glow(input_shape=shape, depth_per_level=3)

    def init_fn(rng: KeyArray, input_shape: ShapeTree) -> Params:
        assert is_shape(input_shape)
        _, params = nvp_init_fn(rng, shape)
        return params

    def apply_fn(params: Params, rng: KeyArray, inputs: ArrayTree) -> Tuple[Array, Dict[str, Array]]:
        # assert isinstance(inputs, Array)
        image, parents = inputs
        batch_size = image.shape[0]
        image = jax.image.resize(image, shape=(batch_size, 32, 32, 3),  method='linear')
        _parents = jnp.expand_dims(concat_parents(parents), axis=(1, 2))
        _parents = jnp.broadcast_to(_parents, shape=(*image.shape[:-1], _parents.shape[-1]))
        _inputs = jnp.concatenate((image, _parents), axis=-1)
        samples = jnp.clip(sample(params, rng, batch_size), a_min=-1, a_max=1)
        _log_prob = log_prob(params, image)
        return -jnp.mean(_log_prob), {'samples': samples, 'log_prob': _log_prob}

    def update_fn(params: Params,
                  optimizer: GradientTransformation,
                  opt_state: OptState,
                  rng: KeyArray,
                  inputs: ArrayTree) -> Tuple[Params, OptState, Array, Dict[str, Array]]:
        (loss, outputs), grads = value_and_grad(apply_fn, has_aux=True)(params, rng, inputs)
        updates, opt_state = optimizer.update(updates=grads, state=opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, outputs

    return Model(init_fn, apply_fn, update_fn)

# def nvp_model() -> Model:
#     shape = (-1, 32, 32, 3)
#     nvp_init_fn, log_prob, sample = glow(input_shape=shape, depth_per_level=3)

#     def init_fn(rng: KeyArray, input_shape: ShapeTree) -> Params:
#         assert is_shape(input_shape)
#         _, params = nvp_init_fn(rng, shape)
#         return params

#     def apply_fn(params: Params, rng: KeyArray, inputs: ArrayTree) -> Tuple[Array, Dict[str, Array]]:
#         # assert isinstance(inputs, Array)
#         image, parents = inputs
#         batch_size = image.shape[0]
#         image = jax.image.resize(image, shape=(batch_size, 32, 32, 3),  method='linear')
#         samples = jnp.clip(sample(params, rng, batch_size), a_min=-1, a_max=1)
#         _log_prob = log_prob(params, image)
#         return -jnp.mean(_log_prob), {'samples': samples, 'log_prob': _log_prob}

#     def update_fn(params: Params,
#                   optimizer: GradientTransformation,
#                   opt_state: OptState,
#                   rng: KeyArray,
#                   inputs: ArrayTree) -> Tuple[Params, OptState, Array, Dict[str, Array]]:
#         (loss, outputs), grads = value_and_grad(apply_fn, has_aux=True)(params, rng, inputs)
#         updates, opt_state = optimizer.update(updates=grads, state=opt_state, params=params)
#         params = optax.apply_updates(params, updates)
#         return params, opt_state, loss, outputs

#     return Model(init_fn, apply_fn, update_fn)


data_dir = Path('data')
job_dir = Path('/tmp/test_nvp')
dataset_name, scenario = confoudned_mnist('digit_hue', data_dir, confound=False, scale=.1, outlier_prob=0.)
train_datasets, test_dataset, parent_dists, input_shape, _ = scenario


def to_numpy_iterator(data: tf.data.Dataset, batch_size: int, drop_remainder: bool = True) -> Any:
    return tfds.as_numpy(data.batch(batch_size,
                                    drop_remainder=drop_remainder,
                                    num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE))


batch_size = 512
train_data = to_numpy_iterator(train_datasets[frozenset()], batch_size=batch_size)
test_data = to_numpy_iterator(test_dataset, batch_size=batch_size, drop_remainder=True)
train(model=nvp_model(),
      job_dir=job_dir,
      seed=10,
      train_data=train_data,
      test_data=test_data,
      input_shape=input_shape,
      optimizer=optax.adamw(learning_rate=1e-5),
      num_steps=1000000,
      log_every=10,
      eval_every=500,
      save_every=1000,
      overwrite=True,
      use_jit=True)
