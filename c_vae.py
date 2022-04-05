import itertools
##
import os
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from typing import Iterable
from typing import Tuple, Union

import jax
import jax.nn as nn
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from jax import jit, random, value_and_grad, vmap
from jax.example_libraries import optimizers
from jax.example_libraries.optimizers import Optimizer, OptimizerState, Params, ParamsFn
from jax.example_libraries.stax import Conv, Dense, elementwise, FanInConcat, FanOut, Flatten, LeakyRelu, parallel, \
    serial, Sigmoid, Relu
from jax.image import resize
from jax.random import KeyArray
from jax.tree_util import tree_map, tree_reduce
from numpy.typing import NDArray
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets.confounded_mnist import digit_colour_scenario
from datasets.utils import image_gallery

tf.config.experimental.set_visible_devices([], 'GPU')
Array = Union[jnp.ndarray, NDArray, Any]
Shape = Tuple[int, ...]
InitFn = Callable[[KeyArray, Shape], Tuple[Shape, Params]]
ApplyFn = Callable
StaxLayer = Tuple[InitFn, ApplyFn]
StaxLayerConstructor = Callable[..., StaxLayer]
UpdateFn = Callable[[int, OptimizerState, Any, KeyArray], Tuple[OptimizerState, Array, Any]]
InitOptimizerFn = Callable[[Params, Optimizer], Tuple[OptimizerState, UpdateFn, ParamsFn]]
Model = Tuple[InitFn, ApplyFn, InitOptimizerFn]


def softplus(x, threshold: float = 20.):
    return (nn.softplus(x) * (x < threshold)) + (x * (x >= threshold)) + 1e-5


class ColourMNIST(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        root = os.path.join(root, 'train') if train else os.path.join(root, 'test')
        self.images = np.load(os.path.join(root, 'images.npy'))
        pickle_load = lambda *a, **k: np.load(*a, allow_pickle=True, **k)
        self.parents = pickle_load(os.path.join(root, 'parents.npy')).item()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample = {}
        sample['x'] = self.images[idx]
        sample['digit'] = self.parents['digit'][idx]
        sample['colour'] = self.parents['colour'][idx]
        return sample


def to_numpy_iterator(data: tf.data.Dataset, batch_size: int, drop_remainder: bool = True) -> Any:
    return tfds.as_numpy(data.batch(batch_size, drop_remainder=drop_remainder).prefetch(tf.data.AUTOTUNE))


def stax_wrapper(fn: Callable[[Array], Array]) -> StaxLayer:
    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        return input_shape, ()

    def apply_fn(params: Params, inputs: Array, **kwargs: Any) -> Array:
        return fn(inputs)

    return init_fn, apply_fn


def reshape(output_shape: Shape) -> StaxLayer:
    def init_fun(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        return output_shape, ()

    def apply_fun(params: Params, inputs: Array, **kwargs: Any) -> Array:
        return jnp.reshape(inputs, output_shape)

    return init_fun, apply_fun


def up_sample(new_shape: Shape) -> StaxLayer:
    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        return new_shape, ()

    def apply_fn(params: Params, inputs: Array, **kwargs: Any) -> Array:
        return vmap(partial(resize, shape=new_shape[1:], method='nearest'))(inputs)

    return init_fn, apply_fn


Reshape = reshape
UpSample = up_sample


def standard_vae(parent_dims: Dict[str, int],
                 latent_dim: int = 16,
                 hidden_dim: int = 128) -> Model:
    """ Implements VAE with independent normal posterior, standard normal prior and, standard normal likelihood (l2)"""
    assert len(parent_dims) > 0
    # enc_init_fn, enc_apply_fn = serial(parallel(Flatten, Flatten), FanInConcat(axis=-1), Dense(512), Relu, Dense(512),
    #                                    Relu,
    #                                    FanOut(2), parallel(Dense(latent_dim), serial(Dense(latent_dim), Exp)))
    # dec_init_fn, dec_apply_fn = serial(FanInConcat(axis=-1), Dense(512), Relu, Dense(512), Relu, Dense(28 * 28 * 3),
    #                                    Reshape((-1, 28, 28, 3)), Sigmoid)
    n_channels = hidden_dim // 4

    encoder_layers = (
        Conv(n_channels, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LeakyRelu,
        Conv(n_channels, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LeakyRelu,
        Flatten, Dense(hidden_dim), LeakyRelu)
    decoder_layers = (
        Dense(hidden_dim), LeakyRelu, Dense(n_channels * 7 * 7),
        LeakyRelu,
        Reshape((-1, 7, 7, n_channels)), UpSample((-1, 14, 14, n_channels)),
        Conv(n_channels, filter_shape=(5, 5), strides=(1, 1), padding='SAME'), LeakyRelu,
        UpSample((-1, 28, 28, n_channels)),
        Conv(3, filter_shape=(5, 5), strides=(1, 1), padding='SAME'),
        Conv(3, filter_shape=(1, 1), strides=(1, 1), padding='SAME'), Sigmoid)

    enc_init_fn, enc_apply_fn = serial(parallel(serial(*encoder_layers), stax_wrapper(lambda x: x)),
                                       FanInConcat(axis=-1), Dense(hidden_dim), LeakyRelu,
                                       Dense(hidden_dim), LeakyRelu,
                                       FanOut(2),
                                       parallel(Dense(latent_dim), serial(Dense(latent_dim)),
                                                elementwise(softplus)))
    dec_init_fn, dec_apply_fn = serial(FanInConcat(axis=-1), *decoder_layers)

    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        k1, k2 = random.split(rng, 2)
        c_dim = sum(parent_dims.values())
        (enc_output_shape, _), enc_params = enc_init_fn(k1, (input_shape, (-1, c_dim)))
        output_shape, dec_params = dec_init_fn(k2, (enc_output_shape, (-1, c_dim)))
        return output_shape, (enc_params, dec_params)

    # @vmap
    # def _kl(mu: Array, variance: Array) -> Array:
    #     return 0.5 * jnp.sum(variance + mu ** 2. - 1. - jnp.log(variance))

    # def rsample(rng: KeyArray, mu: Array, variance: Array) -> Array:
    #     return mu + jnp.sqrt(variance) * random.normal(rng, mu.shape)

    # @vmap
    # def _log_pdf(image: Array, recon: Array, eps: float = 1e-12) -> Array:
    #     return jnp.sum(image * jnp.log(recon + eps) + (1. - image) * jnp.log(1. - recon + eps))

    # log_px = torch.sum(
    #     x * torch.log(x_loc + 1e-12) + (1 - x) * torch.log(1 - x_loc + 1e-12), dim=(1, 2, 3))
    # @vmap
    # def _log_pdf(image: Array, recon: Array, variance: float = .001) -> Array:
    #     return -.5 * jnp.sum((image - recon) ** 2. / variance + jnp.log(2 * jnp.pi * variance))

    def apply_fn(params: Params, inputs: Any, rng: KeyArray) -> Tuple[Array, Dict[str, Array]]:
        k1, k2 = random.split(rng, 2)
        enc_params, dec_params = params
        image, parents = inputs
        _parents = jnp.concatenate([parents[parent_name] for parent_name in sorted(parent_dims.keys())], axis=-1)
        mean_z, scale_z = enc_apply_fn(enc_params, (image, _parents))
        z = mean_z + scale_z * random.normal(k1, mean_z.shape)
        recon = dec_apply_fn(dec_params, (z, _parents))
        log_pdf = jnp.sum(image * jnp.log(recon + 1e-12) + (1. - image) * jnp.log(1. - recon + 1e-12),
                          axis=(1, 2, 3))  # _log_pdf(image, recon)

        # var_scale = .1
        # log_pdf = -.5 * jnp.sum((image - recon) ** 2. / var_scale + jnp.log(2 * jnp.pi * var_scale), axis=(1, 2, 3))
        var_z = scale_z ** 2.
        kl = 0.5 * jnp.sum(var_z + (mean_z ** 2.) - 1. - jnp.log(var_z + 1e-12), axis=-1)  # _kl(mean_z, variance_z)

        elbo = log_pdf - kl
        loss = jnp.mean(-elbo)
        # conditional samples
        keys = random.split(k2, len(parent_dims))
        random_parents = {'colour': parents['colour'], 'digit': parents['colour']}
        _parents = jnp.concatenate([random_parents[parent_name] for parent_name in sorted(parent_dims.keys())], axis=-1)
        mean_z, scale_z = enc_apply_fn(enc_params, (image, _parents))
        z = mean_z + scale_z * random.normal(rng, mean_z.shape)
        random_recon = dec_apply_fn(dec_params, (z, _parents))
        # loss = loss + 1e-1* jnp.sqrt(tree_reduce(lambda x, y: x + y, tree_map(lambda x: jnp.sum(x ** 2), params)))

        return loss, {'image': image,
                      'recon': recon,
                      'log_pdf': log_pdf,
                      'kl': kl,
                      'elbo': elbo,
                      'random_recon': random_recon,
                      'variance': jnp.mean(var_z, axis=-1),
                      'mean': jnp.mean(mean_z, axis=-1),
                      'snr': jnp.abs(jnp.mean(mean_z / var_z, axis=-1)),
                      'loss': loss}

    def init_optimizer_fn(params: Params, optimizer: Optimizer) -> Tuple[OptimizerState, UpdateFn, ParamsFn]:
        opt_init, opt_update, get_params = optimizer

        @jit
        def update(i: int, opt_state: OptimizerState, inputs: Any, rng: KeyArray) -> Tuple[OptimizerState, Array, Any]:
            (loss, outputs), grads = value_and_grad(apply_fn, has_aux=True)(get_params(opt_state), inputs, rng)
            opt_state = opt_update(i, grads, opt_state)
            grad_norm = jnp.sqrt(tree_reduce(lambda x, y: x + y, tree_map(lambda x: jnp.sum(x ** 2), grads)))
            outputs['grad_norm'] = grad_norm[jnp.newaxis]

            return opt_state, loss, outputs

        return opt_init(params), update, get_params

    return init_fn, apply_fn, init_optimizer_fn


def get_writer_fn(job_dir: Path, name: str, logging_fn: Optional[Callable[[str], None]] = None) \
        -> Callable[[Dict, int, Optional[str]], None]:
    logdir = (job_dir / 'logs' / name)
    logdir.mkdir(exist_ok=True, parents=True)
    writer = tf.summary.create_file_writer(str(logdir))

    def log_value(value: Any, tag: str, step: int) -> None:
        value = jnp.mean(value) if value.ndim <= 1 else value
        tf.summary.scalar(tag, value, step) if value.size == 1 else \
            tf.summary.image(tag, np.expand_dims(
                image_gallery(value, num_images_to_display=min(128, value.shape[0])), 0), step)
        if logging_fn is not None:
            logging_fn(f'epoch: {step:d}: \t{tag}: {value:.2f}') if value.size == 1 else None

    def log_eval(evaluation: Dict, step: int, tag: Optional[str] = None) -> None:
        with writer.as_default():
            for new_tag, value in evaluation.items():
                new_tag = tag + '/' + new_tag if tag else new_tag
                log_eval(value, step, new_tag) if isinstance(value, Dict) else log_value(value, new_tag, step)

    return log_eval


def accumulate_output(new_output: Any, cum_output: Optional[Any]) -> Any:
    def update_value(value: Array, new_value: Array) -> Array:
        return value + new_value if value.ndim == 0 \
            else (jnp.concatenate((value, new_value)) if value.ndim == 1 else new_value)

    to_cpu = partial(jax.device_put, device=jax.devices('cpu')[0])
    new_output = jax.tree_map(to_cpu, new_output)
    return new_output if cum_output is None else jax.tree_multimap(update_value, cum_output, new_output)


def train(model: Model,
          job_dir: Path,
          seed: int,
          train_data: Iterable,
          test_data: Optional[Iterable],
          input_shape: Shape,
          optimizer: Optimizer,
          num_steps: int,
          log_every: int,
          eval_every: int,
          save_every: int) -> Params:
    model_path = job_dir / f'model.npy'
    job_dir.mkdir(exist_ok=True, parents=True)
    train_writer = get_writer_fn(job_dir, 'train')
    test_writer = get_writer_fn(job_dir, 'test')

    init_fn, apply_fn, init_optimizer_fn = model
    rng = jax.random.PRNGKey(seed)
    _, params = init_fn(rng, input_shape)
    opt_state, update, get_params = init_optimizer_fn(params, optimizer)
    eye = np.eye(10)
    for step, d in tqdm(enumerate(itertools.cycle(train_data)), total=num_steps):
        # inputs = (d['x'].numpy() / 255., {'digit': eye[d['digit'].numpy()], 'colour': eye[d['colour'].numpy()]})
        inputs = d
        if step >= num_steps:
            break
        rng, _ = jax.random.split(rng)
        opt_state, loss, output = update(step, opt_state, inputs, rng)
        if step % log_every == 0:
            train_writer(output, step, None)
        if jnp.isnan(loss):
            raise ValueError('NaN loss!')

        if step % eval_every == 0 and test_data is not None:
            cum_output = None
            for d in test_data:
                # test_inputs = (
                #     d['x'].numpy() / 255., {'digit': eye[d['digit'].numpy()], 'colour': eye[d['colour'].numpy()]})
                rng, _ = jax.random.split(rng)
                test_inputs = d
                _, output = apply_fn(get_params(opt_state), test_inputs, rng=rng)
                cum_output = accumulate_output(output, cum_output)
            test_writer(cum_output, step, None)

        if step % save_every == 0 or step == num_steps - 1:
            jnp.save(str(model_path), get_params(opt_state))

    return get_params(opt_state)


if __name__ == '__main__':
    data_dir = Path('/vol/biomedic/users/mm6818/projects/grand_canyon/data')
    train_datasets, test_dataset, parent_dims, _, marginals, input_shape = digit_colour_scenario(data_dir, False, False)
    parent_names = parent_dims.keys()

    batch_size = 512
    train_data = to_numpy_iterator(train_datasets[frozenset()], batch_size, drop_remainder=True)
    test_data = to_numpy_iterator(test_dataset, batch_size, drop_remainder=False)
    # ##
    # data_dir = '/vol/biomedic/users/mm6818/projects/grand_canyon/data/mnist_digit_colour'
    # trans = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.RandomCrop((28, 28), padding=2),
    #     transforms.ToTensor()
    # ])
    #
    # trans_test = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.ToTensor()
    # ])
    #
    # train_set = ColourMNIST(data_dir, train=True, transform=trans)
    # test_set = ColourMNIST(data_dir, train=False, transform=trans_test)

    # kwargs = {'batch_size': batch_size, 'num_workers': 0, 'pin_memory': False}
    #
    # train_data = torch.utils.data.DataLoader(
    #     train_set, shuffle=True, drop_last=True, **kwargs)
    # test_data = torch.utils.data.DataLoader(
    #     test_set, shuffle=False, **kwargs)
    ##

    # schedule = optimizers.piecewise_constant(boundaries=[5000, 8000], values=[1e-4, 1e-4 / 2, 1e-4 / 8])
    # optimizer = optimizers.adam(step_size=schedule, b1=0.9, b2=.999)

    ###

    ###
    # def classifier(n_classes: int):
    # width = 128
    # conv = (Conv(width // 4, filter_shape=(3, 3), strides=(1, 1), padding='SAME'), Relu,
    #         Conv(width // 4, filter_shape=(3, 3), strides=(2, 2), padding='SAME'), Relu,
    #         Conv(width // 4, filter_shape=(3, 3), strides=(1, 1), padding='SAME'), Relu,
    #         Conv(width // 4, filter_shape=(3, 3), strides=(2, 2), padding='SAME'), Relu,
    #         Conv(width // 4, filter_shape=(3, 3), strides=(1, 1), padding='SAME'), Relu,
    #         Conv(width // 4, filter_shape=(3, 3), strides=(2, 2), padding='SAME'), Relu,
    #         Flatten, Dense(width*2), Relu, Dense(10))
    # classifiers {key: serial()}

    model = standard_vae(parent_dims, latent_dim=16, hidden_dim=256)
    params = train(model=model,
                   job_dir=Path('/tmp/test_job'),
                   seed=32345,
                   train_data=train_data,
                   test_data=test_data,
                   input_shape=input_shape,
                   optimizer=optimizers.adam(step_size=1e-3),
                   num_steps=10000,
                   log_every=1,
                   eval_every=250,
                   save_every=250)
