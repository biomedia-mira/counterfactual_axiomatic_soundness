import itertools
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Callable

import flows
import jax
import jax.numpy as jnp
import jax.ops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from jax import partial, random, vmap, jit
from jax.experimental import optimizers, stax
from jax.lax import stop_gradient
from jax.experimental.stax import Dense, Relu, LogSoftmax, Flatten
from torch.utils.data.dataloader import DataLoader, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datasets.colormnist import Colorize, get_diagonal_class_conditioned_color_distribution, \
    get_uniform_class_conditioned_color_distribution, show_cm
from datasets.confounded_dataset import CounfoundedDataset

Params = List[Tuple[jnp.ndarray, ...]]


# from jax.lib import xla_bridge print(xla_bridge.get_backend().platform)


def bijection(input_shape: Tuple[int, ...],
              c_dim: int) -> Tuple[Params,
                                   Callable[[Params, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]],
                                   Callable[[Params, jnp.ndarray], jnp.ndarray]]:
    def transform(rng: jnp.ndarray,
                  input_dim: int,
                  output_dim: int) -> Tuple[Params, Callable[[Params, jnp.ndarray], jnp.ndarray]]:
        init_fun, apply_fun = stax.serial(Dense(output_dim), Relu, Dense(output_dim))
        out_shape, params = init_fun(rng, (-1, input_dim))
        return params, apply_fun

    bijection_init = flows.Serial(
        flows.AffineCoupling(transform),
        flows.InvertibleLinear(),
        flows.ActNorm(),
        flows.AffineCoupling(transform),
        flows.InvertibleLinear(),
        flows.ActNorm()
    )

    rng = random.PRNGKey(0)
    params, _direct_fn, _inverse_fn = bijection_init(rng, input_dim=np.prod(input_shape) + c_dim)

    def direct_fn(params: List[Tuple[jnp.ndarray, ...]], inputs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        inputs = vmap(lambda array: jnp.append(jnp.zeros(c_dim), jnp.ravel(array)), in_axes=0)(inputs)
        outputs, _ = _direct_fn(params, inputs)
        return outputs[..., :c_dim], jnp.reshape(outputs[..., c_dim:], (-1, *input_shape))

    def inverse_fn(params: List[Tuple[jnp.ndarray, ...]], outputs: jnp.ndarray) -> jnp.ndarray:
        inputs, _ = _inverse_fn(params, outputs)
        return jnp.reshape(inputs[..., c_dim:], (-1, *input_shape))

    return params, direct_fn, inverse_fn


def classifier(input_shape: Tuple[int, ...], num_classes: int):
    rng = random.PRNGKey(0)
    init_fun, apply_fun = stax.serial(Flatten, Dense(100), Relu, Dense(100), Relu, Dense(num_classes), LogSoftmax)
    _, params = init_fun(rng, (-1, *input_shape))
    return params, apply_fun


def model(img_shape: Tuple[int, ...], c_dim: int, num_classes: int):
    bijection_params, direct_fn, inverse_fn = bijection(img_shape, c_dim)
    classifier_params, classifier_fn = classifier(img_shape, num_classes)

    def virtual_inverse_pass(params: Params, c: jnp.ndarray, z: jnp.ndarray):
        output = stop_gradient(jnp.concatenate((c, jnp.reshape(z, (z.shape[0], -1))), axis=-1))
        virtual_outputs = jnp.repeat(output[..., jnp.newaxis, :], c_dim, axis=1)
        virtual_outputs = jax.ops.index_update(virtual_outputs, jax.ops.index[..., :c_dim], jnp.eye(c_dim, dtype=float))
        virtual_inputs = vmap(partial(inverse_fn, params=params))(outputs=virtual_outputs)
        return stop_gradient(virtual_inputs)

    def forward_fn(params: Tuple[Params, Params], batch: Tuple[jnp.array, jnp.array, jnp.array]):
        bijection_params, classifier_params = params
        inputs, _, _ = batch
        c, z = direct_fn(bijection_params, inputs)
        virtual_inputs = virtual_inverse_pass(bijection_params, c, z)
        virtual_c, virtual_z = vmap(partial(direct_fn, params=bijection_params))(inputs=virtual_inputs)
        y_hat = classifier_fn(classifier_params, z)
        return y_hat, c, z, virtual_c, virtual_z

    def loss_fn(params: Tuple[Params, Params], batch: Tuple[jnp.array, jnp.array, jnp.array]):
        inputs, targets, vars = batch
        color = vars['color']
        y_hat, c, z, virtual_c, virtual_z = forward_fn(params, batch)
        one_hot_targets = jnp.eye(c_dim)[color]
        virtual_targets = jnp.eye(c_dim)[jnp.array(range(c_dim))]
        color_loss = -jnp.mean(jnp.sum(jax.nn.log_softmax(c, axis=-1) * one_hot_targets, axis=1))
        virtual_color_loss = -jnp.mean(jnp.sum((jax.nn.log_softmax(virtual_c, axis=-1) * virtual_targets), axis=-1))
        label_loss = -jnp.mean(jnp.sum(y_hat * jnp.eye(num_classes)[targets], axis=-1))
        total_loss = label_loss + color_loss + virtual_color_loss
        return total_loss, (label_loss, color_loss, virtual_color_loss, y_hat, c, z, virtual_c, virtual_z)

    def get_eval():

        class Evaluation:
            count, correct_label_count, correct_color_count, virtual_correct_color_count = \
                jnp.array(0), jnp.array(0), jnp.array(0), jnp.array(0)
            total_loss, label_loss, color_loss, virtual_color_loss \
                = jnp.array(0.), jnp.array(0.), jnp.array(0.), jnp.array(0.)
            z, virtual_z = None, None

        def update_eval(evaluation: Evaluation, batch_inputs, batch_outputs):
            inputs, labels, vars = batch_inputs
            color = vars['color']
            total_loss, label_loss, color_loss, virtual_color_loss, y_hat, c, z, virtual_c, virtual_z = batch_outputs
            evaluation.count += inputs.shape[0]
            evaluation.correct_label_count += jnp.sum(labels == jnp.argmax(y_hat, axis=-1))
            evaluation.correct_color_count += jnp.sum(color == jnp.argmax(c, axis=-1))
            evaluation.virtual_correct_color_count += jnp.sum(jnp.argmax(virtual_c, axis=-1) == jnp.array(range(c_dim)))
            evaluation.label_loss += label_loss
            evaluation.color_loss += color_loss
            evaluation.virtual_color_loss += virtual_color_loss
            evaluation.total_loss += total_loss
            evaluation.z = z
            evaluation.virtual_z = virtual_z

        def log_eval(step: int, evaluation: Evaluation, writer: Optional[SummaryWriter], logging_fn=print):
            label_accuracy = 100. * evaluation.correct_label_count / evaluation.count
            color_accuracy = 100. * evaluation.correct_color_count / evaluation.count
            virtual_color_accuracy = 100. * evaluation.virtual_correct_color_count / (c_dim * evaluation.count)

            if writer is not None:
                writer.add_scalar('color_accuracy', color_accuracy.to_py(), step)
                writer.add_scalar('virtual_color_accuracy', virtual_color_accuracy.to_py(), step)
                writer.add_scalar('label_accuracy', label_accuracy.to_py(), step)

                writer.add_scalar('total_loss', evaluation.total_loss.to_py(), step)
                writer.add_scalar('label_loss', evaluation.label_loss.to_py(), step)
                writer.add_scalar('color_loss', evaluation.color_loss.to_py(), step)
                writer.add_scalar('virtual_color_loss', evaluation.virtual_color_loss.to_py(), step)

                writer.add_image('z', torchvision.utils.make_grid(torch.tensor(evaluation.z.to_py())), step)
                for i in range(evaluation.virtual_z.shape[1]):
                    writer.add_image(f'virtual_z_{i:d}',
                                     torchvision.utils.make_grid(torch.tensor(evaluation.virtual_z[:, i].to_py())),
                                     step)
            message = f'epoch: {step:d}:\tlabel_accuracy {label_accuracy:.2f}' \
                      f'\tcolor_accuracy {color_accuracy:.2f}\tvirtual_color_accuracy {virtual_color_accuracy:.2f}' \
                      f'\ttotal_loss {evaluation.total_loss:.4f}\tlabel_loss {evaluation.label_loss:.4f}' \
                      f'\ttrue_loss {evaluation.color_loss:.4f}\tvirtual_loss {evaluation.virtual_color_loss:.4f}'
            logging_fn(message)

        return Evaluation, update_eval, log_eval

    return (bijection_params, classifier_params), loss_fn, get_eval


def get_update_fn(get_params: Callable, opt_update: Callable, loss_fn: Callable, has_aux: bool):
    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        (total_loss, batch_outputs), grads = jax.value_and_grad(loss_fn, has_aux=has_aux)(params, batch)
        opt_state = opt_update(i, grads, opt_state)
        return opt_state, (total_loss, *batch_outputs)

    return update


def train(num_epochs: int,
          trainloader: DataLoader,
          testloader: DataLoader,
          eval_every: int,
          save_every: int) -> None:
    job_dir.mkdir(exist_ok=True, parents=True)
    log_dir = job_dir / 'logs'
    if log_dir.exists():
        shutil.rmtree(log_dir)
    (log_dir / 'train').mkdir(exist_ok=True, parents=True)
    train_writer = SummaryWriter(str(log_dir / 'train'))
    (log_dir / 'test').mkdir(exist_ok=True, parents=True)
    test_writer = SummaryWriter(str(log_dir / 'test'))

    opt_init, opt_update, get_params = optimizers.momentum(step_size=lambda x: 0.0001, mass=0.9)
    params, loss_fn, evaluate = model((3, 28, 28), 10, 10)
    init_eval, update_eval, log_eval = evaluate()
    opt_state = opt_init(params)

    update = get_update_fn(get_params, opt_update, loss_fn, has_aux=True)

    itercount = itertools.count()

    for epoch in range(num_epochs):

        train_eval = init_eval()
        for batch_inputs in tqdm(trainloader):
            # convert from torch to numpy
            batch_inputs = batch_inputs[0].numpy() / 255., batch_inputs[1].numpy(), \
                           {key: value.numpy() for key, value in batch_inputs[2].items()}
            iter = next(itercount)
            opt_state, batch_outputs = update(next(itercount), opt_state, batch_inputs)
            print(batch_outputs[0], batch_outputs[1], batch_outputs[2])
            if jnp.isnan(batch_outputs[0]).any():
                raise ValueError('NaN loss')
            update_eval(train_eval, batch_inputs, batch_outputs)
            log_eval(iter, train_eval, writer=train_writer)

        # if epoch % eval_every == 0:
        #     for batch in testloader:
        #         pass
        if epoch % save_every == 0:
            jnp.save(str(job_dir / 'model.np'), params)


if __name__ == '__main__':

    diagonal_cm = get_diagonal_class_conditioned_color_distribution(10, 10, noise=.05)
    show_cm(diagonal_cm)
    uniform_cm = get_uniform_class_conditioned_color_distribution(10, 10)
    show_cm(uniform_cm)

    root = './data/mnist/'
    batch_size = 24
    num_workers = 0
    base_trainset = torchvision.datasets.MNIST(root=root, train=True, download=True)
    trainset = CounfoundedDataset(base_trainset, mechanisms=[Colorize(diagonal_cm, base_trainset.targets)])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    base_testset = torchvision.datasets.MNIST(root=root, train=False, download=True)
    testset = CounfoundedDataset(base_testset, mechanisms=[Colorize(uniform_cm, base_testset.targets)])
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    dataiter = iter(trainloader)
    images, labels, vars = dataiter.next()
    im = torchvision.utils.make_grid(images)
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
    plt.show()
    job_dir = Path('/tmp/test_run_3')
    overwrite = True
    if not job_dir.exists() or overwrite:
        train(100, trainloader, testloader, eval_every=5, save_every=5)
