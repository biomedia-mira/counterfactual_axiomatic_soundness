import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd.functional
import torch.autograd.functional
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader, DataLoader
from torch.utils.tensorboard import SummaryWriter
import jax
from datasets.colormnist import Colorize, get_diagonal_class_conditioned_color_distribution, \
    get_uniform_class_conditioned_color_distribution, show_cm
from datasets.confounded_dataset import CounfoundedDataset
from jax.lax import stop_gradient
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, LogSoftmax
from jax.experimental import optimizers

import time
import itertools

import numpy.random as npr

import jax.numpy as jnp
from jax import jit, grad, random, vmap
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, LogSoftmax
from jax.experimental import optimizers

import flows


def model(img_size: int, c_dim: int):
    rng = random.PRNGKey(0)

    def transform(rng, input_dim: int, output_dim: int):
        net_init, net_apply = stax.serial(Dense(output_dim), Relu, Dense(output_dim))
        out_shape, net_params = net_init(rng, (-1, input_dim))
        return net_params, net_apply

    bijection_init = flows.Serial(
        flows.AffineCoupling(transform),
        flows.InvertibleLinear(),
        flows.ActNorm(),
        flows.AffineCoupling(transform),
        flows.InvertibleLinear(),
        flows.ActNorm()
    )

    params, _direct_fun, _inverse_fun = bijection_init(rng, input_dim=img_size + c_dim)

    def direct_fun(inputs: jnp.ndarray):
        inputs = vmap(lambda array: jnp.append(jnp.zeros(c_dim), jnp.ravel(array)), in_axes=0)(inputs)
        outputs, _ = _direct_fun(params, inputs)
        return outputs[..., :c_dim], outputs[..., c_dim:]

    def inverse_fun(outputs: jnp.ndarray, shape):
        inputs, _ = _inverse_fun(params, outputs)
        return jnp.reshape(inputs[..., c_dim:], (*inputs.shape[:2], *shape))

    def construct_virtual_outputs(c, z):
        output = jnp.concatenate((c, z), axis=-1)
        virtual_outputs = jnp.repeat(output[..., jnp.newaxis, :], c_dim, axis=1)
        virtual_outputs = jax.ops.index_update(virtual_outputs, jax.ops.index[..., :c_dim], jnp.eye(c_dim))
        return virtual_outputs

    def forward_pass(params, batch):
        inputs, _, _ = batch
        input_shape = inputs.shape[1:]

        c, z = direct_fun(inputs)
        virtual_outputs = construct_virtual_outputs(c, z)
        virtual_inputs, _ = stop_gradient(inverse_fun(virtual_outputs, input_shape))
        virtual_c, virtual_z = direct_fun(virtual_inputs)

        return c, virtual_output_color

    def loss(params, batch):
        inputs, targets, colors = batch

        output = direct_fun(inputs)
        c = output[..., -1]

        x = jnp.repeat(output[..., jnp.newaxis, :], 10, axis=1)
        b = jax.ops.index_update(x, jax.ops.index[..., -1], list(range(10)))

        outputs = direct_fun(stop_gradient(inverse_fun(b)))

        preds = predict(params, inputs)
        return -jnp.mean(jnp.sum(preds * targets, axis=1))

    opt_init, opt_update, get_params = optimizers.momentum(step_size=lambda x: 0.001, mass=0.9)

    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, grad(loss)(params, batch), opt_state)

    return 1


def run_epoch(epoch: int,
              num_epochs: int,
              model: Model,
              dataloader: DataLoader,
              writer: SummaryWriter,
              optimizer: Optional[torch.optim.Optimizer] = None,
              prefix: str = 'Train') -> None:
    total_loss = 0.
    total_loss_dict: Dict[str, float] = {}
    accuracy_dict: Dict[str, float] = {}
    count = 0
    for inputs, labels, vars in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        logits_dict, loss_dict, loss = model(inputs, labels)
        prediction_dict = {key: torch.argmax(logits, dim=-1) for key, logits in logits_dict.items()}

        # keep track of stats
        total_loss += loss.item()
        total_loss_dict = total_loss_dict if bool(total_loss_dict) else dict.fromkeys(loss_dict, 0.)
        for key, value in loss_dict.items():
            total_loss_dict[key] += value.item() * labels.size(0)
        accuracy_dict = accuracy_dict if bool(accuracy_dict) else dict.fromkeys(loss_dict, 0)
        for key, value in prediction_dict.items():
            accuracy_dict[key] += (value == labels).sum().item()
        count += labels.size(0)

    # logging
    writer.add_scalar('loss', total_loss / count, global_step=epoch)
    message = f'{prefix} [{epoch + 1:d}, {num_epochs:d}] total-loss: {total_loss / count:.4f} '
    for key, value in total_loss_dict.items():
        writer.add_scalar(f'loss_{key}', value / count, global_step=epoch)
        message += f'loss_{key}: {value / count:.4f} '
    for key, value in accuracy_dict.items():
        writer.add_scalar(f'accuracy: {key}', 100. * value / count, global_step=epoch)
        message += f'accuracy_{key}: {100. * value / count:.2f} '
    print(message)


def train(job_dir: Path,
          device: torch.device,
          trainloader: DataLoader,
          testloader: DataLoader,
          num_epochs: int,
          eval_every: int = 2):
    job_dir.mkdir(exist_ok=True, parents=True)
    log_dir = job_dir / 'logs'
    if log_dir.exists():
        shutil.rmtree(log_dir)
    (log_dir / 'train').mkdir(exist_ok=True, parents=True)
    train_writer = SummaryWriter(str(log_dir / 'train'))
    (log_dir / 'test').mkdir(exist_ok=True, parents=True)
    test_writer = SummaryWriter(str(log_dir / 'test'))

    for epoch in range(num_epochs):

        run_epoch(epoch, num_epochs, model, trainloader, train_writer, optimizer)
        if epoch % eval_every == 0 and epoch > 0:
            run_epoch(epoch, num_epochs, model, testloader, test_writer, prefix='Test')
            torch.save(model.state_dict(), job_dir / 'model.pt')

    print('Finished Training')
    torch.save(model.state_dict(), job_dir / 'model.pt')


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('GPU is not available!')
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
    print(device)

    diagonal_cm = get_diagonal_class_conditioned_color_distribution(10, 10, noise=.05)
    show_cm(diagonal_cm)
    uniform_cm = get_uniform_class_conditioned_color_distribution(10, 10)
    show_cm(uniform_cm)

    root = './data/mnist/'
    batch_size = 24
    num_workers = 0
    base_trainset = torchvision.datasets.MNIST(root=root, train=True, download=False, transform=transforms.ToTensor())
    trainset = CounfoundedDataset(base_trainset, mechanisms=[Colorize(diagonal_cm, base_trainset.targets)])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    base_testset = torchvision.datasets.MNIST(root=root, train=False, download=False, transform=transforms.ToTensor())
    testset = CounfoundedDataset(base_testset, mechanisms=[Colorize(uniform_cm, base_testset.targets)])
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    dataiter = iter(trainloader)
    images, labels, vars = dataiter.next()
    im = torchvision.utils.make_grid(images)
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
    plt.show()

    job_dir = Path('/tmp/test_run')
    overwrite = True
    if not job_dir.exists() or overwrite:
        train(job_dir, device, trainloader, testloader, 300, eval_every=1)
