import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data.dataloader import DataLoader

from datasets.confounded_dataset import ConfoundedDataset, create_data_stream_fun
from datasets.confounding import get_colorize_fun, get_uniform_confusion_matrix, get_diagonal_confusion_matrix
from datasets.confounding import get_perturbation_fun
from model.model import build_model
from morphomnist.perturb import Swelling
from trainer.training import train


def show_images(dataloader):
    dataiter = iter(dataloader)
    images, _ = dataiter.next()
    im = torchvision.utils.make_grid(images)
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
    plt.show()


def create_counfounded_dataset():
    root = './data/mnist/'
    parent_dims = {'label': 10, 'color': 10, 'swelling': 2}
    base_trainset = torchvision.datasets.MNIST(root=root, train=True, download=True)
    base_testset = torchvision.datasets.MNIST(root=root, train=False, download=True)

    swelling_cm = np.zeros((10, 2))
    swelling_cm[slice(0, 10, 2), 0] = .3
    swelling_cm[slice(0, 10, 2), 1] = .7
    swelling_cm[slice(1, 11, 2), 0] = .9
    swelling_cm[slice(1, 11, 2), 1] = .1
    swelling_params = [{'strength': 1, 'radius': 1}, {'strength': 10, 'radius': 7}]
    swelling_fun = get_perturbation_fun(swelling_cm, base_trainset.targets, 'swelling', Swelling, swelling_params)

    colorize_cm = get_diagonal_confusion_matrix(10, 10, noise=.1)
    colorize_fun = get_colorize_fun(colorize_cm, base_trainset.targets)

    trainset = ConfoundedDataset(base_trainset, mechanisms=[swelling_fun, colorize_fun])

    uniform_cm = get_uniform_confusion_matrix(10, 10)
    testset = ConfoundedDataset(base_testset, mechanisms=[get_colorize_fun(uniform_cm, base_testset.targets)])

    return trainset, testset, parent_dims


if __name__ == '__main__':

    root = './data/mnist/'
    batch_size = 100
    num_workers = 0

    trainset, testset, parent_dims = create_counfounded_dataset()

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    show_images(trainloader)

    train_data_stream, marginals = create_data_stream_fun(trainset, parent_dims,
                                                          {'batch_size': batch_size, 'num_workers': num_workers})
    test_data_stream, _ = create_data_stream_fun(testset, parent_dims,
                                                 {'batch_size': batch_size, 'num_workers': num_workers})
    job_dir = Path('/tmp/test_run_4')
    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(exist_ok=True, parents=True)

    init_fun, apply_fun, init_optimizer_fun = build_model(parent_dims=parent_dims, marginals=marginals)

    overwrite = True
    num_epochs = 100
    if not job_dir.exists() or overwrite:
        train(init_fun=init_fun,
              apply_fun=apply_fun,
              init_optimizer_fun=init_optimizer_fun,
              input_shape=(-1, 3, 28, 28),
              job_dir=job_dir,
              num_epochs=num_epochs,
              train_data_stream=train_data_stream,
              test_data_stream=train_data_stream,
              eval_every=10,
              save_every=10)
