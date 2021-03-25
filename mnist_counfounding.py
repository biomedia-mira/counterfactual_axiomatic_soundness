import itertools
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence, Tuple, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter
from hessian_penalty_pytorch import hessian_penalty
import torch.autograd.functional

COLORS = ((1, 0, 0),
          (0, 1, 0),
          (0, 0, 1),
          (1, 1, 0),
          (1, 0, 1),
          (0, 1, 1),
          (1, 1, 1),
          (.5, 0, 0),
          (0, .5, 0),
          (0, 0, .5))


class Encoder(nn.Module):
    def __init__(self, in_channels: int, num_latents: int, latent_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_latents * latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = x.reshape(-1, self.num_latents, self.latent_dim)
        return x


class Decoder(nn.Module):
    def __init__(self, num_latents: int, latent_dim: int, num_classes: int):
        super().__init__()
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.lin = nn.Linear(num_latents * latent_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x.reshape(-1, self.num_latents * self.latent_dim))


class Model2(nn.Module):
    def __init__(self, in_channels: int, num_latents: int, num_classes: int, latent_dim: int = 64):
        super().__init__()
        self.encoder = Encoder(in_channels, num_latents, latent_dim)
        self.decoder = Decoder(num_latents, latent_dim, num_classes)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) \
            -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]:
        z = self.encoder(x)
        logits = self.decoder(z)
        hessian = torch.stack([torch.autograd.functional.hessian(lambda z_: torch.mean(self.decoder(z_)[:, i]),
                                                                 inputs=z, vectorize=True) for i in
                               range(logits.shape[1])])
        n = torch.norm(hessian)
        loss = F.cross_entropy(logits, labels)

        return {'main': logits}, {'main': loss}, loss


class Model(nn.Module):
    def __init__(self, in_channels: int, num_latents: int, num_classes: int, latent_dim: int = 64):
        super().__init__()
        self.encoder = Encoder(in_channels, num_latents, latent_dim)
        self.main_decoder = Decoder(num_latents, latent_dim, num_classes)
        self.decoders = nn.ModuleDict()
        for i in range(num_latents):
            self.decoders[f'decoder_{i:d}'] = Decoder(1, latent_dim, num_classes)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) \
            -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]:

        z = self.encoder(x)
        logits_dict = {'main_decoder': self.main_decoder(z)}
        logits_dict.update({key: decoder(z[:, i]) for i, (key, decoder) in enumerate(self.decoders.items())})

        loss = None
        loss_dict = {}
        if labels is not None:
            loss_dict = {key: F.cross_entropy(logits, labels) for key, logits in logits_dict.items()}
            loss_dict['hessian_penalty'] = hessian_penalty(self.main_decoder, z, G_z=logits_dict['main_decoder'], k=10)
            loss = torch.sum(torch.stack(list(loss_dict.values())))

        return logits_dict, loss_dict, loss


class Mechanism(ABC):
    @abstractmethod
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pass


class Colorize(Mechanism):
    def __init__(self, num_colors: int = 10):
        assert num_colors <= 10
        self.color_mapping = torch.zeros(10, 3)
        for label, color in zip(range(10), itertools.cycle(COLORS[:num_colors])):
            self.color_mapping[label] = torch.tensor(color)

    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return torch.cat([image] * 3, dim=0) * self.color_mapping[label].view(3, 1, 1)


class CounfoundedMNIST(Dataset):
    def __init__(self, root: str, mechanisms: Sequence[Mechanism], train: bool = True, download: bool = False) -> None:
        self.base_dataset = torchvision.datasets.MNIST(root=root, train=train, download=download,
                                                       transform=transforms.ToTensor())
        self.mechanisms = mechanisms
        self.train = train

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, label = self.base_dataset.__getitem__(index)
        for mechanism in self.mechanisms:
            image = mechanism(image, label if self.train else np.random.randint(0, 10))

        return image, label

    def __len__(self):
        return len(self.base_dataset)


def run_epoch(epoch: int,
              num_epochs: int,
              model: Model,
              dataloader: DataLoader,
              writer: SummaryWriter,
              optimizer: Optional[torch.optim.Optimizer] = None):
    total_loss = 0.
    total_loss_dict: Dict[str, float] = {}
    accuracy_dict: Dict[str, float] = {}
    count = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        if optimizer is not None:
            optimizer.zero_grad()
        logits_dict, loss_dict, loss = model(inputs, labels)
        prediction_dict = {key: torch.argmax(logits, dim=-1) for key, logits in logits_dict.items()}
        if optimizer is not None:
            loss.backward()
            optimizer.step()

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
    message = f'{epoch + 1:d}, {num_epochs:d}] total-loss: {total_loss / count:.4f} '
    for key, value in total_loss_dict.items():
        writer.add_scalar(f'loss_{key}', value / count, global_step=epoch)
        message += f'loss_{key}: {value / count:.4f} '
    for key, value in accuracy_dict.items():
        writer.add_scalar(f'accuracy: {key}', 100. * value / count, global_step=epoch)
        message += f'loss_{key}: {100. * value / count:.2} '


def train(job_dir: Path,
          device: torch.device,
          trainloader: DataLoader,
          testloader: DataLoader,
          num_epochs: int,
          eval_every: int = 50):
    job_dir.mkdir(exist_ok=True, parents=True)
    log_dir = job_dir / 'logs'
    if log_dir.exists():
        shutil.rmtree(log_dir)
    (log_dir / 'train').mkdir(exist_ok=True, parents=True)
    train_writer = SummaryWriter(str(log_dir / 'train'))
    (log_dir / 'test').mkdir(exist_ok=True, parents=True)
    test_writer = SummaryWriter(str(log_dir / 'test'))

    in_channels = 3
    num_latents = 2
    num_classes = 10
    latent_dim = 64
    model = Model(in_channels, num_latents, num_classes, latent_dim)
    model = Model2(in_channels, num_latents, num_classes, latent_dim)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        run_epoch(epoch, num_epochs, model, trainloader, train_writer, optimizer)
        if epoch % eval_every == 0 and epoch > 0:
            run_epoch(epoch, num_epochs, model, testloader, test_writer)
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
    path = './data/mnist/'
    batch_size = 24
    num_workers = 0
    trainset = torchvision.datasets.MNIST(path, train=True, download=False, transform=transforms.ToTensor())
    trainset = CounfoundedMNIST(path, mechanisms=[Colorize(10)], train=True, download=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # testset = torchvision.datasets.MNIST(path, train=False, download=False, transform=transforms.ToTensor())
    testset = CounfoundedMNIST(path, mechanisms=[Colorize(10)], train=False, download=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    im = torchvision.utils.make_grid(images)
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
    plt.show()

    job_dir = Path('/tmp/test_run')
    overwrite = True
    if not job_dir.exists() or overwrite:
        train(job_dir, device, trainloader, testloader, 300, eval_every=1)

# class Net(nn.Module):
#     def __init__(self, in_channels, num_outputs):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(9216, 128)
#         self.fc2 = nn.Linear(128, num_outputs)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         z = x
#         x = self.fc2(x)
#         return x, z


# lambda_ = torch.zeros((), requires_grad=True)
# epsilon = 1e-6
# damp = 10 * (epsilon - logit_norm).detach()
# lagrangian = cross_entropy - (-lambda_ - damp) * (epsilon - logit_norm)
#
# epsilon = 2.
# damp = 10 * (epsilon - cross_entropy).detach()
# lagrangian = logit_norm - (-lambda_ - damp) * (epsilon - cross_entropy)
# print(logit_norm.item(), cross_entropy.item(), lambda_.item())
# if lambda_ > 0:
#     lambda_.data = lambda_.data * 0
