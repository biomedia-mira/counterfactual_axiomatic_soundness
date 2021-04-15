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

from datasets.colormnist import Colorize, get_diagonal_class_conditioned_color_distribution, \
    get_uniform_class_conditioned_color_distribution, show_cm
from datasets.confounded_dataset import CounfoundedDataset


class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, features: List[int]):
        super().__init__()
        self.function = nn.ModuleList()
        f = [in_features, *features, out_features]
        for in_f, out_f in zip(f[:-1], f[1:]):
            self.function.append(nn.Linear(in_f, out_f))
            self.function.append(nn.LeakyReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.function(x)


class Network(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

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
        return x


# class Model(nn.Module):
#     def __init__(self, in_channels: int, num_classes: int):
#         super().__init__()
#         self.network = Network(in_channels, num_classes)
#
#     def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) \
#             -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]:
#         logits = self.network(x)
#         logits_dict = {'1': logits}
#         loss = None
#         loss_dict = {}
#         if labels is not None:
#             loss_dict = {key: F.cross_entropy(logits, labels) for key, logits in logits_dict.items()}
#             loss = torch.sum(torch.stack(list(loss_dict.values())))
#
#         return logits_dict, loss_dict, loss


class Model(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        image_size = 28*28*3
        self.f = MLP(image_size, image_size + 1, [100, 100])
        self.f_inverse = MLP(image_size + 1, image_size, [100, 100])


        self.network = Network(in_channels, num_classes)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None,
                vars: Optional[Dict[str, torch.Tensor]] = None) \
            -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]:

        z_c = self.f(x.flatten(1))
        z = z_c[..., :-1]
        c = z_c[..., -1]
        for c_prime in range(10):
            z_c[:, -1] = c_prime
            z_c_prime = self.f(self.f_inverse(z_c))


        logits = self.network(x)

        logits_dict = {'1': logits}
        loss = None
        loss_dict = {}
        if labels is not None:
            loss_dict = {key: F.cross_entropy(logits, labels) for key, logits in logits_dict.items()}
            loss = torch.sum(torch.stack(list(loss_dict.values())))

        return logits_dict, loss_dict, loss


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

    in_channels = 3
    num_classes = 10
    model = Model(in_channels, num_classes)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
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
