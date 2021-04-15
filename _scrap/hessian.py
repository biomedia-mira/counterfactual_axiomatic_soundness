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
