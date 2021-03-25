from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.fft
import torch.nn.functional as F
from PIL import Image
import matplotlib as mpl

sobel_2d = [[[-1, 1], [0, 0]], [[0, -1], [0, 1]]]


# sobel_3d = [[[-1, 1], [0, 0], [0, 0]], [[0, -1], [0, 1], [0, 0]], [[0, 0], [0, 0], [0, 1]]]


def numpy_to_pil_image(array: np.ndarray) -> Image:
    if array.shape[0] == 1:
        array = np.concatenate([array] * 3, axis=0)
    elif array.shape[0] != 3:
        raise ValueError('Unsupported number of dimensions')
    min_ = np.min(array, axis=(1, 2), keepdims=True)
    max_ = np.max(array, axis=(1, 2), keepdims=True)
    rgb = np.moveaxis((255 * (array - min_) / (max_ - min_)).astype(np.uint8), 0, -1)
    return Image.fromarray(rgb)


def entropy_from_histogram(histogram: np.ndarray):
    p = histogram / np.sum(histogram)
    entropy = - np.sum(p * np.log(p + 1e-100) / np.log(histogram.size))
    return entropy


def fft_filter(img_fft_filtered: torch.Tensor, f_low: Optional[int] = None, f_high: Optional[int] = None) -> Tuple[
    torch.Tensor, torch.Tensor]:
    img_fft_filtered = img_fft_filtered.clone()
    if f_high is not None:
        img_fft_filtered[..., f_high:-f_high, :] = 0
        img_fft_filtered[..., f_high:] = 0
    if f_low is not None:
        img_fft_filtered[..., :f_low, :f_low] = 0

    img_recon = torch.fft.irfftn(img_fft_filtered, dim=(2, 3))
    return img_recon, img_fft_filtered


def calc_power_spectrum(img_fft: torch.Tensor):
    return img_fft.abs() ** 2


def fft_plot(img: torch.Tensor, f_low: Optional[int] = None, f_high: Optional[int] = None, bins: int = 256) -> None:
    img_fft = torch.fft.rfftn(img, dim=(2, 3))
    img_recon, img_fft_filtered = fft_filter(img_fft, f_low, f_high)
    residual = img[..., :-1] - img_recon
    residual_fft = img_fft - img_fft_filtered

    fig, axs = plt.subplots(4, 3, gridspec_kw={'wspace': .1, 'hspace': .2})

    for j, (name, img, img_fft) in enumerate(zip(['Image', 'Reconstruction', 'Residual'],
                                                 [img, img_recon, residual],
                                                 [img_fft, img_fft_filtered, residual_fft])):
        # image
        array = img[0].numpy()
        axs[0, j].set_title(name)
        axs[0, j].imshow(numpy_to_pil_image(array))
        axs[0, j].tick_params(labelbottom=False, labelleft=False)

        # histogram
        hist, _, _ = axs[1, j].hist(array.flatten(), bins=bins, log=True)
        axs[1, j].text(0.5, 0.9, f'entropy = {entropy_from_histogram(hist):.3f}', horizontalalignment='center',
                       verticalalignment='center', transform=axs[1, j].transAxes)
        axs[1, j].tick_params(labelbottom=True, labelleft=True if j == 0 else False)

        # power spectrum
        power_spectrum = np.fft.fftshift(calc_power_spectrum(img_fft)[0, 0].numpy(), axes=[0])
        power_spectrum = np.concatenate((np.flip(power_spectrum[..., 1:], axis=1), power_spectrum), axis=1)
        log_power_spectrum = np.log(power_spectrum + 1e-20)
        axs[2, j].imshow(log_power_spectrum, cmap='inferno')

        spectral_dist = power_spectrum / np.sum(power_spectrum)
        entropy = - np.sum(spectral_dist * np.log(spectral_dist + 1e-10) / np.log(power_spectrum.size))
        axs[2, j].text(0.5, 0.9, f'spectral_entropy = {entropy:.3f}', horizontalalignment='center',
                       verticalalignment='center', transform=axs[2, j].transAxes, fontsize=8,
                       bbox={'facecolor': 'white', 'pad': 1})
        axs[2, j].tick_params(labelbottom=True, labelleft=True if j == 0 else False)

        # entropy measures spread of frequencies?
        x = np.indices(spectral_dist.shape) - np.expand_dims(np.array(spectral_dist.shape) // 2, [1, 2])
        mu = np.sum(np.expand_dims(spectral_dist, 0) * x, axis=(1, 2))
        var = np.sum(np.expand_dims(spectral_dist, 0) * (x - np.expand_dims(mu, (1, 2))) ** 2, axis=(1, 2))
        print(mu, var)
        # grad entropy
        grads = calc_image_grads(img)
        x_grad = grads[0, 0].numpy()
        y_grad = grads[1, 0].numpy()
        hist, _, _, _ = axs[3, j].hist2d(x_grad.flatten(), y_grad.flatten(), bins=bins, norm=mpl.colors.LogNorm())
        axs[3, j].text(0.5, 0.9, f'entropy* = {entropy_from_histogram(hist):.3f}', horizontalalignment='center',
                       verticalalignment='center', transform=axs[3, j].transAxes, fontsize=8,
                       bbox={'facecolor': 'white', 'pad': 1})

    fig.show()


def calc_image_grads(image: torch.Tensor) -> torch.Tensor:
    # flatten channel dimension
    shape = image.shape
    image = image.flatten(0, 1).unsqueeze(1)
    weight = torch.tensor(sobel_2d).unsqueeze(1).type(torch.float32)
    image = F.conv2d(image, weight=weight)
    return image.reshape((*shape[:2], 2, *image.shape[2:])).movedim(2, 0)


def image_entropy_calculation(img: torch.Tensor, bins: int = 256):
    grads = calc_image_grads(img)
    x_grad = grads[0, 0].numpy()
    y_grad = grads[1, 0].numpy()

    fig, axs = plt.subplots(2, 2, gridspec_kw={'wspace': .1, 'hspace': .05})
    axs[0, 0].imshow(numpy_to_pil_image(img[0].numpy()))
    axs[0, 1].imshow(numpy_to_pil_image(x_grad))
    axs[1, 1].imshow(numpy_to_pil_image(y_grad))

    hist, _, _, _ = axs[1, 0].hist2d(x_grad.flatten(), y_grad.flatten(), bins=bins, norm=mpl.colors.LogNorm())
    axs[1, 0].text(0.5, 0.9, f'entropy = {entropy_from_histogram(hist):.3f}', horizontalalignment='center',
                   verticalalignment='center', transform=axs[1, 0].transAxes)

    for ax in axs.flat:
        ax.label_outer()
    fig.show()


def analysis(img_tensor):
    fft_plot(img_tensor, f_low=None, f_high=5)


image = Image.open('../data/Brats17_2013_24_1_flair__slice_91.png')
# plt.imshow(image)
# plt.show()

# convert to grayscale array
img = np.moveaxis(np.sum(np.array(image)[..., :3], axis=-1, keepdims=True), -1, 0)
img_tensor = torch.tensor(img.astype(np.float32)).unsqueeze(0)
analysis(img_tensor)
#
# # # # random noise
random_noise = torch.randn_like(img_tensor)
analysis(random_noise)
# #
# # gradient image
grid_x, grid_y = torch.meshgrid(torch.linspace(0, 1, img_tensor.shape[2]), torch.linspace(0, 1, img_tensor.shape[2]))
grad = grid_x.unsqueeze(0).unsqueeze(0)
analysis(grad)
