import logging

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "convolve_fft_torch",
    "view_as_overlapping_patches_torch",
    "view_as_random_overlapping_patches_torch",
    "view_as_windows_torch",
    "TORCH_DEFAULT_DEVICE",
    "interp1d_torch",
    "grid_weights",
    "get_default_generator",
]

TORCH_DEFAULT_DEVICE = "cpu"

log = logging.getLogger(__name__)


def transpose(x):
    """Transpose tensor"""
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))


def grid_weights(x, y, x0, y0):
    """Compute 4-pixel weights such that centroid is preserved."""
    dx = torch.abs(x - x0)
    dx = torch.where(dx < 1, 1 - dx, 0)

    dy = torch.abs(y - y0)
    dy = torch.where(dy < 1, 1 - dy, 0)
    return dx * dy


def cycle_spin(image, patch_shape, generator):
    """Cycle spin

    Parameters
    ----------
    image : `~pytorch.Tensor`
        Image tensor
    patch_shape : tuple of int
        Patch shape
    generator : `~torch.Generator`
        Random number generator

    Returns
    -------
    image, shifts: `~pytorch.Tensor`, tuple of `~pytorch.Tensor`
        Shifted tensor
    """
    x_max, y_max = patch_shape
    x_width, y_width = x_max // 4, y_max // 4
    shift_x = torch.randint(
        -x_width, x_width + 1, (1,), generator=generator, device=image.device
    )
    shift_y = torch.randint(
        -y_width, y_width + 1, (1,), generator=generator, device=image.device
    )
    shifts = (int(shift_x), int(shift_y))

    dims = (image.ndim - 2, image.ndim - 1)
    return torch.roll(image, shifts=shifts, dims=dims), shifts


def cycle_spin_subpixel(image, generator):
    """Cycle spin

    Parameters
    ----------
    image : `~pytorch.Tensor`
        Image tensor
    generator : `~torch.Generator`
        Random number generator

    Returns
    -------
    image: `~pytorch.Tensor`
        Shifted tensor
    """
    grid = torch.arange(-1, 2, device=image.device)
    y, x = torch.meshgrid(grid, grid, indexing="ij")
    x_0 = torch.rand(1, generator=generator, device=image.device) - 0.5
    y_0 = torch.rand(1, generator=generator, device=image.device) - 0.5
    kernel = grid_weights(x, y, x_0, y_0)
    kernel = kernel.reshape((1, 1, 3, 3))
    return F.conv2d(image, kernel, padding="same")


def interp1d_torch(x, xp, fp, **kwargs):
    """Linear interpolation

    Parameters
    ----------
    x : `~torch.Tensor`
        x values
    xp : `~torch.Tensor`
        x values
    fp : `~torch.Tensor`
        x values

    Returns
    -------
    interp : `~torch.Tensor`
        Interpolated x values
    """
    idx = torch.clip(torch.searchsorted(xp, x), 0, len(xp) - 2)
    y0, y1 = fp[idx - 1], fp[idx]
    x0, x1 = xp[idx - 1], xp[idx]

    weights = (x - x0) / (x1 - x0)

    return torch.lerp(y0, y1, weights, **kwargs)


def view_as_windows_torch(image, shape, stride):
    """View tensor as overlapping rectangular windows

    Parameters
    ----------
    image : `~torch.Tensor`
        Image tensor
    shape : tuple
        Shape of the patches.
    stride : int
        Stride of the patches. By default it is half of the patch size.

    Returns
    -------
    windows : `~torch.Tensor`
        Tensor of overlapping windows

    """
    if stride is None:
        stride = shape[0] // 2

    windows = image.unfold(image.ndim - 2, shape[0], stride)
    return windows.unfold(image.ndim - 1, shape[0], stride)


def view_as_overlapping_patches_torch(image, shape, stride=None):
    """View tensor as overlapping rectangular patches

    Parameters
    ----------
    image : `~torch.Tensor`
        Image tensor
    shape : tuple
        Shape of the patches.
    stride : int
        Stride of the patches. By default it is half of the patch size.

    Returns
    -------
    patches : `~torch.Tensor`
        Tensor of overlapping patches of shape
        (n_patches, patch_shape_flat)

    """
    if stride is None:
        stride = shape[0] // 2

    patches = view_as_windows_torch(image=image, shape=shape, stride=stride)
    ncols = shape[0] * shape[1]
    return torch.reshape(patches, (-1, ncols))


def view_as_random_overlapping_patches_torch(image, shape, stride, generator):
    """View tensor as randomly ("jittered") overlapping rectangular patches

    Parameters
    ----------
    image : `~torch.Tensor`
        Image tensor
    shape : tuple
        Shape of the patches.
    stride : int
        Stride of the patches. By default it is half of the patch size.
    generator : `~torch.Generator`
        Random number generator

    Returns
    -------
    patches : `~torch.Tensor`
        Tensor of overlapping patches of shape
        (n_patches, patch_shape_flat)

    """
    overlap = max(shape) - stride
    ny, nx = image.shape[-2:]
    idx = torch.arange(overlap, nx - stride - overlap, stride, device=image.device)
    idy = torch.arange(overlap, ny - stride - overlap, stride, device=image.device)

    jitter_x = torch.randint(
        low=-overlap,
        high=overlap + 1,
        size=(len(idx),),
        generator=generator,
        device=image.device,
    )

    jitter_y = torch.randint(
        low=-overlap,
        high=overlap + 1,
        size=(len(idy),),
        generator=generator,
        device=image.device,
    )

    idx += jitter_x
    idy += jitter_y

    idy, idx = torch.meshgrid(idy, idx, indexing="ij")

    patches = view_as_windows_torch(image=image, shape=shape, stride=1)

    # idx = torch.clip(idx, 0, patches.shape[-1])
    # idy = torch.clip(idy, 0, patches.shape[-2])

    patches = patches[:, :, idy, idx]
    size = np.multiply(*shape)
    n_patches = np.multiply(*idx.shape)
    patches = torch.reshape(patches, (n_patches, size))
    return patches


def _centered(arr, newshape):
    # Return the center newshape portion of the array.
    newshape = torch.tensor(newshape)
    currshape = torch.tensor(arr.shape)
    startind = torch.div(currshape - newshape, 2, rounding_mode="trunc")
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def convolve_fft_torch(image, kernel):
    """Convolve FFT for torch tensors

    Parameters
    ----------
    image : `~torch.Tensor`
        Image tensor
    kernel : `~torch.Tensor`
        Kernel tensor

    Returns
    -------
    result : `~torch.Tensor`
        Convolution result
    """
    # TODO: support 4D tensors
    image_2d, kernel_2d = image[0][0], kernel[0][0]

    shape = [image_2d.shape[i] + kernel_2d.shape[i] - 1 for i in range(image_2d.ndim)]

    image_ft = torch.fft.rfft2(image, s=shape)
    kernel_ft = torch.fft.rfft2(kernel, s=shape)
    result = torch.fft.irfft2(image_ft * kernel_ft, s=shape)
    return _centered(result, image.shape)


def get_default_generator(device):
    """Get default torch generator

    Parameters
    ----------
    device : str
        Device name

    Returns
    -------
    generator : `~torch.Generator`
        Random number generator
    """
    try:
        generator = torch.Generator(device=device)
    except RuntimeError:
        log.warning(
            f"Device {device} not available, falling back to {TORCH_DEFAULT_DEVICE}"
        )
        generator = torch.Generator(device=TORCH_DEFAULT_DEVICE)

    return generator
