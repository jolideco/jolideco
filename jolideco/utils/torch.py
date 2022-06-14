import numpy as np
import torch
import torch.nn.functional as F
from jolideco.core import DEVICE_TORCH

__all__ = [
    "convolve_fft_torch",
    "view_as_overlapping_patches_torch",
    "dataset_to_torch",
]


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

    patches = image.unfold(2, shape[0], stride)
    patches = patches.unfold(3, shape[0], stride)
    ncols = shape[0] * shape[1]
    return torch.reshape(patches, (-1, ncols))


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


def dataset_to_torch(
    dataset, upsampling_factor=None, correct_exposure_edges=False, device=DEVICE_TORCH
):
    """Convert dataset to dataset of pytorch tensors

    Parameters
    ----------
    dataset : dict of `~numpy.ndarray`
        Dict containing `"counts"`, `"psf"` and optionally `"exposure"` and `"background"`
    upsampling_factor : int
        Upsampling factor for exposure, background and psf.
    correct_exposure_edges : bool
        Correct psf leakage at the exposure edges.
    device : `~pytorch.Device`
        Pytorch device

    Returns
    -------
    datasets : dict of `~torch.Tensor`
        Dict of pytorch tensors.
    """
    dims = (np.newaxis, np.newaxis)

    dataset_torch = {}

    for key, value in dataset.items():
        tensor = torch.from_numpy(value[dims]).to(device)

        if key in ["psf", "exposure", "background", "flux"] and upsampling_factor:
            tensor = F.interpolate(tensor, scale_factor=upsampling_factor)

        if key in ["psf", "background", "flux"] and upsampling_factor:
            tensor = tensor / upsampling_factor**2

        dataset_torch[key] = tensor

    if correct_exposure_edges:
        exposure = dataset_torch["exposure"]
        weights = convolve_fft_torch(
            image=torch.ones_like(exposure), kernel=dataset_torch["psf"]
        )
        dataset_torch["exposure"] = exposure / weights

    return dataset_torch
