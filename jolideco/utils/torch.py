import torch

__all__ = ["convolve_fft_torch", "view_as_overlapping_patches_torch"]


def _centered(arr, newshape):
    # Return the center newshape portion of the array.
    newshape = torch.tensor(newshape)
    currshape = torch.tensor(arr.shape)
    startind = torch.div(currshape - newshape, 2, rounding_mode="trunc")
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def view_as_overlapping_patches_torch(image, shape):
    """View tensor as overlapping rectangular patches

    Parameters
    ----------
    image : `~torch.Tensor`
        Image tensor
    shape : tuple
        Shape of the patches.

    Returns
    -------
    patches : `~torch.Tensor`
        Tensor of overlapping patches of shape
        (n_patches, patch_shape_flat)

    """
    step = shape[0] // 2
    patches = image.unfold(2, shape[0], step)
    patches = patches.unfold(3, shape[0], step)
    ncols = shape[0] * shape[1]
    return torch.reshape(patches, (-1, ncols))


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
    image, kernel = image[0][0], kernel[0][0]

    shape = [image.shape[i] + kernel.shape[i] - 1 for i in range(image.ndim)]

    image_ft = torch.fft.rfft2(image, s=shape)
    kernel_ft = torch.fft.rfft2(kernel, s=shape)
    result = torch.fft.irfft2(image_ft * kernel_ft)
    return _centered(result, image.shape)[None, None]
