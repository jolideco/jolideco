import torch

__all__ = ["convolve_fft_torch"]


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
    image, kernel = image[0][0], kernel[0][0]

    shape = [image.shape[i] + kernel.shape[i] - 1 for i in range(image.ndim)]

    image_ft = torch.fft.rfft2(image, s=shape)
    kernel_ft = torch.fft.rfft2(kernel, s=shape)
    result = torch.fft.irfft2(image_ft * kernel_ft, s=shape)
    return _centered(result, image.shape)[None, None]
