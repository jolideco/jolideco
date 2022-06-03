import numpy as np
from numpy.testing import assert_allclose
import torch
from astropy.convolution import convolve, Gaussian2DKernel
from jolideco.utils.torch import convolve_fft_torch


def test_convolve_fft_torch():
    image = np.zeros((9, 9))
    image[3:6, 3:6] = 1
    kernel = np.ones((3, 3))
    kernel = kernel / kernel.sum()

    result_ref = convolve(image, kernel)

    result_torch = convolve_fft_torch(
        image=torch.from_numpy(image[None, None]),
        kernel=torch.from_numpy(kernel[None, None]),
    )

    result = result_torch.detach().numpy()
    assert_allclose(result_ref, result[0][0], atol=1e-12)


def test_convolve_fft_torch_3d():
    images = np.zeros((3, 31, 31))
    images[:, 15, 15] = 1
    kernels = np.stack([Gaussian2DKernel(_, x_size=31) for _ in [1, 2, 3]])

    results = []

    for image, kernel in zip(images, kernels):
        result_image = convolve(image, kernel)
        results.append(result_image)

    result_ref = np.stack(results)

    result_torch = convolve_fft_torch(
        image=torch.from_numpy(images[None]),
        kernel=torch.from_numpy(kernels[None]),
    )

    result = result_torch.detach().numpy()
    assert_allclose(result_ref, result[0], atol=1e-12, rtol=1e-6)