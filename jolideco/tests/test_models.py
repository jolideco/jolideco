import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.convolution import Gaussian2DKernel
import torch
from jolideco.models import NPredModel, FluxComponent


@pytest.fixture
def dataset():
    shape = (1, 1, 25, 25)
    exposure = torch.ones(shape)
    psf = Gaussian2DKernel(3).array
    background = torch.ones(shape)
    return {
        "psf": torch.from_numpy(psf[None, None]),
        "exposure": exposure,
        "background": background,
    }


@pytest.fixture
def dataset_3d():
    shape = (1, 3, 25, 25)
    exposure = torch.ones(shape)
    psf = np.stack([Gaussian2DKernel(_, x_size=25) for _ in [1, 2, 3]])
    background = torch.ones(shape)
    return {
        "psf": torch.from_numpy(psf[None]),
        "exposure": exposure,
        "background": background,
    }


@pytest.fixture
def dataset_3d_rmf():
    shape = (1, 3, 25, 25)
    exposure = torch.ones(shape)
    psf = np.stack([Gaussian2DKernel(_, x_size=25) for _ in [1, 2, 3]]).astype(
        np.float32
    )
    background = torch.zeros(shape)
    rmf = torch.ones((3, 1)) / 3.0
    return {
        "psf": torch.from_numpy(psf[None]),
        "exposure": exposure,
        "background": background,
        "rmf": rmf,
    }


def test_simple_npred_model(dataset):
    flux_init = torch.zeros(dataset["exposure"].shape)
    flux_init[0, 0, 10, 10] = 1

    flux = FluxComponent(flux_init=flux_init)
    npred_model = NPredModel(components={"flux": flux})

    npred = npred_model(**dataset)

    npred = npred.detach().numpy()[0, 0]
    assert_allclose(npred[10, 10], 1.017218, rtol=1e-5)
    assert_allclose(npred.sum(), 513.039549, rtol=1e-5)


def test_simple_npred_model_3d(dataset_3d):
    flux_init = torch.zeros(dataset_3d["exposure"].shape)
    flux_init[0, :, 12, 12] = 1

    flux = FluxComponent(flux_init=flux_init)
    npred_model = NPredModel(components={"flux": flux})

    npred = npred_model(**dataset_3d)

    npred = npred.detach().numpy()[0]
    assert npred.shape == (3, 25, 25)
    assert_allclose(npred[0, 10, 10], 1.002915, rtol=1e-5)
    assert_allclose(npred.sum(), 1653.52562, rtol=1e-5)


def test_simple_npred_model_3d_rmf(dataset_3d_rmf):
    flux_init = torch.zeros(dataset_3d_rmf["exposure"].shape)
    flux_init[0, :, 12, 12] = 1

    flux = FluxComponent(flux_init=flux_init)
    npred_model = NPredModel(components={"flux": flux})

    npred = npred_model(**dataset_3d_rmf)

    npred = npred.detach().numpy()[0]
    assert npred.shape == (1, 25, 25)
    assert_allclose(npred[0, 10, 10], 0.00963, rtol=1e-4)
    assert_allclose(npred.sum(), 1, rtol=2e-5)
