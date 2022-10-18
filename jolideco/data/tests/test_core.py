import pytest
import numpy as np
from numpy.testing import assert_allclose
from jolideco.data import (
    disk_source_gauss_psf,
    gauss_and_point_sources_gauss_psf,
    point_source_gauss_psf,
)


@pytest.fixture()
def random_state():
    return np.random.RandomState(836)


def test_data_point_source_gauss_psf(random_state):
    data = point_source_gauss_psf(random_state=random_state)

    assert_allclose(data["counts"][0][0], 2)
    assert_allclose(data["exposure"][0][0], 1)
    assert_allclose(data["background"][0][0], 2.0)
    assert_allclose(data["psf"][0][0], 1.44298e-05, rtol=1e-5)
    assert_allclose(data["flux"][16][16], 1000, rtol=1e-5)


def test_data_disk_source_gauss_psf(random_state):
    data = disk_source_gauss_psf(random_state=random_state)

    assert_allclose(data["counts"][0][0], 1)
    assert_allclose(data["exposure"][0][0], 0.5)
    assert_allclose(data["background"][0][0], 2.0)
    assert_allclose(data["psf"][0][0], 1.44298e-05, rtol=1e-5)
    assert_allclose(data["flux"][16][16], 35.367765, rtol=1e-5)


def test_data_gauss_and_point_sources_gauss_psf(random_state):
    data = gauss_and_point_sources_gauss_psf(random_state=random_state)

    assert_allclose(data["counts"][0][0], 1)
    assert_allclose(data["exposure"][0][0], 0.5)
    assert_allclose(data["background"][0][0], 2.0)
    assert_allclose(data["psf"][0][0], 4.477632e-09, rtol=1e-5)
    assert_allclose(data["flux"][16][16], 36.664897, rtol=1e-5)
