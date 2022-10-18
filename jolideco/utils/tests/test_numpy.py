import numpy as np
from numpy.testing import assert_allclose
from jolideco.utils.numpy import (
    reconstruct_from_overlapping_patches,
    view_as_overlapping_patches,
)


def test_reconstruct_from_overlapping_patches():
    patch_shape, stride = (8, 8), 4
    image = np.ones((64, 64))

    patches = view_as_overlapping_patches(image=image, shape=patch_shape, stride=stride)
    patches = patches.reshape((-1, 8, 8))

    reco = reconstruct_from_overlapping_patches(
        patches=patches, image_shape=image.shape, stride=stride
    )

    width = patch_shape[0] // 2
    assert_allclose(reco[width:-width, width:-width], 1)
