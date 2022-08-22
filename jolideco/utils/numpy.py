import numpy as np
from itertools import product


__all__ = [
    "view_as_overlapping_patches",
    "is_power2",
    "get_pixel_weights",
    "reconstruct_from_overlapping_patches",
]


def is_power2(n):
    """Check if an integer is a power of 2."""
    return (n > 0) & ((n & (n - 1)) == 0)


def evaluate_trapez(x, width, slope):
    """One dimensional Trapezoid model function"""
    # Compute the four points where the trapezoid changes slope
    x2 = min(-width / 2.0, 0)
    x3 = max(width / 2.0, 0)
    x1 = x2 - 1.0 / slope
    x4 = x3 + 1.0 / slope

    # Compute model values in pieces between the change points
    range_a = np.logical_and(x >= x1, x < x2)
    range_b = np.logical_and(x >= x2, x < x3)
    range_c = np.logical_and(x >= x3, x < x4)
    val_a = slope * (x - x1)
    val_c = slope * (x4 - x)
    return np.select([range_a, range_b, range_c], [val_a, 1, val_c])


def get_pixel_weights(patch_shape, stride):
    """Compute pixel weights for overlapping patches

    Parameters
    ----------
    patch_shape : tuple of int
        Patch shape
    stride : int
        Stride of the patches.

    Returns
    -------
    weights : `~numpy.ndarray`
        Weights array
    """
    width = np.max(patch_shape)
    overlap = width - stride

    value = (width - 1.0) / 2

    x = np.linspace(-value, value, width)

    values = evaluate_trapez(x=x, width=(stride - overlap), slope=1.0 / overlap)
    weights = values * values[:, np.newaxis]
    weights = weights / weights.sum() * stride**2
    return weights


def view_as_overlapping_patches(image, shape, stride=None):
    """View as overlapping patches

    Parameters
    ----------
    image : `~numpy.ndarray`
        Image tensor
    shape : tuple of int
        Shape of the patches.

    Returns
    -------
    patches : `~numpy.ndarray`
        Array of overlapping patches of shape
        (n_patches, patch_shape_flat)

    """
    from skimage.util import view_as_windows

    if stride is None:
        stride = shape[0] // 2

    patches = view_as_windows(image, shape, step=stride)
    ncols = shape[0] * shape[1]
    return patches.reshape(-1, ncols)


def reconstruct_from_overlapping_patches(patches, image_shape, stride=None):
    """Reconstruct an image from overlapping patches.

    Parameters
    ----------
    patches : `~numpy.ndarray`
        Array of overlapping patches of shape
        (n_patches, patch_shape_y, patch_shape_x)
    image_shape : tuple of int
        Image shape
    stride : int
        Stride of the patches. By default it is half of the patch size.

    Returns
    -------
    image : `~numpy.ndarray`
        Image array

    """
    if stride is None:
        stride = patches.shape[-1] // 2

    image_height, image_width = image_shape
    patch_height, patch_width = patches.shape[1:]
    image = np.zeros(image_shape)

    # compute the dimensions of the patches array
    n_h = image_height - patch_height + 1
    n_w = image_width - patch_width + 1

    weights = get_pixel_weights(patch_shape=patches.shape[1:], stride=stride)

    for patch, (i, j) in zip(
        patches, product(range(0, n_h, stride), range(0, n_w, stride))
    ):
        image[i : i + patch_height, j : j + patch_width] += (  # noqa E203 and W503
            weights * patch
        )

    return image
