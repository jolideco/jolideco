
__all__ = ["view_as_overlapping_patches"]


def view_as_overlapping_patches(image, shape):
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

    step = shape[0] // 2
    patches = view_as_windows(image, shape, step=step)
    ncols = shape[0] * shape[1]
    return patches.reshape(-1, ncols)
