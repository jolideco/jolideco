def view_as_overlapping_patches(image, shape):
    """View as overlapping patches"""
    step = shape[0] // 2
    patches = view_as_windows(image, shape, step=step)
    ncols = shape[0] * shape[1]
    return patches.reshape(-1, ncols)
