from itertools import zip_longest

import matplotlib.pyplot as plt

__all__ = ["plot_trace_loss", "plot_example_dataset", "add_cbar"]


def add_cbar(im, ax, fig):
    """Add cbar to a given axis and figure"""
    bbox = ax.get_position()
    loright = bbox.corners()[-2]
    rect = [loright[0] + 0.02, loright[1], 0.02, bbox.height]
    cax = fig.add_axes(rect)
    return fig.colorbar(im, cax=cax, orientation="vertical")


def plot_trace_loss(ax, trace_loss, which=None, **kwargs):
    """Plot trace loss

    Parameters
    ----------
    ax : `~matplotlib.pyplot.Axes`
        Plotting axes
    trace_loss : `~astropy.table.Table`
        Parameter trace table
    which : list of str
        Which traces to plot.

    """
    if which is None:
        which = trace_loss.colnames

    for name in which:
        ax.plot(trace_loss[name], label=name, **kwargs)

    ax.semilogx()
    ax.semilogy()
    ax.set_xlabel("# Iteration")
    ax.set_ylabel("Loss value")
    ax.legend()


def plot_example_dataset(data, figsize=(12, 7), **kwargs):
    """Plot example dataset

    Parameters
    ----------
    data : dict of `~numpy.ndarray`
        Data
    figsize : tuple
        Figure size
    **kwargs : dict
        Keyword arguments passed to `~matplotlib.pyplot.imshow`
    """
    data = data.copy()

    wcs = data.pop("wcs", None)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=figsize,
        subplot_kw={"projection": wcs},
    )

    for name, ax in zip_longest(data.keys(), axes.flat):
        if name is None:
            ax.set_visible(False)
            continue

        im = ax.imshow(data[name], origin="lower", **kwargs)
        ax.set_title(name.title())
        fig.colorbar(im, ax=ax)
