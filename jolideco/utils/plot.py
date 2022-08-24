import matplotlib.pyplot as plt

__all__ = ["plot_trace_loss"]


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
    plt.legend()
