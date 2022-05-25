import matplotlib.pyplot as plt

__all__ = ["plot_trace_loss"]


def plot_trace_loss(ax, trace_loss, **kwargs):
    """Plot trace loss

    Parameters
    ----------
    ax : `~matplotlib.pyplot.Axes`
        Plotting axes
    trace_loss : `~astropy.table.Table`
        Parameter trace table
    """
    for name in trace_loss.colnames:
        ax.plot(trace_loss[name], label=name, **kwargs)

    ax.semilogx()
    ax.set_xlabel("# Iteration")
    ax.set_ylabel("Loss value")
    plt.legend()
