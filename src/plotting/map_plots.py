import matplotlib.pyplot as plt


def point_process_scatter_plot(points, fig_ax=None, c="red", s=10, **kwargs):
    if fig_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), tight_layout=True)
    else:
        fig, ax = fig_ax
    ax.scatter(points[:, 0], points[:, 1], s=s, c=c, **kwargs)
    ax.grid(alpha=0.3)
    return fig, ax
