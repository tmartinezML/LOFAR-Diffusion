import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from astropy.wcs import WCS

from data.transforms import minmax_scale


def point_process_scatter_plot(points, fig_ax=None, c="red", s=10, **kwargs):
    if fig_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), tight_layout=True)
    else:
        fig, ax = fig_ax
    ax.scatter(points[:, 0], points[:, 1], s=s, c=c, **kwargs)
    ax.grid(alpha=0.3)
    return fig, ax


def plot_sky_map(
    map_array,
    scale_fn=lambda x: np.tanh(7.5 * x),
    wcs=None,
    fig_ax=None,
    cbar=True,
    norm=None,
    size=(9, 9),
):
    # Scale map
    scaled_map = scale_fn(map_array)

    # Plot map
    fig, ax = fig_ax or plt.subplots(
        figsize=size, subplot_kw={"projection": wcs}, constrained_layout=True
    )

    im = ax.imshow(scaled_map.squeeze(), origin="lower", norm=norm)

    if cbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)

    if wcs is not None:
        ax.axis("off")
    else:
        ax.grid(alpha=0.1)
        ax.set_xlabel("RA")
        ax.set_ylabel("Dec")

    return fig, ax, im


def double_map_plot(
    map1, map2, minmax=True, scale_fn=lambda x: np.tanh(7.5 * x), wcs=None, size=(8, 4)
):

    fig, axs = plt.subplots(
        1,
        2,
        figsize=size,
        gridspec_kw={"hspace": 0.01, "wspace": (0.03 if minmax else 0.15)},
        sharex=True,
        sharey=True,
        #constrained_layout=True,
    )

    match wcs:
        case None:
            wcs1, wcs2 = None, None
        case (WCS(), WCS()):
            wcs1, wcs2 = wcs
        case WCS():
            wcs1, wcs2 = wcs, wcs

    if wcs is not None:
        axs[0].remove()  # Remove the existing subplot
        axs[0] = fig.add_subplot(1, 2, 1, projection=wcs1, sharex=axs[1], sharey=axs[1])
        axs[0].coords[1].set_ticklabel(rotation="vertical")

        axs[1].remove()  # Remove the existing subplot
        axs[1] = fig.add_subplot(1, 2, 2, projection=wcs2, sharex=axs[0], sharey=axs[0])
        axs[1].coords[1].set_auto_axislabel(False)
        axs[1].coords[1].set_ticklabel(rotation="vertical")
        if minmax:
            axs[1].coords[1].set_ticklabel_position("r")

    norm = None
    if minmax:
        map1 = minmax_scale(map1)
        map2 = minmax_scale(map2)
        norm = Normalize(vmin=0, vmax=1)
        # axs[2].axis("off")

    _, _, im1 = plot_sky_map(
        map1,
        scale_fn=scale_fn,
        fig_ax=(fig, axs[0]),
        cbar=not minmax,
        norm=norm,
    )
    _, _, im2 = plot_sky_map(
        map2,
        scale_fn=scale_fn,
        fig_ax=(fig, axs[1]),
        cbar=not minmax,
        norm=norm,
    )
    axs[1].set_ylabel("")

    if minmax:
        # Add new subplot axis for colorbar
        fig.subplots_adjust(right=0.85)
        pos = axs[1].get_position()
        cax = fig.add_axes(
            [pos.x1 + 0.04, pos.y0, 0.03, pos.height]
        )  # Position for the colorbar
        fig.colorbar(im1, cax=cax)  # , fraction=0.046, pad=0.05)

    return fig, axs
