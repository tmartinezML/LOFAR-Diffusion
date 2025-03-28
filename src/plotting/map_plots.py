import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from astropy.wcs import WCS

import utils.paths as paths
import maps.map_utils as mputil
from data.transforms import minmax_scale


def plot_source_cutouts(
    src_img,
    title=None,
    mask=None,
    model_cutout=None,
    sim_cutout=None,
    fig_ax=None,
):

    if fig_ax is None:
        fig, axs = plt.subplots(
            1, 3, figsize=(10, 4), sharex=True, sharey=True, constrained_layout=True
        )
    else:
        fig, axs = fig_ax

    plot_sky_map(src_img, fig_ax=(fig, axs[0]), norm_quantile=None, cbar=False)
    axs[0].set_title("Source")

    if model_cutout is not None:
        norm = Normalize(vmin=0, vmax=src_img.max())

        plot_sky_map(
            model_cutout,
            fig_ax=(fig, axs[1]),
            norm_quantile=None,
            cbar=False,
            norm=norm,
        )
        axs[1].set_title("Sky Model")

    if sim_cutout is not None:
        plot_sky_map(sim_cutout, fig_ax=(fig, axs[2]), norm_quantile=None, cbar=False)
        axs[2].set_title("Simulated Obs.")

    if mask is not None:
        for ax in axs[:-1]:
            ax.contour(mask, levels=[0.5], colors="red", alpha=0.2, linewidths=1)

    for ax in axs:
        ax.axis("off")

    fig.suptitle(title, fontsize=21)

    return fig, axs


def point_process_scatter_plot(points, fig_ax=None, c="red", s=10, **kwargs):
    if fig_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), tight_layout=True)
    else:
        fig, ax = fig_ax
    ax.scatter(points[:, 0], points[:, 1], s=s, c=c, **kwargs)
    ax.grid(alpha=0.3)
    return fig, ax


def plot_sky_map(
    map_inp,
    scale_fn=lambda x: x,
    wcs=None,
    fig_ax=None,
    cbar=True,
    cbar_label="Flux Density (Jy/beam)",
    norm=None,
    vmin=None,
    vmax=None,
    norm_quantile=0.995,
    size=(9, 9),
    **imshow_kwargs,
):
    # Get map array
    match map_inp:
        # Map array
        case np.ndarray():
            map_array = map_inp

        # Anything else: Load map (invalid input handled within get_image)
        case _:
            map_array, wcs_in = mputil.get_image(map_inp)
            wcs = wcs or wcs_in

    # Scale map
    scaled_map = scale_fn(map_array)

    # Set up color norm
    if norm_quantile is not None and norm is None:
        norm = Normalize(
            vmin=np.quantile(scaled_map[scaled_map < 0], norm_quantile),
            vmax=np.quantile(scaled_map[scaled_map > 0], norm_quantile),
            clip=True,
        )

    elif norm is None:
        norm = Normalize(
            vmin=vmin or np.nanmin(scaled_map), vmax=vmax or np.nanmax(scaled_map)
        )

    # Plot map
    fig, ax = fig_ax or plt.subplots(
        figsize=size,
        subplot_kw={"projection": wcs},
        constrained_layout=True,
    )

    im = ax.imshow(scaled_map.squeeze(), origin="lower", norm=norm, **imshow_kwargs)

    if cbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05, label=cbar_label)

    if wcs is None and fig_ax is None:
        ax.axis("off")
    elif wcs is not None:
        ax.grid(alpha=0.1)
        ax.set_xlabel("RA")
        ax.set_ylabel("Dec")

    return fig, ax, im


def double_map_plot(
    map1,
    map2,
    minmax=False,
    scale_fn=lambda x: x,
    norm_quantile=0.995,
    wcs=None,
    size=(14, 7),
    fig_ax=None,
    norm=(None, None),
    cbar_label="Flux Density (Jy/beam)",
    kw_map1={},
    kw_map2={},
):
    single_cbar = False

    match norm:
        case (norm1, norm2):
            norm1, norm2 = norm

        case None:
            norm1, norm2 = None, None

        case Normalize():
            print("Using single norm for both maps")
            norm1 = norm2 = norm
            single_cbar = True

    if minmax:
        single_cbar = True
        map1 = minmax_scale(map1)
        map2 = minmax_scale(map2)
        norm = Normalize(vmin=0, vmax=1)
        norm1, norm2 = norm, norm

    elif (
        norm_quantile is not None
        and not single_cbar
        and ((norm1, norm2) == (None, None))
    ):
        single_cbar = False

        def get_vmin(map):
            negative_flag = map < 0
            if negative_flag.sum() == 0:
                return 0
            return np.quantile(map[negative_flag], norm_quantile)

        norm1 = Normalize(
            vmin=get_vmin(map1),
            vmax=np.quantile(map1[map1 > 0], norm_quantile),
            clip=True,
        )
        norm2 = Normalize(
            vmin=get_vmin(map2),
            vmax=np.quantile(map2[map2 > 0], norm_quantile),
            clip=True,
        )

    fig, axs = fig_ax or plt.subplots(
        1,
        2,
        figsize=size,
        gridspec_kw={"hspace": 0.01, "wspace": (0.05 if single_cbar else 0.15)},
        sharex=True,
        sharey=True,
        # constrained_layout=True,
    )

    match wcs:
        case None:
            wcs1, wcs2 = None, None
        case (WCS(), WCS()):
            wcs1, wcs2 = wcs
        case WCS():
            wcs1, wcs2 = wcs, wcs
        case _:
            raise ValueError(f"Unknown WCS input: {type(wcs)}")

    if wcs is not None:
        axs[0].remove()  # Remove the existing subplot
        axs[0] = fig.add_subplot(1, 2, 1, projection=wcs1, sharex=axs[1], sharey=axs[1])
        axs[0].coords[1].set_ticklabel(rotation="vertical")

        axs[1].remove()  # Remove the existing subplot
        axs[1] = fig.add_subplot(1, 2, 2, projection=wcs2, sharex=axs[0], sharey=axs[0])
        axs[1].coords[1].set_auto_axislabel(False)
        axs[1].coords[1].set_ticklabel(rotation="vertical")

        for ax in axs:
            ax.grid(alpha=0.05, ls="--")
            ax.set_xlabel("RA")

        axs[0].set_ylabel("Dec")

        if single_cbar:
            axs[1].coords[1].set_ticklabel_position("r")

    match cbar_label:
        case str() | None:
            cbar_label1 = cbar_label2 = cbar_label

        case (str(), str()):
            cbar_label1, cbar_label2 = cbar_label

        case _:
            raise ValueError(f"Unknown cbar_label input: {cbar_label}")

    _, _, im1 = plot_sky_map(
        map1,
        scale_fn=scale_fn,
        fig_ax=(fig, axs[0]),
        cbar=not single_cbar,
        cbar_label=cbar_label1,
        norm=norm1,
        norm_quantile=norm_quantile,
        **kw_map1,
    )
    _, _, im2 = plot_sky_map(
        map2,
        scale_fn=scale_fn,
        fig_ax=(fig, axs[1]),
        cbar=True,  # not single_cbar,
        cbar_label=cbar_label2,
        norm=norm2,
        norm_quantile=norm_quantile,
        **kw_map2,
    )
    axs[1].set_ylabel("")

    if False:  # single_cbar:
        # Add new subplot axis for colorbar
        fig.subplots_adjust(right=0.85)
        pos = axs[1].get_position()
        cax = fig.add_axes(
            [pos.x1 + 0.04, pos.y0, 0.03, pos.height]
        )  # Position for the colorbar
        fig.colorbar(im1, cax=cax)  # , fraction=0.046, pad=0.05)

    return fig, axs
