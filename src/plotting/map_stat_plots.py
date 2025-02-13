import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from scipy.stats import binned_statistic_2d

import plotting.plot_utils as plot_utils


def stat2d_plot(
    x,
    y,
    z,
    nbins=100,
    xlim=(None, None),
    ylim=(None, None),
    zlim=(None, None),
    log=(True, True, False),
    ax_labels=("", "", ""),
    pcolor_kw={},
    hist2d_kw={},
):
    fig, axs = plt.subplots(
        1, 2, figsize=(12, 5), constrained_layout=True, sharex=True, sharey=True
    )

    # Set log parameter:
    match log:
        case bool():
            xlog = ylog = zlog = log
        case (bool(), bool()):
            xlog, ylog = log[0]
            zlog = log[1]
        case (bool(), bool(), bool()):
            xlog, ylog, zlog = log
        case _:
            raise ValueError(f"Invalid log input: {log}")

    # Plot 2D histogram on the right
    _, _, (xbins, ybins) = hist2D_plot(
        x,
        y,
        nbins=nbins,
        xlim=xlim,
        ylim=ylim,
        normalize=False,
        log=(xlog, ylog),
        fig_ax=(fig, axs[1]),
        id_line=False,
        **hist2d_kw,
    )

    # Get binned statistic
    stat = binned_statistic_2d(
        x=x,
        y=y,
        values=z,
        statistic="mean",
        bins=(xbins, ybins),
    )

    # Parse z limit (i.e. vmin and vmax)
    match zlim:
        case tuple():
            vmin, vmax = zlim
        case int() | float() | None:
            vmax = zlim
            vmin = -vmax if vmax is not None else None

    # Set color normalization
    clrNorm = LogNorm if zlog else Normalize
    clrNorm = clrNorm(vmin=vmin, vmax=vmax)

    # Plot 2D statistic on the left
    p = axs[0].pcolor(
        xbins,
        ybins,
        stat.statistic.T,
        norm=clrNorm,
        cmap="bwr" if (vmin is not None) and (vmin == -vmax) else "viridis",
        **pcolor_kw,
    )
    fig.colorbar(
        p,
        ax=axs[0],
        label=f"Mean {lbl}" if len((lbl := ax_labels[2])) else "",
    )

    # Remove all axis labels
    for ax in axs:
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Set sup axis labels
    fig.supxlabel(ax_labels[0])
    fig.supylabel(ax_labels[1])

    return fig, axs, (xbins, ybins)


def hist2D_plot(
    x,
    y,
    nbins=100,
    xlim=(None, None),
    ylim=(None, None),
    normalize=False,
    log=True,
    ax_labels=("", ""),
    log_color=False,
    id_line=True,
    pcolor_kw={},
    fig_ax=None,
):
    # Get number of bins from input
    match nbins:
        case int():
            nbinsx = nbinsy = nbins

        case (int(), int()):
            nbinsx, nbinsy = nbins

        case _:
            raise ValueError(f"Invalid nbins input: {nbins}")

    # Get plot limits from input or, if not provided, from data.
    xmin, xmax = [l or xl for l, xl in zip(xlim, (x.min(), x.max()))]
    ymin, ymax = [l or yl for l, yl in zip(ylim, (y.min(), y.max()))]

    # Get boolean flag for log scale
    match log:
        case bool():
            xlog = ylog = log
        case (bool(), bool()):
            xlog, ylog = log
        case _:
            raise ValueError(f"Invalid log input: {log}")

    # Get bins for 2D histogram. Limits are ther same as plot limits.
    bin_fn = lambda islog: np.geomspace if islog else np.linspace
    xbins = bin_fn(xlog)(xmin, xmax, nbinsx)
    ybins = bin_fn(ylog)(ymin, ymax, nbinsy)

    H, _, _ = np.histogram2d(
        x,
        y,
        bins=(xbins, ybins),
    )

    # Normalize if desired
    match normalize:
        case False:
            pass
        case True:
            print("Total Norm.")
            H = H / H.sum()
        case "x":
            H = H / H.sum(axis=0)
        case "y":
            H = H / H.sum(axis=1)[:, np.newaxis]
        case _:
            raise ValueError(f"Invalid normalize input: {normalize}")

    fig, ax = fig_ax or plt.subplots(1, 1, figsize=(10, 8), constrained_layout=True)

    p = ax.pcolor(
        xbins,
        ybins,
        H.T,
        cmap="viridis",
        norm=LogNorm() if log_color else None,
        **pcolor_kw,
    )
    fig.colorbar(p, ax=ax, label="Counts" if not normalize else "Norm. Counts")

    # Plot 1:1 line
    if id_line:
        mn = min(xmin, ymin)
        mx = max(xmax, ymax)
        ax.axline([mn, mn], [mx, mx], color="red", linestyle="--", alpha=0.25)

    # Set axis limits in case 1:1 line is outside
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)

    # Set axis labels
    ax.set_xlabel(ax_labels[0])
    ax.set_ylabel(ax_labels[1])

    # Set log scale if desired
    ax.set_xscale("log" if xlog else "linear")
    ax.set_yscale("log" if ylog else "linear")

    ax.grid(alpha=0.3, ls=":")
    return fig, ax, (xbins, ybins)


def scat_hist_plot(
    x,
    y,
    bins=None,
    xlim=(None, None),
    ylim=(None, None),
    plt_label=None,
    ax_labels=("", ""),
    scatter_kw={},
):

    # Get bins from input
    match bins:
        case None:
            xbins = plot_utils.auto_log_bins((x, y), num=100)
            ybins = xbins

        case int() | np.array():
            xbins = bins
            ybins = bins

        case tuple():
            xbins, ybins = bins

        case _:
            raise ValueError(f"Invalid bins input: {bins}")

    fig, axs = plt.subplots(
        2,
        2,
        figsize=(9, 9),
        constrained_layout=True,
        width_ratios=[1, 0.6],
        height_ratios=[0.6, 1],
        sharex="col",
        sharey="row",
    )

    hist_kw = dict(histtype="step", label=plt_label)

    ax = axs[0][0]
    ax.hist(x, log=True, bins=xbins, **hist_kw)

    ax = axs[0][1]
    # remove axis
    ax.axis("off")

    ax = axs[1][1]
    hist_kw["orientation"] = "horizontal"

    c, _, _ = ax.hist(y, log=False, bins=ybins, **hist_kw)
    ax.set_xscale("log")
    # For xmax, round to next decimal of current order of magnitude
    xmax = 10 ** (np.log10(c.max()) * 1.05)
    ax.set_xlim(left=7e-1, right=xmax)

    ax = axs[1][0]
    default_kw = dict(s=0.05, alpha=0.8, marker=".", label=plt_label)
    default_kw.update(scatter_kw)
    ax.scatter(
        x,
        y,
        **default_kw,
    )
    # Plot 1:1 line
    xmin, xmax = [x or axx for x, axx in zip(xlim, ax.get_xlim())]
    ymin, ymax = [y or axy for y, axy in zip(ylim, ax.get_ylim())]
    mn = min(xmin, ymin)
    mx = max(xmax, ymax)
    ax.axline([mn, mn], [mx, mx], color="black", linestyle="--", alpha=0.25)
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(ax_labels[0])
    ax.set_ylabel(ax_labels[1])

    for ax in axs.flatten():
        ax.grid(alpha=0.3)
        # Plot legend only if ax has artists with labels
        if ax.get_legend_handles_labels()[0]:
            ax.legend()

    return fig, axs, (xbins, ybins)
