from pathlib import Path
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import colormaps as cm

import analysis.bdsf_evaluation as bdsfeval
import analysis.model_evaluation as meval
import utils.paths as paths
from utils.stats_utils import norm
from plotting.plot_utils import add_distribution_plot, plot_collection

# Plot the following metrics separately and save each of them to pdf:
# Pixel Value distribution, Image Mean, Image Sigma, COM Contours,
# Total flux gaus, nsrc, q0.1_area

mm = 1 / 25.4  # mm in inches
# plt.rcParams.update({"font.size": 5})
plt.style.use("seaborn-paper")
plt.rcParams.update(
    {
        "figure.constrained_layout.use": True,
        "figure.dpi": 300,
        "font.family": "Nimbus Roman",
        "font.size": 4.5,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Roman",
        "mathtext.it": "Nimbus Roman:italic",
        "mathtext.bf": "Nimbus Roman:bold",
        "mathtext.cal": "Nimbus Roman:italic",
        "text.usetex": False,
    }
)


def COM_contour_plot(COM, fig_ax=None, color="black", label=None):

    if fig_ax is None:
        # Initialize figure
        fig, ax = plt.subplots(1, 1, figsize=(88 * mm,) * 2)
        ax.set_aspect("equal")
        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)
        ax.invert_yaxis()
        ax.set_xlabel("COM x")
        ax.set_ylabel("COM y")

    else:
        fig, ax = fig_ax

    X = np.arange(-40, 40) + 0.5  # Should be centered on pixels
    counts_norm = norm(COM)
    cs = ax.contour(
        X,
        X,
        counts_norm,
        levels=[1e-4, 1e-3, 1e-2],
        colors=color,
        linewidths=0.5,
    )
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.e")

    ax._children.append(mlines.Line2D([0], [0], color=color))
    ax._children[-1].set_label(label)
    ax.legend()

    return fig, ax


def single_metric_plot(
    distr_dict,
    key,
    fig_ax=None,
    label=None,
    xlabel=None,
    color="black",
    alpha=0.8,
):

    # Get data
    counts, edges = distr_dict[key]

    if "area" in key:
        counts, edges = counts[:-1], edges[:-1]

    if key == "COM":
        return COM_contour_plot(counts, fig_ax=fig_ax, color=color, label=label)

    if fig_ax is None:
        # Initialize figure
        fig, ax = plt.subplots(
            1, 1, dpi=150, tight_layout=True, figsize=(88 * mm, 88 * 2 / 3 * mm)
        )
    else:
        fig, ax = fig_ax

    add_distribution_plot(
        counts,
        edges,
        ax,
        label=label,
        color=color,
        alpha=alpha,
    )

    # Axis and plot properties
    ax.set_ylabel("Relative Frequency")
    ax.set_xlabel(xlabel if xlabel is not None else key.replace("_", " "))

    if label is not None:
        ax.legend()
    ax.set_yscale("log")

    return fig, ax


def paper_plots_bdsf(
    img_dir,
    labels=None,
    cmap="cividis",
    colors=None,
    plot_train=True,
    force_train=False,
    train_path=paths.LOFAR_SUBSETS["0-clip"],
    train_label="Train Data",
    bdsf_dir=None,
    h5_kwargs={},
    **distribution_kwargs,
):

    match img_dir:
        case Path():
            # Set output path
            fig_name = img_dir.stem
            img_dir = [img_dir]
            bdsf_dir = [bdsf_dir]
            if labels is None:
                labels = ["Generated"]

        case list():
            if labels is None:
                labels = [d.stem for d in img_dir]

    metrics = ["total_flux_gaus", "nsrc", "q0.1_area"]
    xlabels = ["Model Flux (a.u.)", "Identified Sources", "$A_\mathrm{10\%}$ (px)"]
    gen_distr_dict_list = [
        bdsfeval.get_bdsf_distributions(d, out_dir=p) for d, p in zip(img_dir, bdsf_dir)
    ]
    # Get metrics dict
    if train_path is not None:
        lofar_distr_dict = bdsfeval.get_bdsf_distributions(
            train_path, force=force_train
        )

    # Set colors for paper plots
    colors = colors or [cm["viridis"](i) for i in [0.2, 0.85]]

    out = {}
    for metric, xlabel in zip(metrics, xlabels):
        collection = (
            [lofar_distr_dict, *gen_distr_dict_list]
            if plot_train
            else gen_distr_dict_list
        )
        fig, axs = plot_collection(
            collection,
            single_metric_plot,
            colors=colors or (["darkviolet"] if plot_train else None),
            labels=[train_label, *labels] if plot_train else labels,
            cmap=cmap,
            key=metric,
            xlabel=xlabel,
        )
        fig.savefig(paths.PAPER_PLOT_DIR / f"{metric}.pdf", bbox_inches="tight")

        out[metric] = (fig, axs)
    return out


def paper_plots_pxstats(
    img_dir,
    labels=None,
    cmap="cividis",
    colors=None,
    plot_train=True,
    force_train=False,
    train_path=paths.LOFAR_SUBSETS["0-clip"],
    train_label="Train Data",
    h5_kwargs={},
    **distribution_kwargs,
):

    match img_dir:
        case Path():
            # Set output path
            fig_name = img_dir.stem
            img_dir = [img_dir]
            if labels is None:
                labels = ["Generated"]

        case list():
            if labels is None:
                labels = [d.stem for d in img_dir]

    metrics = ["Pixel_Intensity", "Image_Mean", "Image_Sigma", "COM"]
    xlabels = ["Pixel Values", "Image Mean", "Image Sigma", "COM"]

    gen_distr_dict_list = [meval.get_distributions(d) for d in img_dir]
    # Get metrics dict
    if train_path is not None:
        lofar_distr_dict = meval.get_distributions(train_path, force=force_train)

    # Set colors for paper plots
    colors = colors or [cm["viridis"](i) for i in [0.2, 0.85]]

    out = {}
    for metric, xlabel in zip(metrics, xlabels):
        collection = (
            [lofar_distr_dict, *gen_distr_dict_list]
            if plot_train
            else gen_distr_dict_list
        )
        fig, axs = plot_collection(
            collection,
            single_metric_plot,
            colors=colors or (["darkviolet"] if plot_train else None),
            labels=[train_label, *labels] if plot_train else labels,
            cmap=cmap,
            key=metric,
            xlabel=xlabel,
        )
        fig.savefig(paths.PAPER_PLOT_DIR / f"{metric}.pdf", bbox_inches="tight")

        out[metric] = (fig, axs)
    return out
