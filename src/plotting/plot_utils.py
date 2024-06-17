import numpy as np
import matplotlib as mpl

from utils.stats_utils import norm, norm_err, norm_err_poisson, centers


def add_distribution_plot(
    counts,
    edges,
    ax,
    label="",
    color="black",
    alpha=0.5,
    fill=False,
    normalize=True,
    label_count=True,
    errorbar=True,
):
    # Extract counts and edges from distribution
    c_norm = norm(counts) if normalize else counts
    c_norm_err = norm_err_poisson(counts) if normalize else np.sqrt(counts)

    # Add number of images to label
    if label is not None and label_count:
        label += f" (n={int(np.sum(counts)):_})"

    # Check if lines or collections are already present in axes
    # (relevant for x limits)
    data_present = len(ax.lines) + len(ax.collections) > 0

    # Plot distribution and error bars
    ax.stairs(c_norm, edges, alpha=alpha, fill=fill, label=label, color=color, lw=1.5)
    if errorbar:
        ax.errorbar(
            centers(edges),
            c_norm,
            yerr=c_norm_err,
            alpha=0.75 * alpha,
            color=color,
            ls="none",
            elinewidth=0.5,
            capsize=2.5,
            capthick=0.5,
        )

    # Set x limits
    _, x_max = ax.get_xlim()
    fullbins = edges[1:][counts > 0]
    new_xmax = (max(fullbins) if len(fullbins) else edges[-1]) * 1.02
    if data_present:
        new_xmax = max(new_xmax, x_max)
    ax.set_xlim(edges[0] - 0.02 * new_xmax, new_xmax)
    return counts, edges


def ghost_axis(ax):
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Hide all ticks and labels
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.labelpad = 50


def plot_collection(
    distr_dict_list, plot_fn, labels=None, colors=None, cmap="cividis", **plot_kwargs
):
    # Set colors if not passed
    if colors is None:
        colors = mpl.colormaps[cmap](np.linspace(0, 1, len(distr_dict_list)))
    elif len(colors) < len(distr_dict_list):
        colors = [
            *colors,
            *mpl.colormaps[cmap](np.linspace(0, 1, len(distr_dict_list) - len(colors))),
        ]

    fig_axs = None
    for i, distr_dict in enumerate(distr_dict_list):
        fig_axs = plot_fn(
            distr_dict,
            fig_ax=fig_axs,
            color=colors[i],
            label=labels[i] if labels is not None else None,
            **plot_kwargs,
        )

    return fig_axs


def add_distribution_delta_plot(delta, ax2, key, color="tab:blue"):
    # Extract delta and edges from distribution
    delta, delta_err, edges = delta[key]

    # Plot delta and error bars
    ax2.stairs(delta, edges, fill=True, alpha=0.5, color=color)
    ax2.errorbar(
        centers(edges),
        delta,
        yerr=delta_err,
        alpha=0.5,
        color=color,
        ls="none",
        elinewidth=0.5,
        capsize=2.5,
        capthick=0.5,
    )
