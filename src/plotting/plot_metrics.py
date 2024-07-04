import numpy as np
from itertools import compress
from plotting.plot_utils import add_distribution_delta_plot, add_distribution_plot, ghost_axis, plot_collection

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec

from analysis.stats_utils import norm


def init_pixel_metrics_plot():
    # Init figure
    fig = plt.figure(figsize=(16, 16), dpi=150, tight_layout=True)

    # Set gridspec
    gs_outer = gridspec.GridSpec(3, 2, figure=fig)
    for i in range(3):
        fig.add_subplot(gs_outer[i, 0])
        ghost_axis(fig.add_subplot(gs_outer[i, 1]))
        gs_outer[i, 1].subgridspec(2, 2, hspace=0.01, wspace=0.01).subplots()

        # Remove ticks that overlap
        fig.axes[2 + 6 * i].tick_params(
            axis='x', which='both', bottom=False, top=True,
            labelbottom=False, labeltop=True,
        )
        fig.axes[3 + 6 * i].tick_params(
            which='both', bottom=False, top=True, right=True, left=False,
            labelbottom=False, labeltop=True, labelright=True, labelleft=False,
        )
        fig.axes[5 + 6 * i].tick_params(
            axis='y', which='both', right=True, left=False,
            labelright=True, labelleft=False
        )

    for i, ax in enumerate(fig.axes):
        if i in [1, 7, 13]:
            continue
        ax.set_yscale('log')
        ax.grid(alpha=0.2)

    return fig, fig.axes


def pixel_metrics_plot(distributions_dict, pixel_dist=None,
                       fig_ax=None, color=None, label=None):
    # Init figure and axes
    if fig_ax is None:
        fig, axs = init_pixel_metrics_plot()
    else:
        fig, axs = fig_ax

    # Assert pixel distribution is either in dict or passed
    assert pixel_dist is not None or 'Pixel_Intensity' in distributions_dict, (
        "Pixel distribution should be passed or be present in distributions_dict."
    )
    pixel_dist = pixel_dist or distributions_dict['Pixel_Intensity'][0]

    # Keys to plot
    keys = [
        'Bin_Pixels', 'Image_Mean', 'Bin_Mean',
        'Image_Sigma', 'Bin_Sigma'
    ]
    color = color if color is not None else "black"

    # Assert all keys are present in distributions_dict
    assert all(mask := [k in distributions_dict for k in keys]), (
        f"Not all keys are present in distributions_dict:"
        f" {compress(keys, ~mask)}"
    )

    # Plot pixel distribution first
    ax = axs[0]
    add_distribution_plot(
        pixel_dist, np.linspace(0, 1, len(pixel_dist) + 1), ax,
        label=label, color=color
    )
    ax.set_xlabel('Pixel Value')

    # Plot data & set axes properties
    i = 1
    for key in keys:
        if key.startswith('Bin_'):
            # Plot binned metrics
            sub_dict = distributions_dict[key]
            # Ghost axis for label
            axs[i].set_xlabel(key.replace('_', ' '), labelpad=20)
            i += 1
            for sub_key in sub_dict.keys():
                counts, edges = sub_dict[sub_key]
                add_distribution_plot(
                    counts, edges, axs[i],
                    label=str(sub_key), color=color
                )
                if 'Mean' in key:
                    axs[i].set_xlim(sub_key)
                i += 1
        else:
            counts, edges = distributions_dict[key]
            add_distribution_plot(
                counts, edges, axs[i],
                label=label, color=color
            )
            axs[i].set_xlabel(key.replace('_', ' '))
            i += 1

    for i, ax in enumerate(axs):
        if i in [1, 7, 13]:
            continue
        ax.legend()

    return fig, axs


def shape_metrics_plot(distributions_dict, COM=None,
                       fig_ax=None, color=None, label=None):
    if fig_ax is None:
        fig, axs = plt.subplots(
            3, 2, figsize=(16, 16), dpi=150, tight_layout=True
        )
    else:
        fig, axs = fig_ax

    # Assert COM is either in dict or passed
    assert COM is not None or 'COM' in distributions_dict, (
        "COM should be passed or be present in distributions_dict."
    )
    COM = COM or distributions_dict['COM'][0]

    # Keys to plot
    keys = [
        'WPCA_Angle', 'WPCA_Elongation', 'COM_Radius', 'COM_Angle', 'Scatter'
    ]
    color = color if color is not None else "black"

    for ax, key in zip(axs.flatten()[:-1], keys):
        counts, edges = distributions_dict[key]
        add_distribution_plot(
            counts, edges, ax,
            label=label, color=color
        )
        ax.set_xlabel(key.replace('_', ' '))

    # COM Distribution
    ax = axs[-1][-1]
    ax.set_aspect('equal')
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    ax.invert_yaxis()
    ax.set_xlabel('COM x')
    ax.set_ylabel('COM y')
    ax.set_title("COM Distribution Contours")
    # ax.scatter(*COM.T - 40, s=0.2, color=color, alpha=0.2, label=label)

    # New: Plot contours
    X = np.arange(-40, 40) + 0.5  # Should be centered on pixels
    counts_norm = norm(COM)
    cs = ax.contour(
        X, X, counts_norm, levels=[1e-4, 1e-3, 1e-2],
        colors=color, linewidths=0.5,
    )
    ax.clabel(cs, inline=True, fontsize=6, fmt='%.e')

    for ax in axs.flatten()[:-1]:
        if label is not None:
            ax.legend(fontsize='small')
        ax.grid(alpha=0.2)
        ax.set_yscale("log")
    # Put ticks on rhs for plots on the right
    for ax in axs[:, 1]:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

    return fig, axs


def metrics_plots(distr, labels=None, colors=None, cmap='cividis',
                  **plot_kwargs):

    out = []
    for plot_fnc in [pixel_metrics_plot, shape_metrics_plot]:
        match distr:
            case list():
                assert labels is None or type(labels) == list, (
                    "If distr is a list, labels should be a list as well."
                )
                assert colors is None or type(colors) == list, (
                    "If distr is a list, colors should be a list as well."
                )
                out.append(plot_collection(
                    distr, plot_fnc, labels=labels, colors=colors, cmap=cmap,
                    **plot_kwargs
                ))
            case dict():
                assert labels is None or type(labels) == str, (
                    "If distr is a dictionary, labels should be a string."
                )
                assert colors is None or type(colors) == str, (
                    "If distr is a dictionary, colors should be a string."
                )
                out.append(plot_fnc(
                    distr, fig_ax=None, color=colors, label=labels,
                    **plot_kwargs
                ))
    return out


def plot_distributions(distr_gen, distr_lofar, error,
                       title="Real vs. Generated Distributions", labels=None,
                       landscape_mode=False):
    # Set plot size and layout
    if not landscape_mode:  # Default
        rows = 6
        cols = 1
        figsize = (8, 16)
    else:
        rows = 2
        cols = 3
        figsize = (24, 6)

    # Create figure and axes
    fig, axs = plt.subplots(rows, cols, dpi=150, tight_layout=True,
                            figsize=figsize,
                            height_ratios=[5, 1] * int(rows / 2))

    # Loop through axes to plot distributions. Every 2nd axis is for the
    # per-bin delta between both distributions.
    for ax, ax2, key in zip(
        axs.transpose().flat[::2],  # Large plots for distributions
        axs.transpose().flat[1::2],  # Small plots for per-bin delta
        distr_lofar.keys()
    ):

        # Plot generated distributions.
        # distr_gen can be either a single dictionary or a list of
        # dictionaries.
        if isinstance(distr_gen, dict):  # Single dictionary
            c_gen, e_gen = distr_gen[key]
            add_distribution_plot(
                c_gen, e_gen, ax, color="tab:blue", alpha=0.5, fill=True,
                label=f"Generated"
            )
        elif isinstance(distr_gen, list):  # List of dictionaries
            assert all(isinstance(d, dict) for d in distr_gen), (
                "If distr_gen is a list, it should contain dictionaries."
            )
            # Set colors for each distribution
            cmap = 'cividis_r'
            colors = mpl.colormaps[cmap](np.linspace(0, 1, len(distr_gen)))

            # Plot each distribution
            for i, dg in enumerate(distr_gen):
                label = labels[i] if labels is not None else None
                c_gen, e_gen = dg[key]
                add_distribution_plot(
                    c_gen, e_gen, ax, color=colors[i], alpha=0.8, label=label
                )

        # Plot distributions of real lofar data
        c_lofar, e_lofar = distr_lofar[key]
        add_distribution_plot(
            c_lofar, e_lofar, ax, color="tab:orange", alpha=0.7,
            label=f"Real"
        )

        # Set properties for distributions plot
        # (x-axis limits, labels, log-scale, grid, legend)
        xmax = np.max(np.concatenate([
            e_gen[1:][c_gen > 0], e_lofar[1:][c_lofar > 0]
        ])).item()
        ax.set_xlim(left=0, right=xmax)
        ax.set_ylabel("Frequency")
        ax.set_xlabel(key)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend()

        # Plot per-bin relative delta between both distributions (thin plots)
        if isinstance(error, dict):  # Single dictionary
            add_distribution_delta_plot(error, ax2, key)

        elif isinstance(error, list):  # List of dictionaries
            assert all(isinstance(d, dict) for d in error), (
                "If error is a list, it should contain dictionaries."
            )
            # cmap and colors are defined above, where distributions are plotted
            for i, e in enumerate(error):
                add_distribution_delta_plot(e, ax2, key, color=colors[i])

        # Set properties for delta plot
        # (x-axis limits, labels, grid)
        ax2.set_ylabel(r"$2 \cdot \frac{Real-Gen}{Real+Gen}$")
        ax2.grid(alpha=0.3)
        ax2.set_xlim(left=0, right=xmax)
        ax2.set_ylim(-2.5, 2.5)

    # If necessary, remove y labels from all but the first column
    if landscape_mode:
        for ax in axs.transpose()[1:].flat:
            ax.set_ylabel(None)

    # Add title to figure
    fig.suptitle(f"{title}")

    return fig


def plot_W1_distances(W1_dict_list, x_values=None, x_label=''):
    n_trials = len(W1_dict_list)
    n_metrics = len(W1_dict_list[0].keys())

    # Set x values if not passed
    x_values = x_values or range(n_trials)

    # Set colors for each distribution
    cmap = 'tab20'
    colors = mpl.colormaps[cmap](np.linspace(0, 1, n_metrics))

    # Create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150,
                           tight_layout=True)

    # Loop through axes & keys to plot W1 values.
    # Get keys from first dictionary in list
    for i, key in enumerate(W1_dict_list[0].keys()):
        # Plot W1 values with error
        W1, W1_err = np.split(np.array([d[key] for d in W1_dict_list]).T, 2)
        ax.errorbar(
            x_values, W1.squeeze(), yerr=W1_err.squeeze(),
            color=colors[i], marker='.', label=key.replace('_', ' '),
            ls="--", elinewidth=0.5, capsize=2.5, capthick=0.5,
        )
        # Set properties for W1 plot
        ax.grid(alpha=0.3)

    # Set labels depending on layout
    ax.set_ylabel('W1 Distance')
    ax.set_xlabel(x_label)
    ax.legend()
    return fig, ax


def bdsf_metrics_plot(distributions_dict,
                      fig_ax=None, color=None, label=None
                      ):
    # Initialize figure and axes if not passed
    if fig_ax is None:
        fig, axs = plt.subplots(
            3, 2, figsize=(16, 16), dpi=150, tight_layout=True
        )
    else:
        fig, axs = fig_ax

    # Keys to plot
    keys = [
        'total_flux_gaus', 'ngaus', 'nsrc',
        'q0.9_area', 'q0.5_area', 'q0.1_area'
    ]
    color = color if color is not None else "black"

    for ax, key in zip(axs.T.flatten(), keys):
        counts, edges = distributions_dict[key]

        # Temporary: Remove last bin for area distributions,
        # because it is only filled by images with no gaussians.
        if 'area' in key:
            counts, edges = counts[:-1], edges[:-1]

        add_distribution_plot(
            counts, edges, ax,
            label=label, color=color, alpha=0.8,
        )
        ax.set_xlabel(key.replace('_', ' '))

    for ax in axs.flatten():
        if label is not None:
            ax.legend()
        ax.grid(alpha=0.2)
        ax.set_yscale("log")

    # Put ticks on rhs for plots on the right
    for ax in axs[:, 1]:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

    return fig, axs
